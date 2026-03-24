#include <volk.h>       // volk loads Vulkan function pointers for us.
#include <GLFW/glfw3.h> // GLFW creates the window and reads input.

#include <algorithm>    // std::clamp and std::find_if live here.
#include <array>        // std::array gives us fixed-size arrays.
#include <cmath>        // std::sin, std::cos, std::tan, std::sqrt, and std::fmod live here.
#include <cstdint>      // Fixed-width integer types like uint32_t live here.
#include <cstring>      // std::strcmp lives here.
#include <filesystem>   // std::filesystem::path lives here.
#include <fstream>      // std::ifstream lives here.
#include <iostream>     // std::cout and std::cerr live here.
#include <optional>     // std::optional lives here.
#include <set>          // std::set lives here.
#include <stdexcept>    // std::runtime_error lives here.
#include <string>       // std::string lives here.
#include <string_view>  // std::string_view lives here.
#include <vector>       // std::vector lives here.

namespace { // Everything in this file stays private to this translation unit.

// The width of the window in pixels.
constexpr int kWindowWidth = 1280;

// The height of the window in pixels.
constexpr int kWindowHeight = 720;

// We allow two frames to be prepared while the GPU is still finishing older work.
constexpr std::size_t kMaxFramesInFlight = 2;

// A hand-written value for pi because older C++ code often does this instead of using std::numbers.
constexpr float kPi = 3.1415926535f;

// How fast the camera moves through the world when a movement key is held down.
constexpr float kCameraMoveSpeed = 7.0f;

// How sensitive mouse movement should be when rotating the camera.
constexpr float kMouseLookSensitivity = 0.0025f;

// We do not allow the camera to look perfectly straight up or down because that can break the math.
constexpr float kMaxPitch = 89.0f * kPi / 180.0f;

// This is the starting horizontal rotation of the camera in radians.
constexpr float kInitialCameraYaw = -3.0f * kPi / 4.0f;

// This is the starting vertical rotation of the camera in radians.
constexpr float kInitialCameraPitch = -0.45277846f;

// These are the Vulkan validation layers we want when running debug builds.
constexpr std::array<const char*, 1> kValidationLayers = {"VK_LAYER_KHRONOS_validation"};

// These are device extensions that are required for this app to run.
constexpr std::array<const char*, 1> kDeviceExtensions = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};

// This tells the shaders which hardcoded mesh to use.
enum MeshType : int32_t {
    // A cube made from 12 triangles.
    kMeshCube = 0,

    // A flat square plane made from 2 triangles.
    kMeshPlane = 1,
};

// A very small 3D vector type for positions, directions, and movement.
struct Vec3 {
    // Horizontal axis.
    float x = 0.0f;

    // Vertical axis.
    float y = 0.0f;

    // Depth axis.
    float z = 0.0f;
};

// A very small 4D vector type for colors and packed transform data.
struct Vec4 {
    // First component.
    float x = 0.0f;

    // Second component.
    float y = 0.0f;

    // Third component.
    float z = 0.0f;

    // Fourth component. For colors this is alpha, and for transforms we use it as scale.
    float w = 1.0f;
};

// A 4x4 matrix stored as 16 floats in column-major order.
struct Mat4 {
    // The raw matrix data.
    std::array<float, 16> m{};
};

// This is the chunk of data we send to the shader for each object we draw.
struct alignas(16) PushConstants {
    // Combined camera matrix that turns world-space positions into clip-space positions.
    std::array<float, 16> viewProjection{};

    // Per-object color.
    std::array<float, 4> color{};

    // Per-object transform packed as position xyz + uniform scale w.
    std::array<float, 4> transform{};

    // Small integer parameters. We currently use params[0] for the mesh type.
    std::array<int32_t, 4> params{};
};

// Vulkan guarantees at least 128 bytes of push constants. We assert our struct fits.
static_assert(sizeof(PushConstants) <= 128, "Push constants must fit within Vulkan's guaranteed minimum.");

// This describes one thing we want to draw in the scene.
struct SceneObject {
    // Which hardcoded mesh the shader should use.
    MeshType meshType = kMeshCube;

    // Where the object lives in the world.
    Vec3 position{};

    // A single number that scales the whole object uniformly.
    float scale = 1.0f;

    // The object's base color.
    Vec4 color{};

    // How many vertices the draw call should emit.
    uint32_t vertexCount = 0;
};

// Keep the fire effect lightweight by using a fixed pool of particles.
constexpr std::size_t kFireParticleCount = 96;

// Each fire billboard is drawn as two triangles.
constexpr uint32_t kFireQuadVertexCount = 6;

// The emitter sits in the middle of the scene, slightly above the ground plane.
constexpr Vec3 kFireEmitterOrigin = {0.0f, 0.05f, 0.0f};

// One CPU-simulated flame particle.
struct FireParticle {
    // Current world-space position.
    Vec3 position{};

    // Current velocity used to move the particle upward and outward.
    Vec3 velocity{};

    // Seconds since the particle was spawned.
    float age = 0.0f;

    // Total lifetime in seconds before the particle respawns.
    float lifetime = 1.0f;

    // Billboard size in world units.
    float size = 0.25f;

    // Brightness multiplier used by the shader.
    float intensity = 1.0f;
};

// Push constants for the additive fire sprites.
struct alignas(16) FirePushConstants {
    // Combined camera matrix that turns world-space positions into clip-space positions.
    std::array<float, 16> viewProjection{};

    // Particle center xyz plus billboard size w.
    std::array<float, 4> positionAndSize{};

    // Camera right vector xyz plus remaining life ratio w.
    std::array<float, 4> cameraRightAndLife{};

    // Camera up vector xyz plus particle intensity w.
    std::array<float, 4> cameraUpAndIntensity{};
};

// The fire push constants also stay within Vulkan's guaranteed minimum size.
static_assert(sizeof(FirePushConstants) <= 128, "Fire push constants must fit within Vulkan's guaranteed minimum.");

// One oversized plane plus a few cubes is enough to prove out the basic 3D path:
// projection, depth testing, multiple draw calls, and per-object transforms.
constexpr std::array<SceneObject, 6> kSceneObjects = {{
    // The ground plane.
    {kMeshPlane, {0.0f, 0.0f, 0.0f}, 12.0f, {0.33f, 0.39f, 0.28f, 1.0f}, 6},

    // Cube 1.
    {kMeshCube, {-3.0f, 0.5f, -2.5f}, 1.0f, {0.92f, 0.37f, 0.28f, 1.0f}, 36},

    // Cube 2.
    {kMeshCube, {-0.7f, 0.5f, -1.0f}, 1.0f, {0.28f, 0.76f, 0.44f, 1.0f}, 36},

    // Cube 3.
    {kMeshCube, {1.6f, 0.5f, -2.2f}, 1.0f, {0.26f, 0.56f, 0.94f, 1.0f}, 36},

    // Cube 4.
    {kMeshCube, {-1.8f, 0.5f, 1.6f}, 1.0f, {0.90f, 0.74f, 0.22f, 1.0f}, 36},

    // Cube 5.
    {kMeshCube, {2.2f, 0.5f, 1.8f}, 1.0f, {0.74f, 0.42f, 0.92f, 1.0f}, 36},
}};

// Vulkan can use different queue families for graphics and presentation.
struct QueueFamilies {
    // Queue family index that can draw graphics commands.
    std::optional<uint32_t> graphics;

    // Queue family index that can present images to the window.
    std::optional<uint32_t> present;

    // This helper returns true only when we have both queues we need.
    [[nodiscard]] bool complete() const {
        return graphics.has_value() && present.has_value();
    }
};

// This bundles the swapchain-related capabilities a GPU reports for our surface.
struct SwapchainSupport {
    // Limits such as min/max image count and min/max size.
    VkSurfaceCapabilitiesKHR capabilities{};

    // Surface formats the GPU/window system supports.
    std::vector<VkSurfaceFormatKHR> formats;

    // Presentation modes the GPU/window system supports.
    std::vector<VkPresentModeKHR> presentModes;
};

// This struct holds almost all long-lived state for the app.
struct App {
    // The actual OS window created by GLFW.
    GLFWwindow* window = nullptr;

    // Top-level Vulkan instance.
    VkInstance instance = VK_NULL_HANDLE;

    // Optional debug callback object used in debug builds.
    VkDebugUtilsMessengerEXT debugMessenger = VK_NULL_HANDLE;

    // The connection between Vulkan and our window.
    VkSurfaceKHR surface = VK_NULL_HANDLE;

    // The chosen physical GPU.
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;

    // The logical device created from the chosen physical GPU.
    VkDevice device = VK_NULL_HANDLE;

    // Queue used for drawing commands.
    VkQueue graphicsQueue = VK_NULL_HANDLE;

    // Queue used for presenting finished images to the screen.
    VkQueue presentQueue = VK_NULL_HANDLE;

    // Swapchain object that owns the images shown on screen.
    VkSwapchainKHR swapchain = VK_NULL_HANDLE;

    // The raw images inside the swapchain.
    std::vector<VkImage> swapchainImages;

    // Image views that let us use those swapchain images as render targets.
    std::vector<VkImageView> swapchainImageViews;

    // The pixel format used by the swapchain images.
    VkFormat swapchainFormat = VK_FORMAT_UNDEFINED;

    // The actual width and height of the swapchain images.
    VkExtent2D swapchainExtent{};

    // The pixel format used by the depth buffer images.
    VkFormat depthFormat = VK_FORMAT_UNDEFINED;

    // One depth image per swapchain image.
    std::vector<VkImage> depthImages;

    // GPU memory backing those depth images.
    std::vector<VkDeviceMemory> depthMemories;

    // Views used to bind the depth images as depth attachments.
    std::vector<VkImageView> depthImageViews;

    // Render pass describing how color and depth are used during drawing.
    VkRenderPass renderPass = VK_NULL_HANDLE;

    // Pipeline layout describing which push constants the shaders expect.
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;

    // The compiled graphics pipeline state object.
    VkPipeline pipeline = VK_NULL_HANDLE;

    // Pipeline layout describing the push constants used by the fire billboards.
    VkPipelineLayout firePipelineLayout = VK_NULL_HANDLE;

    // The additive graphics pipeline used to draw the fire particles.
    VkPipeline firePipeline = VK_NULL_HANDLE;

    // One framebuffer per swapchain image.
    std::vector<VkFramebuffer> framebuffers;

    // Pool that allocates command buffers.
    VkCommandPool commandPool = VK_NULL_HANDLE;

    // One command buffer per framebuffer / swapchain image.
    std::vector<VkCommandBuffer> commandBuffers;

    // Semaphores used when an image becomes available to render into.
    std::array<VkSemaphore, kMaxFramesInFlight> imageAvailable{};

    // Fences used to know when the GPU has finished a frame.
    std::array<VkFence, kMaxFramesInFlight> inFlightFences{};

    // One render-finished semaphore per swapchain image to avoid reuse hazards.
    std::vector<VkSemaphore> renderFinishedPerImage;

    // Tracks which fence currently owns each swapchain image.
    std::vector<VkFence> imageInFlight;

    // Which CPU-side frame slot we are using right now.
    std::size_t currentFrame = 0;

    // Fixed-size particle pool used by the fire effect.
    std::array<FireParticle, kFireParticleCount> fireParticles{};

    // World-space origin of the fire emitter.
    Vec3 fireEmitterOrigin = kFireEmitterOrigin;

    // Monotonic counter used to vary respawn seeds.
    uint32_t fireSpawnCounter = 0;

    // Camera world position.
    Vec3 cameraPosition = {8.0f, 6.0f, 8.0f};

    // Camera left/right rotation in radians.
    float cameraYaw = kInitialCameraYaw;

    // Camera up/down rotation in radians.
    float cameraPitch = kInitialCameraPitch;

    // Whether the mouse is currently captured by the app.
    bool mouseCaptured = true;

    // Remembers last frame's Escape key state so Escape toggles only once per press.
    bool captureToggleKeyWasDown = false;

    // Helps us ignore the first mouse delta after capturing the cursor.
    bool firstMouseSample = true;

    // Previous mouse x position.
    double lastCursorX = 0.0;

    // Previous mouse y position.
    double lastCursorY = 0.0;

    // Previous frame time so we can compute delta time.
    double lastFrameTime = 0.0;

    // Whether validation layers are available and enabled.
    bool validationEnabled = false;
};

// Read a compiled SPIR-V shader file into memory.
[[nodiscard]] std::vector<char> ReadBinaryFile(const std::filesystem::path& path) {
    // Open the file in binary mode and start at the end so we can ask how big it is.
    std::ifstream file(path, std::ios::binary | std::ios::ate);

    // If the file could not be opened, fail immediately with a readable message.
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open " + path.string());
    }

    // Remember the file size so we know how many bytes to allocate.
    const auto size = static_cast<std::size_t>(file.tellg());

    // Allocate exactly enough memory to hold the file.
    std::vector<char> bytes(size);

    // Jump back to the beginning so we can read the file contents.
    file.seekg(0);

    // Only read when there is something to read.
    if (size > 0) {
        file.read(bytes.data(), static_cast<std::streamsize>(size));
    }

    // Return the loaded bytes.
    return bytes;
}

// Ignore a known broken Vulkan layer registration from TikTok LIVE Studio so the
// console only shows messages that matter for this sample.
[[nodiscard]] bool ShouldIgnoreDebugMessage(const std::string_view message) {
    // Check whether the loader is complaining about a missing JSON manifest file.
    const bool missingJson = message.find("loader_get_json: Failed to open JSON file") != std::string_view::npos;

    // Check whether that missing file belongs to TikTok LIVE Studio's stale Vulkan layer entry.
    const bool tiktokLayer = message.find("LiveStudioVulkanLayer") != std::string_view::npos ||
                             message.find("TikTok LIVE Studio") != std::string_view::npos;

    // Ignore only when both conditions are true.
    return missingJson && tiktokLayer;
}

// Vulkan calls this function whenever the validation or debug layer wants to report something.
VKAPI_ATTR VkBool32 VKAPI_CALL DebugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                                             VkDebugUtilsMessageTypeFlagsEXT,
                                             const VkDebugUtilsMessengerCallbackDataEXT* callbackData,
                                             void*) {
    // Safely pull the message text out of Vulkan's callback data.
    const std::string_view message =
        (callbackData != nullptr && callbackData->pMessage != nullptr) ? callbackData->pMessage : "";

    // Print warnings and errors unless they are the specific external noise we chose to ignore.
    if (!ShouldIgnoreDebugMessage(message) && severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
        std::cerr << "[Vulkan] " << message << '\n';
    }

    // Returning VK_FALSE tells Vulkan to continue normally.
    return VK_FALSE;
}

// Compute the dot product of two vectors.
[[nodiscard]] float Dot(const Vec3& a, const Vec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

// Compute the cross product of two vectors.
[[nodiscard]] Vec3 Cross(const Vec3& a, const Vec3& b) {
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    };
}

// Turn any non-zero vector into a unit-length vector.
[[nodiscard]] Vec3 Normalize(const Vec3& value) {
    // Compute the vector's length.
    const float length = std::sqrt(Dot(value, value));

    // Divide every component by the length.
    return {value.x / length, value.y / length, value.z / length};
}

// Add two vectors component by component.
[[nodiscard]] Vec3 Add(const Vec3& left, const Vec3& right) {
    return {left.x + right.x, left.y + right.y, left.z + right.z};
}

// Multiply each component of a vector by a single number.
[[nodiscard]] Vec3 Scale(const Vec3& value, const float scalar) {
    return {value.x * scalar, value.y * scalar, value.z * scalar};
}

// Return only the fractional part of a floating-point number.
[[nodiscard]] float Fract(const float value) {
    return value - std::floor(value);
}

// Small deterministic hash used to vary particle respawns without a random-number engine.
[[nodiscard]] float Hash01(const float seed) {
    return Fract(std::sin(seed) * 43758.5453123f);
}

// Build a 4x4 identity matrix, which is the "do nothing" matrix.
[[nodiscard]] Mat4 IdentityMatrix() {
    // Start with all zeros.
    Mat4 result{};

    // Put ones on the main diagonal.
    result.m[0] = 1.0f;
    result.m[5] = 1.0f;
    result.m[10] = 1.0f;
    result.m[15] = 1.0f;

    // Return the finished identity matrix.
    return result;
}

// Column-major matrix multiplication to match GLSL's default matrix layout.
[[nodiscard]] Mat4 Multiply(const Mat4& left, const Mat4& right) {
    // Create a new matrix to hold the answer.
    Mat4 result{};

    // Go through every column of the result matrix.
    for (int column = 0; column < 4; ++column) {
        // Go through every row of that column.
        for (int row = 0; row < 4; ++row) {
            // This will accumulate the sum for one cell in the matrix.
            float sum = 0.0f;

            // Multiply one row by one column, which is how matrix multiplication works.
            for (int inner = 0; inner < 4; ++inner) {
                sum += left.m[inner * 4 + row] * right.m[column * 4 + inner];
            }

            // Store the finished cell in the result matrix.
            result.m[column * 4 + row] = sum;
        }
    }

    // Return the combined matrix.
    return result;
}

// Right-handed look-at matrix. This builds the camera transform.
[[nodiscard]] Mat4 LookAt(const Vec3& eye, const Vec3& center, const Vec3& up) {
    // "forward" points from the camera position toward the target we want to look at.
    const Vec3 forward = Normalize({center.x - eye.x, center.y - eye.y, center.z - eye.z});

    // "side" points to the camera's right.
    const Vec3 side = Normalize(Cross(forward, up));

    // "actualUp" is the corrected up direction after the other two directions are chosen.
    const Vec3 actualUp = Cross(side, forward);

    // Start from an identity matrix before filling in camera basis vectors.
    Mat4 result = IdentityMatrix();

    // First column = right direction.
    result.m[0] = side.x;
    result.m[1] = actualUp.x;
    result.m[2] = -forward.x;

    // Second column = up direction.
    result.m[4] = side.y;
    result.m[5] = actualUp.y;
    result.m[6] = -forward.y;

    // Third column = backward direction because view matrices invert the camera transform.
    result.m[8] = side.z;
    result.m[9] = actualUp.z;
    result.m[10] = -forward.z;

    // Fourth column = translation that moves the world opposite the camera position.
    result.m[12] = -Dot(side, eye);
    result.m[13] = -Dot(actualUp, eye);
    result.m[14] = Dot(forward, eye);

    // Return the finished view matrix.
    return result;
}

// Perspective projection using Vulkan's 0..1 depth range. The negative Y entry
// keeps the image upright with a normal viewport.
[[nodiscard]] Mat4 Perspective(float fovRadians, float aspectRatio, float nearPlane, float farPlane) {
    // Start with a zero matrix.
    Mat4 result{};

    // tan(fov / 2) is part of the standard perspective projection formula.
    const float tanHalfFov = std::tan(fovRadians * 0.5f);

    // Scale X based on the screen shape and field of view.
    result.m[0] = 1.0f / (aspectRatio * tanHalfFov);

    // Scale Y based on field of view. It is negative because Vulkan's clip space is upside down compared to OpenGL.
    result.m[5] = -1.0f / tanHalfFov;

    // This maps depth into Vulkan's 0..1 range.
    result.m[10] = farPlane / (nearPlane - farPlane);

    // This enables perspective division.
    result.m[11] = -1.0f;

    // This controls how near and far depth values are remapped.
    result.m[14] = -(farPlane * nearPlane) / (farPlane - nearPlane);

    // Return the finished projection matrix.
    return result;
}

// Pack one object's draw data into the push-constant struct expected by the shader.
[[nodiscard]] PushConstants MakePushConstants(const Mat4& viewProjection, const SceneObject& object) {
    // Start with a zero-initialized struct.
    PushConstants constants{};

    // Copy the camera matrix into the push constants.
    constants.viewProjection = viewProjection.m;

    // Copy the object's color into the push constants.
    constants.color = {object.color.x, object.color.y, object.color.z, object.color.w};

    // Copy object position and scale into the push constants.
    constants.transform = {object.position.x, object.position.y, object.position.z, object.scale};

    // Tell the shader which mesh it should use for this draw call.
    constants.params = {static_cast<int32_t>(object.meshType), 0, 0, 0};

    // Return the finished push-constant packet.
    return constants;
}

// Advance one particle forward in time.
void AdvanceFireParticle(FireParticle& particle, const float deltaSeconds) {
    // Ignore zero or negative deltas.
    if (deltaSeconds <= 0.0f) {
        return;
    }

    // Age the particle and move it according to its velocity.
    particle.age += deltaSeconds;
    particle.position = Add(particle.position, Scale(particle.velocity, deltaSeconds));

    // Fire rises, so the particle gains a little upward velocity over time.
    particle.velocity.y += 0.35f * deltaSeconds;

    // Add small sideways noise so the flame flickers instead of moving in a perfectly straight line.
    particle.position.x += std::sin(particle.age * 13.0f + particle.intensity * 11.0f) * 0.18f * deltaSeconds;
    particle.position.z += std::cos(particle.age * 11.0f + particle.size * 17.0f) * 0.18f * deltaSeconds;

    // Let the billboard grow slightly as the particle rises.
    particle.size += 0.08f * deltaSeconds;
}

// Reset one particle back to the emitter with a new randomized direction and lifetime.
void RespawnFireParticle(App& app, FireParticle& particle, const std::size_t particleIndex) {
    const float seed = static_cast<float>(particleIndex) * 17.0f + static_cast<float>(app.fireSpawnCounter + 1) * 31.0f;
    app.fireSpawnCounter += 1;

    const float angle = Hash01(seed + 0.11f) * 2.0f * kPi;
    const float radius = std::sqrt(Hash01(seed + 1.37f)) * 0.28f;
    const float height = Hash01(seed + 2.61f) * 0.08f;
    const float outwardSpeed = 0.25f + Hash01(seed + 3.97f) * 0.35f;

    particle.position = {
        app.fireEmitterOrigin.x + std::cos(angle) * radius,
        app.fireEmitterOrigin.y + height,
        app.fireEmitterOrigin.z + std::sin(angle) * radius,
    };

    particle.velocity = {
        std::cos(angle) * outwardSpeed * 0.35f,
        1.2f + Hash01(seed + 4.53f) * 1.4f,
        std::sin(angle) * outwardSpeed * 0.35f,
    };

    particle.age = 0.0f;
    particle.lifetime = 0.9f + Hash01(seed + 5.73f) * 0.9f;
    particle.size = 0.18f + Hash01(seed + 6.91f) * 0.22f;
    particle.intensity = 0.55f + Hash01(seed + 7.67f) * 0.45f;
}

// Seed the particle pool once so the fire starts already filled in.
void InitializeFireParticles(App& app) {
    for (std::size_t i = 0; i < app.fireParticles.size(); ++i) {
        RespawnFireParticle(app, app.fireParticles[i], i);

        // Warm each particle partway through its life so the first frame already looks like a flame.
        const float warmupSeconds =
            Hash01(static_cast<float>(i) * 9.13f + 41.0f) * app.fireParticles[i].lifetime * 0.75f;
        AdvanceFireParticle(app.fireParticles[i], warmupSeconds);
    }
}

// Step every fire particle and respawn expired ones.
void UpdateFireParticles(App& app, const float deltaSeconds) {
    for (std::size_t i = 0; i < app.fireParticles.size(); ++i) {
        AdvanceFireParticle(app.fireParticles[i], deltaSeconds);

        if (app.fireParticles[i].age >= app.fireParticles[i].lifetime) {
            RespawnFireParticle(app, app.fireParticles[i], i);
        }
    }
}

// Pack one fire particle into the push constants expected by the fire shaders.
[[nodiscard]] FirePushConstants MakeFirePushConstants(const Mat4& viewProjection,
                                                      const FireParticle& particle,
                                                      const Vec3& cameraRight,
                                                      const Vec3& cameraUp) {
    FirePushConstants constants{};
    const float lifeRemaining = std::clamp(1.0f - particle.age / particle.lifetime, 0.0f, 1.0f);

    constants.viewProjection = viewProjection.m;
    constants.positionAndSize = {particle.position.x, particle.position.y, particle.position.z, particle.size};
    constants.cameraRightAndLife = {cameraRight.x, cameraRight.y, cameraRight.z, lifeRemaining};
    constants.cameraUpAndIntensity = {cameraUp.x, cameraUp.y, cameraUp.z, std::clamp(particle.intensity, 0.0f, 1.0f)};
    return constants;
}

// Change whether the app owns the mouse cursor.
void SetMouseCapture(App& app, const bool captured) {
    // Remember the new capture state.
    app.mouseCaptured = captured;

    // The first mouse sample after a capture change should not rotate the camera wildly.
    app.firstMouseSample = true;

    // Tell GLFW whether the cursor should be locked inside the window or shown normally.
    glfwSetInputMode(app.window, GLFW_CURSOR, captured ? GLFW_CURSOR_DISABLED : GLFW_CURSOR_NORMAL);

    // If the platform supports raw mouse motion, enable it for smoother mouse look.
    if (glfwRawMouseMotionSupported()) {
        glfwSetInputMode(app.window, GLFW_RAW_MOUSE_MOTION, captured ? GLFW_TRUE : GLFW_FALSE);
    }
}

// Turn the camera's yaw and pitch angles into a normalized forward direction vector.
[[nodiscard]] Vec3 BuildCameraForward(const App& app) {
    return Normalize({
        std::cos(app.cameraPitch) * std::cos(app.cameraYaw),
        std::sin(app.cameraPitch),
        std::cos(app.cameraPitch) * std::sin(app.cameraYaw),
    });
}

// Read keyboard + mouse input, update the free-fly camera, and return frame delta time.
[[nodiscard]] float UpdateFreeCamera(App& app) {
    // Check whether Escape is currently pressed.
    const bool escapeKeyDown = glfwGetKey(app.window, GLFW_KEY_ESCAPE) == GLFW_PRESS;
    std::cout << "Bool pentru escape:" << escapeKeyDown << "\n";

    // Toggle mouse capture only once when Escape changes from "up" to "down".
    if (escapeKeyDown && !app.captureToggleKeyWasDown) {
        SetMouseCapture(app, !app.mouseCaptured);
    }

    // Remember Escape's state so we can detect the next key press edge.
    app.captureToggleKeyWasDown = escapeKeyDown;

    // When the cursor is free, allow a left click to capture it again.
    if (!app.mouseCaptured && glfwGetMouseButton(app.window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) {
        SetMouseCapture(app, true);
    }

    // Ask GLFW what time it is right now.
    const double now = glfwGetTime();

    // Compute how much time passed since the previous frame.
    const double deltaSeconds = now - app.lastFrameTime;

    // Store the current time so the next frame can compute its own delta.
    app.lastFrameTime = now;

    // Only rotate the camera with the mouse while the cursor is captured.
    if (app.mouseCaptured) {
        // These will hold the new mouse position.
        double cursorX = 0.0;
        double cursorY = 0.0;

        // Ask GLFW where the mouse currently is.
        glfwGetCursorPos(app.window, &cursorX, &cursorY);

        // The first sample after capture is used only to initialize our "previous mouse position".
        if (app.firstMouseSample) {
            app.lastCursorX = cursorX;
            app.lastCursorY = cursorY;
            app.firstMouseSample = false;
        } else {
            // Compute how far the mouse moved since the previous frame.
            const float deltaX = static_cast<float>(cursorX - app.lastCursorX);
            const float deltaY = static_cast<float>(cursorY - app.lastCursorY);

            // Update our saved mouse position for the next frame.
            app.lastCursorX = cursorX;
            app.lastCursorY = cursorY;

            // Moving the mouse left/right changes yaw.
            app.cameraYaw += deltaX * kMouseLookSensitivity;

            // Moving the mouse up/down changes pitch. We subtract because screen Y grows downward.
            app.cameraPitch -= deltaY * kMouseLookSensitivity;

            // Clamp pitch so the camera never flips over.
            app.cameraPitch = std::clamp(app.cameraPitch, -kMaxPitch, kMaxPitch);
        }
    }

    // Rebuild the forward direction from the current yaw and pitch.
    const Vec3 forward = BuildCameraForward(app);

    // This is the world's "up" direction.
    const Vec3 worldUp = {0.0f, 1.0f, 0.0f};

    // The camera's right direction is perpendicular to forward and up.
    const Vec3 right = Normalize(Cross(forward, worldUp));

    // Start with no movement. Keys will add directions into this vector.
    Vec3 movement{};

    // W moves forward.
    if (glfwGetKey(app.window, GLFW_KEY_W) == GLFW_PRESS) {
        movement = Add(movement, forward);
    }

    // S moves backward.
    if (glfwGetKey(app.window, GLFW_KEY_S) == GLFW_PRESS) {
        movement = Add(movement, Scale(forward, -1.0f));
    }

    // D strafes right.
    if (glfwGetKey(app.window, GLFW_KEY_D) == GLFW_PRESS) {
        movement = Add(movement, right);
    }

    // A strafes left.
    if (glfwGetKey(app.window, GLFW_KEY_A) == GLFW_PRESS) {
        movement = Add(movement, Scale(right, -1.0f));
    }

    // Space moves upward.
    if (glfwGetKey(app.window, GLFW_KEY_SPACE) == GLFW_PRESS) {
        movement = Add(movement, worldUp);
    }

    // Shift moves downward.
    if (glfwGetKey(app.window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
        glfwGetKey(app.window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS) {
        movement = Add(movement, Scale(worldUp, -1.0f));
    }

    // Only move when at least one movement key contributed something.
    if (Dot(movement, movement) > 0.0f) {
        // Normalize so diagonal movement is not faster, then scale by speed and frame time.
        app.cameraPosition = Add(app.cameraPosition,
                                 Scale(Normalize(movement), kCameraMoveSpeed * static_cast<float>(deltaSeconds)));
    }

    // Return the frame delta so other animated systems can stay in sync with the camera update.
    return static_cast<float>(deltaSeconds);
}

// Build the combined camera matrix used by the vertex shader.
[[nodiscard]] Mat4 BuildViewProjection(const App& app) {
    // Get the camera's forward direction.
    const Vec3 forward = BuildCameraForward(app);

    // Build the view matrix from the camera position and the point one unit in front of it.
    const Mat4 view = LookAt(app.cameraPosition, Add(app.cameraPosition, forward), {0.0f, 1.0f, 0.0f});

    // Build the projection matrix from field of view, aspect ratio, and near/far clip planes.
    const Mat4 projection = Perspective(45.0f * kPi / 180.0f,
                                        static_cast<float>(app.swapchainExtent.width) /
                                            static_cast<float>(app.swapchainExtent.height),
                                        0.1f,
                                        100.0f);

    // Multiply them together so the shader gets one matrix instead of two.
    return Multiply(projection, view);
}

// Ask Vulkan whether the validation layers we want are actually available on this machine.
[[nodiscard]] bool CheckValidationLayerSupport() {
    // Vulkan writes the number of available layers into this variable.
    uint32_t layerCount = 0;

    // First call asks only for the count.
    vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

    // Allocate enough space to hold every layer description.
    std::vector<VkLayerProperties> layers(layerCount);

    // Second call fills in the actual layer descriptions.
    vkEnumerateInstanceLayerProperties(&layerCount, layers.data());

    // Go through every layer we require.
    for (const char* required : kValidationLayers) {
        // Search the available layer list for a matching name.
        const auto it = std::find_if(layers.begin(), layers.end(), [required](const VkLayerProperties& layer) {
            return std::strcmp(layer.layerName, required) == 0;
        });

        // If even one required layer is missing, validation is not fully supported.
        if (it == layers.end()) {
            return false;
        }
    }

    // All required layers were found.
    return true;
}

// Build the list of Vulkan instance extensions required by GLFW, plus debug tools when needed.
[[nodiscard]] std::vector<const char*> GetRequiredExtensions(const bool validationEnabled) {
    // GLFW writes the number of required extensions here.
    uint32_t count = 0;

    // GLFW returns a pointer to an array of extension names.
    const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&count);

    // Copy those extension names into a std::vector for easier editing.
    std::vector<const char*> extensions(glfwExtensions, glfwExtensions + count);

    // When validation is on, we also need the debug messenger extension.
    if (validationEnabled) {
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    // Return the final extension list.
    return extensions;
}

// Find queue families on a GPU that can draw graphics and present to our window.
[[nodiscard]] QueueFamilies FindQueueFamilies(const App& app, const VkPhysicalDevice device) {
    // Start with an empty answer.
    QueueFamilies families;

    // Vulkan writes the number of queue families into this variable.
    uint32_t queueCount = 0;

    // First call asks only for the count.
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueCount, nullptr);

    // Allocate room for every queue-family description.
    std::vector<VkQueueFamilyProperties> queues(queueCount);

    // Second call fills in the queue-family descriptions.
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueCount, queues.data());

    // Walk through every queue family the GPU exposes.
    for (uint32_t i = 0; i < queueCount; ++i) {
        // If this family supports graphics commands, remember it.
        if ((queues[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0) {
            families.graphics = i;
        }

        // Ask Vulkan whether this queue family can present images to our window surface.
        VkBool32 supportsPresent = VK_FALSE;
        vkGetPhysicalDeviceSurfaceSupportKHR(device, i, app.surface, &supportsPresent);

        // If it can present, remember it.
        if (supportsPresent == VK_TRUE) {
            families.present = i;
        }

        // Stop early once we have everything we need.
        if (families.complete()) {
            break;
        }
    }

    // Return the queue-family selection.
    return families;
}

// Check whether the GPU supports all required device extensions.
[[nodiscard]] bool CheckDeviceExtensions(const VkPhysicalDevice device) {
    // Vulkan writes the number of extension descriptions here.
    uint32_t extensionCount = 0;

    // First call asks only for the count.
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);

    // Allocate enough room to hold every extension description.
    std::vector<VkExtensionProperties> extensions(extensionCount);

    // Second call fills in those extension descriptions.
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, extensions.data());

    // Put every extension we require into a set so we can erase matches as we find them.
    std::set<std::string_view> required(kDeviceExtensions.begin(), kDeviceExtensions.end());

    // Remove each supported extension from the required set.
    for (const auto& extension : extensions) {
        required.erase(extension.extensionName);
    }

    // If the set is empty, the device supports everything we need.
    return required.empty();
}

// Ask the GPU what swapchain-related capabilities it supports for our window surface.
[[nodiscard]] SwapchainSupport QuerySwapchainSupport(const App& app, const VkPhysicalDevice device) {
    // Start with an empty answer object.
    SwapchainSupport support;

    // Fill in global swapchain rules such as image-count limits and current transform.
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, app.surface, &support.capabilities);

    // Ask how many surface formats the GPU supports for this surface.
    uint32_t formatCount = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(device, app.surface, &formatCount, nullptr);

    // If at least one format exists, resize the vector and fetch them.
    if (formatCount > 0) {
        support.formats.resize(formatCount);
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, app.surface, &formatCount, support.formats.data());
    }

    // Ask how many presentation modes the GPU supports for this surface.
    uint32_t presentModeCount = 0;
    vkGetPhysicalDeviceSurfacePresentModesKHR(device, app.surface, &presentModeCount, nullptr);

    // If at least one present mode exists, resize the vector and fetch them.
    if (presentModeCount > 0) {
        support.presentModes.resize(presentModeCount);
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, app.surface, &presentModeCount, support.presentModes.data());
    }

    // Return the gathered swapchain support information.
    return support;
}

// Decide whether a physical GPU is good enough to run this sample.
[[nodiscard]] bool IsDeviceSuitable(const App& app, const VkPhysicalDevice device) {
    // The device must have the queue families we need.
    const QueueFamilies families = FindQueueFamilies(app, device);
    if (!families.complete() || !CheckDeviceExtensions(device)) {
        return false;
    }

    // The device must also support at least one swapchain format and one present mode.
    const SwapchainSupport swapchainSupport = QuerySwapchainSupport(app, device);
    return !swapchainSupport.formats.empty() && !swapchainSupport.presentModes.empty();
}

// Choose the best surface format we can find.
[[nodiscard]] VkSurfaceFormatKHR ChooseSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& formats) {
    // Prefer SRGB so colors look correct on screen.
    for (const auto& format : formats) {
        if (format.format == VK_FORMAT_B8G8R8A8_SRGB &&
            format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            return format;
        }
    }

    // Fall back to the first supported format if the ideal one is unavailable.
    return formats.front();
}

// Choose the best present mode we can find.
[[nodiscard]] VkPresentModeKHR ChoosePresentMode(const std::vector<VkPresentModeKHR>& presentModes) {
    // Prefer MAILBOX for lower latency when it exists.
    for (const auto mode : presentModes) {
        if (mode == VK_PRESENT_MODE_MAILBOX_KHR) {
            return mode;
        }
    }

    // FIFO is always supported and is the safe fallback.
    return VK_PRESENT_MODE_FIFO_KHR;
}

// Choose the swapchain image size we will render into.
[[nodiscard]] VkExtent2D ChooseExtent(const SwapchainSupport& support) {
    // Some platforms force the current extent, and if they do, we must use it.
    if (support.capabilities.currentExtent.width != UINT32_MAX) {
        return support.capabilities.currentExtent;
    }

    // Otherwise, clamp our preferred window size into the range allowed by the surface.
    VkExtent2D extent{};
    extent.width = std::clamp(static_cast<uint32_t>(kWindowWidth),
                              support.capabilities.minImageExtent.width,
                              support.capabilities.maxImageExtent.width);
    extent.height = std::clamp(static_cast<uint32_t>(kWindowHeight),
                               support.capabilities.minImageExtent.height,
                               support.capabilities.maxImageExtent.height);

    // Return the chosen extent.
    return extent;
}

// Find a memory type index that satisfies both the resource and property requirements.
[[nodiscard]] uint32_t FindMemoryType(const App& app, const uint32_t typeFilter, const VkMemoryPropertyFlags properties) {
    // Ask the GPU what memory types exist.
    VkPhysicalDeviceMemoryProperties memoryProperties{};
    vkGetPhysicalDeviceMemoryProperties(app.physicalDevice, &memoryProperties);

    // Search every memory type for one that is both allowed and has the flags we want.
    for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; ++i) {
        const bool supportedByResource = (typeFilter & (1u << i)) != 0;
        const bool hasFlags = (memoryProperties.memoryTypes[i].propertyFlags & properties) == properties;
        if (supportedByResource && hasFlags) {
            return i;
        }
    }

    throw std::runtime_error("Failed to find suitable memory type.");
}

// Some depth formats also store stencil values. This helper tells us when that is true.
[[nodiscard]] bool HasStencilComponent(const VkFormat format) {
    return format == VK_FORMAT_D32_SFLOAT_S8_UINT || format == VK_FORMAT_D24_UNORM_S8_UINT;
}

// Try a list of candidate formats and return the first one that supports the requested feature flags.
[[nodiscard]] VkFormat FindSupportedFormat(const App& app,
                                           const std::vector<VkFormat>& candidates,
                                           const VkImageTiling tiling,
                                           const VkFormatFeatureFlags features) {
    // Test each candidate one by one.
    for (const VkFormat format : candidates) {
        VkFormatProperties properties{};
        vkGetPhysicalDeviceFormatProperties(app.physicalDevice, format, &properties);

        // Check whether the format supports the required features for the chosen tiling mode.
        const bool supportsLinear =
            tiling == VK_IMAGE_TILING_LINEAR && (properties.linearTilingFeatures & features) == features;
        const bool supportsOptimal =
            tiling == VK_IMAGE_TILING_OPTIMAL && (properties.optimalTilingFeatures & features) == features;

        if (supportsLinear || supportsOptimal) {
            return format;
        }
    }

    throw std::runtime_error("Failed to find supported format.");
}

// Choose a depth format that can be used as a depth attachment.
[[nodiscard]] VkFormat FindDepthFormat(const App& app) {
    return FindSupportedFormat(app,
                               {VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT},
                               VK_IMAGE_TILING_OPTIMAL,
                               VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);
}

// Create a 2D image and the GPU memory backing it.
void CreateImage(App& app,
                 const uint32_t width,
                 const uint32_t height,
                 const VkFormat format,
                 const VkImageUsageFlags usage,
                 const VkMemoryPropertyFlags properties,
                 VkImage& image,
                 VkDeviceMemory& memory) {
    // Describe the image we want Vulkan to create.
    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent.width = width;
    imageInfo.extent.height = height;
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.format = format;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage = usage;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    // Ask Vulkan to create the image object.
    if (vkCreateImage(app.device, &imageInfo, nullptr, &image) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create image.");
    }

    // Ask Vulkan how much memory the image needs.
    VkMemoryRequirements memoryRequirements{};
    vkGetImageMemoryRequirements(app.device, image, &memoryRequirements);

    // Describe the memory allocation we want to make for the image.
    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memoryRequirements.size;
    allocInfo.memoryTypeIndex = FindMemoryType(app, memoryRequirements.memoryTypeBits, properties);

    // Ask Vulkan to allocate the memory block.
    if (vkAllocateMemory(app.device, &allocInfo, nullptr, &memory) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate image memory.");
    }

    // Bind the allocated memory to the image object.
    vkBindImageMemory(app.device, image, memory, 0);
}

// Create an image view so Vulkan knows how to interpret an image.
[[nodiscard]] VkImageView CreateImageView(const App& app,
                                          const VkImage image,
                                          const VkFormat format,
                                          VkImageAspectFlags aspectMask) {
    // If the format includes stencil data, include the stencil aspect too.
    if (HasStencilComponent(format)) {
        aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
    }

    // Describe how the image should be viewed.
    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = format;
    viewInfo.subresourceRange.aspectMask = aspectMask;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;

    // This variable will receive the created image-view handle.
    VkImageView imageView = VK_NULL_HANDLE;

    // Ask Vulkan to create the image view.
    if (vkCreateImageView(app.device, &viewInfo, nullptr, &imageView) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create image view.");
    }

    // Return the created image view.
    return imageView;
}

// Create the actual desktop window.
void CreateWindow(App& app) {
    // Initialize GLFW before using any of its features.
    if (glfwInit() != GLFW_TRUE) {
        throw std::runtime_error("Failed to initialize GLFW.");
    }

    // Tell GLFW we are not creating an OpenGL context.
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    // Keep resizing off to make the sample simpler.
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

    // Create the window with our chosen size and title.
    app.window = glfwCreateWindow(kWindowWidth, kWindowHeight, "Vulkan Cubes", nullptr, nullptr);

    // Fail immediately if the window was not created.
    if (app.window == nullptr) {
        throw std::runtime_error("Failed to create window.");
    }

    // Start in captured mode so mouse movement immediately controls the camera.
    glfwSetInputMode(app.window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    if (glfwRawMouseMotionSupported()) {
        glfwSetInputMode(app.window, GLFW_RAW_MOUSE_MOTION, GLFW_TRUE);
    }
}

// Create the top-level Vulkan instance.
void CreateInstance(App& app) {
    // GLFW can tell us whether the Vulkan loader is available on this system.
    if (!glfwVulkanSupported()) {
        throw std::runtime_error("GLFW could not find the Vulkan loader.");
    }

#ifndef NDEBUG
    app.validationEnabled = CheckValidationLayerSupport();
#endif

    // Fill in general metadata about the program.
    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "Vulkan Cubes";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "None";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_1;

    // Build the extension list the instance needs.
    const auto extensions = GetRequiredExtensions(app.validationEnabled);

    // Describe how we want the instance created.
    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;
    createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    createInfo.ppEnabledExtensionNames = extensions.data();

    // When validation is on, we also prepare the debug messenger during instance creation.
    VkDebugUtilsMessengerCreateInfoEXT debugInfo{};
    if (app.validationEnabled) {
        createInfo.enabledLayerCount = static_cast<uint32_t>(kValidationLayers.size());
        createInfo.ppEnabledLayerNames = kValidationLayers.data();

        debugInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        debugInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                                    VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        debugInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                                VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                                VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        debugInfo.pfnUserCallback = DebugCallback;

        // Point pNext at the debug create info so early validation messages can be reported too.
        createInfo.pNext = &debugInfo;
    }

    // Ask Vulkan to create the instance.
    if (vkCreateInstance(&createInfo, nullptr, &app.instance) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create Vulkan instance.");
    }
}

// Create the debug messenger that prints validation warnings and errors.
void CreateDebugMessenger(App& app) {
    // Skip this step when validation is not enabled.
    if (!app.validationEnabled) {
        return;
    }

    // Describe the kinds of messages we want to receive.
    VkDebugUtilsMessengerCreateInfoEXT createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                                 VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                             VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                             VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    createInfo.pfnUserCallback = DebugCallback;

    // Ask Vulkan to create the debug messenger.
    if (vkCreateDebugUtilsMessengerEXT(app.instance, &createInfo, nullptr, &app.debugMessenger) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create debug messenger.");
    }
}

// Create the surface that connects Vulkan to the GLFW window.
void CreateSurface(App& app) {
    if (glfwCreateWindowSurface(app.instance, app.window, nullptr, &app.surface) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create window surface.");
    }
}

// Pick the first GPU that satisfies all of our requirements.
void PickPhysicalDevice(App& app) {
    // Vulkan writes the number of GPUs here.
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(app.instance, &deviceCount, nullptr);

    // If there are no Vulkan-capable GPUs, the sample cannot run.
    if (deviceCount == 0) {
        throw std::runtime_error("No Vulkan-capable GPU found.");
    }

    // Allocate room for every GPU handle.
    std::vector<VkPhysicalDevice> devices(deviceCount);

    // Fetch the GPU handles.
    vkEnumeratePhysicalDevices(app.instance, &deviceCount, devices.data());

    // Use the first suitable GPU we find.
    for (const auto device : devices) {
        if (IsDeviceSuitable(app, device)) {
            app.physicalDevice = device;
            return;
        }
    }

    // If we get here, none of the GPUs met our needs.
    throw std::runtime_error("No suitable Vulkan GPU found.");
}

// Create the logical device and fetch the queue handles we will use.
void CreateDevice(App& app) {
    // Find the queue families we need on the chosen GPU.
    const QueueFamilies families = FindQueueFamilies(app, app.physicalDevice);

    // Put them in a set so we do not accidentally request the same family twice.
    std::set<uint32_t> uniqueFamilies = {families.graphics.value(), families.present.value()};

    // Queue priority is a number between 0 and 1.
    constexpr float priority = 1.0f;
    std::vector<VkDeviceQueueCreateInfo> queueInfos;

    // Build one queue-create description per unique family.
    for (const auto family : uniqueFamilies) {
        VkDeviceQueueCreateInfo queueInfo{};
        queueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueInfo.queueFamilyIndex = family;
        queueInfo.queueCount = 1;
        queueInfo.pQueuePriorities = &priority;
        queueInfos.push_back(queueInfo);
    }

    // We are not enabling any optional physical-device features yet.
    VkPhysicalDeviceFeatures features{};

    // Describe how we want the logical device created.
    VkDeviceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueInfos.size());
    createInfo.pQueueCreateInfos = queueInfos.data();
    createInfo.pEnabledFeatures = &features;
    createInfo.enabledExtensionCount = static_cast<uint32_t>(kDeviceExtensions.size());
    createInfo.ppEnabledExtensionNames = kDeviceExtensions.data();

    // When validation is enabled, attach the validation layers to the device too.
    if (app.validationEnabled) {
        createInfo.enabledLayerCount = static_cast<uint32_t>(kValidationLayers.size());
        createInfo.ppEnabledLayerNames = kValidationLayers.data();
    }

    // Ask Vulkan to create the logical device.
    if (vkCreateDevice(app.physicalDevice, &createInfo, nullptr, &app.device) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create logical device.");
    }

    // Fetch the actual queue handles from the new logical device.
    vkGetDeviceQueue(app.device, families.graphics.value(), 0, &app.graphicsQueue);
    vkGetDeviceQueue(app.device, families.present.value(), 0, &app.presentQueue);
}

// Create the swapchain, which owns the images that will eventually appear on screen.
void CreateSwapchain(App& app) {
    // Ask the GPU what swapchain choices it supports.
    const SwapchainSupport support = QuerySwapchainSupport(app, app.physicalDevice);

    // Choose the format, present mode, and size we want to use.
    const VkSurfaceFormatKHR surfaceFormat = ChooseSurfaceFormat(support.formats);
    const VkPresentModeKHR presentMode = ChoosePresentMode(support.presentModes);
    const VkExtent2D extent = ChooseExtent(support);

    // Ask for one more image than the minimum so the GPU can work more smoothly.
    uint32_t imageCount = support.capabilities.minImageCount + 1;

    // Clamp that request if the surface also reports a maximum.
    if (support.capabilities.maxImageCount > 0 && imageCount > support.capabilities.maxImageCount) {
        imageCount = support.capabilities.maxImageCount;
    }

    // We need the queue-family indices again to choose image sharing mode.
    const QueueFamilies families = FindQueueFamilies(app, app.physicalDevice);
    const uint32_t familyIndices[] = {families.graphics.value(), families.present.value()};

    // Describe the swapchain we want.
    VkSwapchainCreateInfoKHR createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    createInfo.surface = app.surface;
    createInfo.minImageCount = imageCount;
    createInfo.imageFormat = surfaceFormat.format;
    createInfo.imageColorSpace = surfaceFormat.colorSpace;
    createInfo.imageExtent = extent;
    createInfo.imageArrayLayers = 1;
    createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

    // If different queue families need the images, let both families access them directly.
    if (families.graphics != families.present) {
        createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
        createInfo.queueFamilyIndexCount = 2;
        createInfo.pQueueFamilyIndices = familyIndices;
    } else {
        // Otherwise, exclusive mode is simpler and usually a little faster.
        createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    }

    // Use the surface's current transform, opaque blending, and the chosen present mode.
    createInfo.preTransform = support.capabilities.currentTransform;
    createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    createInfo.presentMode = presentMode;
    createInfo.clipped = VK_TRUE;

    // Ask Vulkan to create the swapchain.
    if (vkCreateSwapchainKHR(app.device, &createInfo, nullptr, &app.swapchain) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create swapchain.");
    }

    // Ask how many images were actually created.
    vkGetSwapchainImagesKHR(app.device, app.swapchain, &imageCount, nullptr);

    // Resize the vector so it can hold every swapchain image.
    app.swapchainImages.resize(imageCount);

    // Fetch the swapchain image handles.
    vkGetSwapchainImagesKHR(app.device, app.swapchain, &imageCount, app.swapchainImages.data());

    // Remember the format and extent for later use.
    app.swapchainFormat = surfaceFormat.format;
    app.swapchainExtent = extent;
}

// Create one image view for each swapchain image.
void CreateSwapchainImageViews(App& app) {
    // Prepare one slot per swapchain image.
    app.swapchainImageViews.resize(app.swapchainImages.size());

    // Build one image view per image.
    for (std::size_t i = 0; i < app.swapchainImages.size(); ++i) {
        app.swapchainImageViews[i] =
            CreateImageView(app, app.swapchainImages[i], app.swapchainFormat, VK_IMAGE_ASPECT_COLOR_BIT);
    }
}

// Create one depth image and depth image view per swapchain image.
void CreateDepthResources(App& app) {
    // Pick a depth format the current GPU supports.
    app.depthFormat = FindDepthFormat(app);

    // Prepare storage for the per-image depth resources.
    app.depthImages.resize(app.swapchainImages.size(), VK_NULL_HANDLE);
    app.depthMemories.resize(app.swapchainImages.size(), VK_NULL_HANDLE);
    app.depthImageViews.resize(app.swapchainImages.size(), VK_NULL_HANDLE);

    // We keep one depth image per swapchain image so each framebuffer owns a
    // matching color+depth attachment pair.
    for (std::size_t i = 0; i < app.swapchainImages.size(); ++i) {
        CreateImage(app,
                    app.swapchainExtent.width,
                    app.swapchainExtent.height,
                    app.depthFormat,
                    VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                    app.depthImages[i],
                    app.depthMemories[i]);

        app.depthImageViews[i] =
            CreateImageView(app, app.depthImages[i], app.depthFormat, VK_IMAGE_ASPECT_DEPTH_BIT);
    }
}

// Create the render pass that describes our color + depth attachments.
void CreateRenderPass(App& app) {
    // Describe the color attachment, which is the swapchain image we present to the screen.
    VkAttachmentDescription colorAttachment{};
    colorAttachment.format = app.swapchainFormat;
    colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    // Describe the depth attachment, which stores depth values during rasterization.
    VkAttachmentDescription depthAttachment{};
    depthAttachment.format = app.depthFormat;
    depthAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    // Tell Vulkan that subpass attachment 0 is the color attachment.
    VkAttachmentReference colorAttachmentRef{};
    colorAttachmentRef.attachment = 0;
    colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    // Tell Vulkan that subpass attachment 1 is the depth attachment.
    VkAttachmentReference depthAttachmentRef{};
    depthAttachmentRef.attachment = 1;
    depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    // Describe the single subpass that will use those attachments.
    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorAttachmentRef;
    subpass.pDepthStencilAttachment = &depthAttachmentRef;

    // Describe an external dependency so Vulkan knows how to synchronize into our subpass.
    VkSubpassDependency dependency{};
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

    // Put both attachment descriptions into a fixed-size array.
    const std::array<VkAttachmentDescription, 2> attachments = {colorAttachment, depthAttachment};

    // Describe the render pass as a whole.
    VkRenderPassCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    createInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
    createInfo.pAttachments = attachments.data();
    createInfo.subpassCount = 1;
    createInfo.pSubpasses = &subpass;
    createInfo.dependencyCount = 1;
    createInfo.pDependencies = &dependency;

    // Ask Vulkan to create the render pass.
    if (vkCreateRenderPass(app.device, &createInfo, nullptr, &app.renderPass) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create render pass.");
    }
}

// Load a compiled SPIR-V shader file and create a shader module from it.
[[nodiscard]] VkShaderModule LoadShaderModule(const App& app, const std::filesystem::path& path) {
    // Read the raw SPIR-V bytes from disk.
    const auto code = ReadBinaryFile(path);

    // Describe the shader module we want to create from those bytes.
    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = code.size();
    createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

    // This will receive the created shader-module handle.
    VkShaderModule module = VK_NULL_HANDLE;

    // Ask Vulkan to create the shader module.
    if (vkCreateShaderModule(app.device, &createInfo, nullptr, &module) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create shader module for " + path.string());
    }

    // Return the created shader module.
    return module;
}

// Create the graphics pipeline, which packages almost all GPU rendering state into one object.
void CreatePipeline(App& app) {
    // Load the vertex shader module from disk.
    const VkShaderModule vertModule = LoadShaderModule(app, "shaders/triangle.vert.spv");

    // Load the fragment shader module from disk.
    const VkShaderModule fragModule = LoadShaderModule(app, "shaders/triangle.frag.spv");

    // Describe the vertex-shader stage.
    VkPipelineShaderStageCreateInfo vertStage{};
    vertStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertStage.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vertStage.module = vertModule;
    vertStage.pName = "main";

    // Describe the fragment-shader stage.
    VkPipelineShaderStageCreateInfo fragStage{};
    fragStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragStage.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragStage.module = fragModule;
    fragStage.pName = "main";

    // Put both shader stages into a small array.
    const std::array<VkPipelineShaderStageCreateInfo, 2> shaderStages = {vertStage, fragStage};

    // No vertex buffer yet. The vertex shader contains hardcoded positions for cubes and the plane.
    VkPipelineVertexInputStateCreateInfo vertexInput{};
    vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = static_cast<float>(app.swapchainExtent.width);
    viewport.height = static_cast<float>(app.swapchainExtent.height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    VkRect2D scissor{};
    scissor.offset = {0, 0};
    scissor.extent = app.swapchainExtent;

    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.pViewports = &viewport;
    viewportState.scissorCount = 1;
    viewportState.pScissors = &scissor;

    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = VK_CULL_MODE_NONE;
    rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;

    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = VK_TRUE;
    depthStencil.depthWriteEnable = VK_TRUE;
    depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;

    VkPipelineColorBlendAttachmentState colorBlendAttachment{};
    colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                          VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

    VkPipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &colorBlendAttachment;

    // Describe the push-constant region the vertex shader will read.
    VkPushConstantRange pushConstantRange{};
    pushConstantRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    pushConstantRange.offset = 0;
    pushConstantRange.size = sizeof(PushConstants);

    // Describe the pipeline layout, which is where push constants and descriptor sets are declared.
    VkPipelineLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutInfo.pushConstantRangeCount = 1;
    layoutInfo.pPushConstantRanges = &pushConstantRange;

    // Ask Vulkan to create the pipeline layout.
    if (vkCreatePipelineLayout(app.device, &layoutInfo, nullptr, &app.pipelineLayout) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create pipeline layout.");
    }

    // Describe the final graphics pipeline as a whole.
    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount = static_cast<uint32_t>(shaderStages.size());
    pipelineInfo.pStages = shaderStages.data();
    pipelineInfo.pVertexInputState = &vertexInput;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pDepthStencilState = &depthStencil;
    pipelineInfo.pColorBlendState = &colorBlending;
    pipelineInfo.layout = app.pipelineLayout;
    pipelineInfo.renderPass = app.renderPass;
    pipelineInfo.subpass = 0;

    // Ask Vulkan to create the graphics pipeline.
    const VkResult result = vkCreateGraphicsPipelines(app.device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &app.pipeline);

    // The pipeline no longer needs the standalone shader-module objects after creation.
    vkDestroyShaderModule(app.device, fragModule, nullptr);
    vkDestroyShaderModule(app.device, vertModule, nullptr);

    // If pipeline creation failed, stop immediately.
    if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to create graphics pipeline.");
    }
}

// Create the additive pipeline used to render the fire billboards.
void CreateFirePipeline(App& app) {
    // Load the fire shaders from disk.
    const VkShaderModule vertModule = LoadShaderModule(app, "shaders/fire.vert.spv");
    const VkShaderModule fragModule = LoadShaderModule(app, "shaders/fire.frag.spv");

    // Describe the vertex-shader stage.
    VkPipelineShaderStageCreateInfo vertStage{};
    vertStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertStage.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vertStage.module = vertModule;
    vertStage.pName = "main";

    // Describe the fragment-shader stage.
    VkPipelineShaderStageCreateInfo fragStage{};
    fragStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragStage.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragStage.module = fragModule;
    fragStage.pName = "main";

    // Put both fire shader stages into a small array.
    const std::array<VkPipelineShaderStageCreateInfo, 2> shaderStages = {vertStage, fragStage};

    // The fire billboards also generate their vertices procedurally in the shader.
    VkPipelineVertexInputStateCreateInfo vertexInput{};
    vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

    // Each billboard is rendered as two triangles.
    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    // Use the full swapchain image as the viewport.
    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = static_cast<float>(app.swapchainExtent.width);
    viewport.height = static_cast<float>(app.swapchainExtent.height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    VkRect2D scissor{};
    scissor.offset = {0, 0};
    scissor.extent = app.swapchainExtent;

    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.pViewports = &viewport;
    viewportState.scissorCount = 1;
    viewportState.pScissors = &scissor;

    // Disable culling so the sprites remain visible from both sides.
    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = VK_CULL_MODE_NONE;
    rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;

    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    // Let the fire respect existing scene depth while avoiding writes that would punch holes in itself.
    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = VK_TRUE;
    depthStencil.depthWriteEnable = VK_FALSE;
    depthStencil.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;

    // Add fire color on top of whatever was already drawn in the scene.
    VkPipelineColorBlendAttachmentState colorBlendAttachment{};
    colorBlendAttachment.blendEnable = VK_TRUE;
    colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
    colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
    colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
    colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;
    colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                          VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

    VkPipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &colorBlendAttachment;

    // The fire shaders need the same push-constant data in both shader stages.
    VkPushConstantRange pushConstantRange{};
    pushConstantRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    pushConstantRange.offset = 0;
    pushConstantRange.size = sizeof(FirePushConstants);

    VkPipelineLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutInfo.pushConstantRangeCount = 1;
    layoutInfo.pPushConstantRanges = &pushConstantRange;

    if (vkCreatePipelineLayout(app.device, &layoutInfo, nullptr, &app.firePipelineLayout) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create fire pipeline layout.");
    }

    // Describe the fire graphics pipeline as a whole.
    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount = static_cast<uint32_t>(shaderStages.size());
    pipelineInfo.pStages = shaderStages.data();
    pipelineInfo.pVertexInputState = &vertexInput;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pDepthStencilState = &depthStencil;
    pipelineInfo.pColorBlendState = &colorBlending;
    pipelineInfo.layout = app.firePipelineLayout;
    pipelineInfo.renderPass = app.renderPass;
    pipelineInfo.subpass = 0;

    const VkResult result = vkCreateGraphicsPipelines(app.device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &app.firePipeline);

    // Shader modules are no longer needed after pipeline creation.
    vkDestroyShaderModule(app.device, fragModule, nullptr);
    vkDestroyShaderModule(app.device, vertModule, nullptr);

    if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to create fire graphics pipeline.");
    }
}

// Create a framebuffer for each swapchain image.
void CreateFramebuffers(App& app) {
    // Prepare one framebuffer slot per swapchain image.
    app.framebuffers.resize(app.swapchainImageViews.size());

    // Build a framebuffer from one color view and one matching depth view.
    for (std::size_t i = 0; i < app.swapchainImageViews.size(); ++i) {
        const std::array<VkImageView, 2> attachments = {app.swapchainImageViews[i], app.depthImageViews[i]};

        VkFramebufferCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        createInfo.renderPass = app.renderPass;
        createInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
        createInfo.pAttachments = attachments.data();
        createInfo.width = app.swapchainExtent.width;
        createInfo.height = app.swapchainExtent.height;
        createInfo.layers = 1;

        if (vkCreateFramebuffer(app.device, &createInfo, nullptr, &app.framebuffers[i]) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create framebuffer.");
        }
    }
}

// Create the command pool used to allocate command buffers.
void CreateCommandPool(App& app) {
    // We allocate command buffers from the graphics queue family.
    const QueueFamilies families = FindQueueFamilies(app, app.physicalDevice);

    // Describe the command pool.
    VkCommandPoolCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    createInfo.queueFamilyIndex = families.graphics.value();

    // Allow individual command buffers to be reset and recorded again every frame.
    createInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    // Ask Vulkan to create the command pool.
    if (vkCreateCommandPool(app.device, &createInfo, nullptr, &app.commandPool) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create command pool.");
    }
}

// Allocate one primary command buffer per framebuffer.
void CreateCommandBuffers(App& app) {
    // Prepare one command-buffer slot per framebuffer / swapchain image.
    app.commandBuffers.resize(app.framebuffers.size());

    // Describe the command-buffer allocation request.
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = app.commandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = static_cast<uint32_t>(app.commandBuffers.size());

    // Ask Vulkan to allocate the command buffers.
    if (vkAllocateCommandBuffers(app.device, &allocInfo, app.commandBuffers.data()) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate command buffers.");
    }
}

// Record one command buffer for one swapchain image using the current camera matrix.
void RecordCommandBuffer(App& app, const uint32_t imageIndex, const Mat4& viewProjection) {
    // Describe how this command buffer will be recorded.
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    // Start recording commands into this command buffer.
    if (vkBeginCommandBuffer(app.commandBuffers[imageIndex], &beginInfo) != VK_SUCCESS) {
        throw std::runtime_error("Failed to begin command buffer.");
    }

    // We clear color and depth every frame before drawing.
    std::array<VkClearValue, 2> clearValues{};
    clearValues[0].color.float32[0] = 0.09f;
    clearValues[0].color.float32[1] = 0.11f;
    clearValues[0].color.float32[2] = 0.15f;
    clearValues[0].color.float32[3] = 1.0f;
    clearValues[1].depthStencil.depth = 1.0f;

    // Describe the render pass instance that will draw into the current framebuffer.
    VkRenderPassBeginInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    renderPassInfo.renderPass = app.renderPass;
    renderPassInfo.framebuffer = app.framebuffers[imageIndex];
    renderPassInfo.renderArea.offset = {0, 0};
    renderPassInfo.renderArea.extent = app.swapchainExtent;
    renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
    renderPassInfo.pClearValues = clearValues.data();

    // Begin the render pass.
    vkCmdBeginRenderPass(app.commandBuffers[imageIndex], &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

    // Bind the graphics pipeline so all later draw calls use it.
    vkCmdBindPipeline(app.commandBuffers[imageIndex], VK_PIPELINE_BIND_POINT_GRAPHICS, app.pipeline);

    // Each object reuses the same pipeline. Only the push constants and
    // vertex count change between the plane and the cubes.
    for (const SceneObject& object : kSceneObjects) {
        const PushConstants constants = MakePushConstants(viewProjection, object);
        vkCmdPushConstants(app.commandBuffers[imageIndex],
                           app.pipelineLayout,
                           VK_SHADER_STAGE_VERTEX_BIT,
                           0,
                           sizeof(PushConstants),
                           &constants);
        vkCmdDraw(app.commandBuffers[imageIndex], object.vertexCount, 1, 0, 0);
    }

    // Rebuild the camera billboard basis so the fire always faces the viewer.
    const Vec3 cameraForward = BuildCameraForward(app);
    const Vec3 cameraRight = Normalize(Cross(cameraForward, {0.0f, 1.0f, 0.0f}));
    const Vec3 cameraUp = Normalize(Cross(cameraRight, cameraForward));

    // Render the fire as additive billboards after the opaque scene geometry.
    vkCmdBindPipeline(app.commandBuffers[imageIndex], VK_PIPELINE_BIND_POINT_GRAPHICS, app.firePipeline);

    for (const FireParticle& particle : app.fireParticles) {
        const FirePushConstants constants = MakeFirePushConstants(viewProjection, particle, cameraRight, cameraUp);
        vkCmdPushConstants(app.commandBuffers[imageIndex],
                           app.firePipelineLayout,
                           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                           0,
                           sizeof(FirePushConstants),
                           &constants);
        vkCmdDraw(app.commandBuffers[imageIndex], kFireQuadVertexCount, 1, 0, 0);
    }

    // Finish the render pass.
    vkCmdEndRenderPass(app.commandBuffers[imageIndex]);

    // Finish command-buffer recording.
    if (vkEndCommandBuffer(app.commandBuffers[imageIndex]) != VK_SUCCESS) {
        throw std::runtime_error("Failed to record command buffer.");
    }
}

// Create the semaphores and fences used to coordinate CPU work, GPU work, and presentation.
void CreateSyncObjects(App& app) {
    // Describe how semaphores should be created.
    VkSemaphoreCreateInfo semaphoreInfo{};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    // Describe how fences should be created.
    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;

    // Start fences in the signaled state so the first frame does not wait forever.
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    // Create one render-finished semaphore per swapchain image.
    app.renderFinishedPerImage.resize(app.swapchainImages.size(), VK_NULL_HANDLE);

    // Start with no fence owning any swapchain image.
    app.imageInFlight.assign(app.swapchainImages.size(), VK_NULL_HANDLE);

    // Create the per-image render-finished semaphores.
    for (std::size_t i = 0; i < app.swapchainImages.size(); ++i) {
        if (vkCreateSemaphore(app.device, &semaphoreInfo, nullptr, &app.renderFinishedPerImage[i]) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create render-finished semaphore.");
        }
    }

    // Create the per-frame image-available semaphores and in-flight fences.
    for (std::size_t i = 0; i < kMaxFramesInFlight; ++i) {
        if (vkCreateSemaphore(app.device, &semaphoreInfo, nullptr, &app.imageAvailable[i]) != VK_SUCCESS ||
            vkCreateFence(app.device, &fenceInfo, nullptr, &app.inFlightFences[i]) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create frame synchronization objects.");
        }
    }
}

// Render and present one frame.
void DrawFrame(App& app) {
    // Wait until the GPU has finished the previous work associated with this CPU frame slot.
    vkWaitForFences(app.device, 1, &app.inFlightFences[app.currentFrame], VK_TRUE, UINT64_MAX);

    // This variable will receive the index of the swapchain image we acquired.
    uint32_t imageIndex = 0;

    // Ask Vulkan for the next available swapchain image.
    const VkResult acquireResult = vkAcquireNextImageKHR(app.device,
                                                         app.swapchain,
                                                         UINT64_MAX,
                                                         app.imageAvailable[app.currentFrame],
                                                         VK_NULL_HANDLE,
                                                         &imageIndex);
    if (acquireResult != VK_SUCCESS && acquireResult != VK_SUBOPTIMAL_KHR) {
        throw std::runtime_error("Failed to acquire swapchain image.");
    }

    // If another frame is still using this image, wait for that frame to finish.
    if (app.imageInFlight[imageIndex] != VK_NULL_HANDLE) {
        vkWaitForFences(app.device, 1, &app.imageInFlight[imageIndex], VK_TRUE, UINT64_MAX);
    }

    // Mark this swapchain image as now being owned by the current frame's fence.
    app.imageInFlight[imageIndex] = app.inFlightFences[app.currentFrame];

    // Reset the current frame fence because new GPU work is about to use it.
    vkResetFences(app.device, 1, &app.inFlightFences[app.currentFrame]);

    // Update the camera from keyboard and mouse input.
    const float deltaSeconds = UpdateFreeCamera(app);

    // Advance the fire simulation using the same frame delta as the camera.
    UpdateFireParticles(app, deltaSeconds);

    // Build the current camera matrix.
    const Mat4 viewProjection = BuildViewProjection(app);

    // Reset this image's command buffer so we can record fresh commands into it.
    if (vkResetCommandBuffer(app.commandBuffers[imageIndex], 0) != VK_SUCCESS) {
        throw std::runtime_error("Failed to reset command buffer.");
    }

    // Record the draw commands for this frame.
    RecordCommandBuffer(app, imageIndex, viewProjection);

    // The GPU must wait until the swapchain image is available before drawing.
    const VkSemaphore waitSemaphores[] = {app.imageAvailable[app.currentFrame]};

    // We wait in the color-attachment-output stage because that is where the swapchain image is used.
    const VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};

    // The GPU will signal this semaphore once rendering is finished.
    const VkSemaphore signalSemaphores[] = {app.renderFinishedPerImage[imageIndex]};

    // Describe the graphics queue submission.
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = waitSemaphores;
    submitInfo.pWaitDstStageMask = waitStages;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &app.commandBuffers[imageIndex];
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = signalSemaphores;

    // Submit the recorded command buffer to the graphics queue.
    if (vkQueueSubmit(app.graphicsQueue, 1, &submitInfo, app.inFlightFences[app.currentFrame]) != VK_SUCCESS) {
        throw std::runtime_error("Failed to submit command buffer.");
    }

    // Describe how we want to present the finished swapchain image.
    VkPresentInfoKHR presentInfo{};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = signalSemaphores;
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = &app.swapchain;
    presentInfo.pImageIndices = &imageIndex;

    // Ask the present queue to show the finished image.
    const VkResult presentResult = vkQueuePresentKHR(app.presentQueue, &presentInfo);
    if (presentResult != VK_SUCCESS && presentResult != VK_SUBOPTIMAL_KHR) {
        throw std::runtime_error("Failed to present swapchain image.");
    }

    // Move to the next CPU frame slot.
    app.currentFrame = (app.currentFrame + 1) % kMaxFramesInFlight;
}

// Destroy all Vulkan and GLFW resources in the reverse order they were created.
void Cleanup(App& app) {
    // Only clean up device-owned resources if the device exists.
    if (app.device != VK_NULL_HANDLE) {
        // Wait for the device so nothing is still using these resources while we destroy them.
        vkDeviceWaitIdle(app.device);

        // Destroy every render-finished semaphore.
        for (auto semaphore : app.renderFinishedPerImage) {
            if (semaphore != VK_NULL_HANDLE) {
                vkDestroySemaphore(app.device, semaphore, nullptr);
            }
        }

        // Destroy the per-frame semaphores and fences.
        for (std::size_t i = 0; i < kMaxFramesInFlight; ++i) {
            if (app.imageAvailable[i] != VK_NULL_HANDLE) {
                vkDestroySemaphore(app.device, app.imageAvailable[i], nullptr);
            }
            if (app.inFlightFences[i] != VK_NULL_HANDLE) {
                vkDestroyFence(app.device, app.inFlightFences[i], nullptr);
            }
        }

        // Destroy the command pool and all command buffers allocated from it.
        if (app.commandPool != VK_NULL_HANDLE) {
            vkDestroyCommandPool(app.device, app.commandPool, nullptr);
        }

        // Destroy every framebuffer.
        for (auto framebuffer : app.framebuffers) {
            if (framebuffer != VK_NULL_HANDLE) {
                vkDestroyFramebuffer(app.device, framebuffer, nullptr);
            }
        }

        // Destroy pipeline objects.
        if (app.firePipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(app.device, app.firePipeline, nullptr);
        }
        if (app.firePipelineLayout != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(app.device, app.firePipelineLayout, nullptr);
        }
        if (app.pipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(app.device, app.pipeline, nullptr);
        }
        if (app.pipelineLayout != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(app.device, app.pipelineLayout, nullptr);
        }
        if (app.renderPass != VK_NULL_HANDLE) {
            vkDestroyRenderPass(app.device, app.renderPass, nullptr);
        }

        // Destroy depth-image views first.
        for (auto imageView : app.depthImageViews) {
            if (imageView != VK_NULL_HANDLE) {
                vkDestroyImageView(app.device, imageView, nullptr);
            }
        }

        // Then destroy the depth images themselves.
        for (auto image : app.depthImages) {
            if (image != VK_NULL_HANDLE) {
                vkDestroyImage(app.device, image, nullptr);
            }
        }

        // Finally free the memory backing the depth images.
        for (auto memory : app.depthMemories) {
            if (memory != VK_NULL_HANDLE) {
                vkFreeMemory(app.device, memory, nullptr);
            }
        }

        // Destroy the swapchain image views.
        for (auto imageView : app.swapchainImageViews) {
            if (imageView != VK_NULL_HANDLE) {
                vkDestroyImageView(app.device, imageView, nullptr);
            }
        }

        // Destroy the swapchain.
        if (app.swapchain != VK_NULL_HANDLE) {
            vkDestroySwapchainKHR(app.device, app.swapchain, nullptr);
        }

        // Destroy the logical device itself.
        vkDestroyDevice(app.device, nullptr);
    }

    // Destroy instance-level resources after device-level resources are gone.
    if (app.surface != VK_NULL_HANDLE) {
        vkDestroySurfaceKHR(app.instance, app.surface, nullptr);
    }
    if (app.debugMessenger != VK_NULL_HANDLE) {
        vkDestroyDebugUtilsMessengerEXT(app.instance, app.debugMessenger, nullptr);
    }
    if (app.instance != VK_NULL_HANDLE) {
        vkDestroyInstance(app.instance, nullptr);
    }
    if (app.window != nullptr) {
        glfwDestroyWindow(app.window);
    }

    // Shut down GLFW after the window is destroyed.
    glfwTerminate();
}

// Build the app, run the main loop, and clean everything up again.
void Run() {
    // Start with a zero-initialized App struct.
    App app;

    // Wrap the whole program in a try/catch-friendly cleanup structure.
    try {
        // Create the desktop window first.
        CreateWindow(app);

        // Initialize volk before using Vulkan function pointers.
        if (volkInitialize() != VK_SUCCESS) {
            throw std::runtime_error("Failed to initialize volk.");
        }

        // Create and connect all major Vulkan objects in dependency order.
        CreateInstance(app);
        volkLoadInstance(app.instance);
        CreateDebugMessenger(app);
        CreateSurface(app);
        PickPhysicalDevice(app);
        CreateDevice(app);
        volkLoadDevice(app.device);
        CreateSwapchain(app);
        CreateSwapchainImageViews(app);
        CreateDepthResources(app);
        CreateRenderPass(app);
        CreatePipeline(app);
        CreateFirePipeline(app);
        CreateFramebuffers(app);
        CreateCommandPool(app);
        CreateCommandBuffers(app);
        CreateSyncObjects(app);

        // Remember the current time so the camera's first delta time is sane.
        app.lastFrameTime = glfwGetTime();

        // Seed the particle pool before the first frame is rendered.
        InitializeFireParticles(app);

        // Run until the user closes the window.
        while (!glfwWindowShouldClose(app.window)) {
            // Let GLFW process OS events like mouse movement and key presses.
            glfwPollEvents();

            // Render one frame.
            DrawFrame(app);
        }

        // Clean up everything after the loop exits normally.
        Cleanup(app);
    } catch (...) {
        // Clean up even when an exception escapes.
        Cleanup(app);
        throw;
    }
}

}  // namespace

// Standard C++ entry point.
int main() {
    // Use a try/catch so we can print a readable error instead of crashing silently.
    try {
        // Run the application.
        Run();

        // Return 0 to the operating system to say "success".
        return 0;
    } catch (const std::exception& exception) {
        // Print the error message for the user.
        std::cerr << exception.what() << '\n';

        // Return non-zero to say "something failed".
        return 1;
    }
}
