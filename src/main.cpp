#include <volk.h>
#include <GLFW/glfw3.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr int kWindowWidth = 1280;
constexpr int kWindowHeight = 720;
constexpr std::size_t kMaxFramesInFlight = 2;
constexpr float kPi = 3.1415926535f;
constexpr std::array<const char*, 1> kValidationLayers = {"VK_LAYER_KHRONOS_validation"};
constexpr std::array<const char*, 1> kDeviceExtensions = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};

enum MeshType : int32_t {
    kMeshCube = 0,
    kMeshPlane = 1,
};

struct Vec3 {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
};

struct Vec4 {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    float w = 1.0f;
};

struct Mat4 {
    std::array<float, 16> m{};
};

struct alignas(16) PushConstants {
    std::array<float, 16> viewProjection{};
    std::array<float, 4> color{};
    std::array<float, 4> transform{};
    std::array<int32_t, 4> params{};
};

static_assert(sizeof(PushConstants) <= 128, "Push constants must fit within Vulkan's guaranteed minimum.");

struct SceneObject {
    MeshType meshType = kMeshCube;
    Vec3 position{};
    float scale = 1.0f;
    Vec4 color{};
    uint32_t vertexCount = 0;
};

// One oversized plane plus a few cubes is enough to prove out the basic 3D path:
// projection, depth testing, multiple draw calls, and per-object transforms.
constexpr std::array<SceneObject, 6> kSceneObjects = {{
    {kMeshPlane, {0.0f, 0.0f, 0.0f}, 12.0f, {0.33f, 0.39f, 0.28f, 1.0f}, 6},
    {kMeshCube, {-3.0f, 0.5f, -2.5f}, 1.0f, {0.92f, 0.37f, 0.28f, 1.0f}, 36},
    {kMeshCube, {-0.7f, 0.5f, -1.0f}, 1.0f, {0.28f, 0.76f, 0.44f, 1.0f}, 36},
    {kMeshCube, {1.6f, 0.5f, -2.2f}, 1.0f, {0.26f, 0.56f, 0.94f, 1.0f}, 36},
    {kMeshCube, {-1.8f, 0.5f, 1.6f}, 1.0f, {0.90f, 0.74f, 0.22f, 1.0f}, 36},
    {kMeshCube, {2.2f, 0.5f, 1.8f}, 1.0f, {0.74f, 0.42f, 0.92f, 1.0f}, 36},
}};

struct QueueFamilies {
    std::optional<uint32_t> graphics;
    std::optional<uint32_t> present;

    [[nodiscard]] bool complete() const {
        return graphics.has_value() && present.has_value();
    }
};

struct SwapchainSupport {
    VkSurfaceCapabilitiesKHR capabilities{};
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};

struct App {
    GLFWwindow* window = nullptr;

    VkInstance instance = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT debugMessenger = VK_NULL_HANDLE;
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue graphicsQueue = VK_NULL_HANDLE;
    VkQueue presentQueue = VK_NULL_HANDLE;

    VkSwapchainKHR swapchain = VK_NULL_HANDLE;
    std::vector<VkImage> swapchainImages;
    std::vector<VkImageView> swapchainImageViews;
    VkFormat swapchainFormat = VK_FORMAT_UNDEFINED;
    VkExtent2D swapchainExtent{};

    VkFormat depthFormat = VK_FORMAT_UNDEFINED;
    std::vector<VkImage> depthImages;
    std::vector<VkDeviceMemory> depthMemories;
    std::vector<VkImageView> depthImageViews;

    VkRenderPass renderPass = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    std::vector<VkFramebuffer> framebuffers;

    VkCommandPool commandPool = VK_NULL_HANDLE;
    std::vector<VkCommandBuffer> commandBuffers;

    std::array<VkSemaphore, kMaxFramesInFlight> imageAvailable{};
    std::array<VkFence, kMaxFramesInFlight> inFlightFences{};
    std::vector<VkSemaphore> renderFinishedPerImage;
    std::vector<VkFence> imageInFlight;
    std::size_t currentFrame = 0;

    bool validationEnabled = false;
};

// Read a compiled SPIR-V shader file into memory.
[[nodiscard]] std::vector<char> ReadBinaryFile(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open " + path.string());
    }

    const auto size = static_cast<std::size_t>(file.tellg());
    std::vector<char> bytes(size);
    file.seekg(0);
    if (size > 0) {
        file.read(bytes.data(), static_cast<std::streamsize>(size));
    }

    return bytes;
}

// Ignore a known broken Vulkan layer registration from TikTok LIVE Studio so the
// console only shows messages that matter for this sample.
[[nodiscard]] bool ShouldIgnoreDebugMessage(const std::string_view message) {
    const bool missingJson = message.find("loader_get_json: Failed to open JSON file") != std::string_view::npos;
    const bool tiktokLayer = message.find("LiveStudioVulkanLayer") != std::string_view::npos ||
                             message.find("TikTok LIVE Studio") != std::string_view::npos;
    return missingJson && tiktokLayer;
}

VKAPI_ATTR VkBool32 VKAPI_CALL DebugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                                             VkDebugUtilsMessageTypeFlagsEXT,
                                             const VkDebugUtilsMessengerCallbackDataEXT* callbackData,
                                             void*) {
    const std::string_view message =
        (callbackData != nullptr && callbackData->pMessage != nullptr) ? callbackData->pMessage : "";

    if (!ShouldIgnoreDebugMessage(message) && severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
        std::cerr << "[Vulkan] " << message << '\n';
    }

    return VK_FALSE;
}

[[nodiscard]] float Dot(const Vec3& a, const Vec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

[[nodiscard]] Vec3 Cross(const Vec3& a, const Vec3& b) {
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    };
}

[[nodiscard]] Vec3 Normalize(const Vec3& value) {
    const float length = std::sqrt(Dot(value, value));
    return {value.x / length, value.y / length, value.z / length};
}

[[nodiscard]] Mat4 IdentityMatrix() {
    Mat4 result{};
    result.m[0] = 1.0f;
    result.m[5] = 1.0f;
    result.m[10] = 1.0f;
    result.m[15] = 1.0f;
    return result;
}

// Column-major matrix multiplication to match GLSL's default matrix layout.
[[nodiscard]] Mat4 Multiply(const Mat4& left, const Mat4& right) {
    Mat4 result{};

    for (int column = 0; column < 4; ++column) {
        for (int row = 0; row < 4; ++row) {
            float sum = 0.0f;
            for (int inner = 0; inner < 4; ++inner) {
                sum += left.m[inner * 4 + row] * right.m[column * 4 + inner];
            }
            result.m[column * 4 + row] = sum;
        }
    }

    return result;
}

// Right-handed look-at matrix. This builds the camera transform.
[[nodiscard]] Mat4 LookAt(const Vec3& eye, const Vec3& center, const Vec3& up) {
    const Vec3 forward = Normalize({center.x - eye.x, center.y - eye.y, center.z - eye.z});
    const Vec3 side = Normalize(Cross(forward, up));
    const Vec3 actualUp = Cross(side, forward);

    Mat4 result = IdentityMatrix();
    result.m[0] = side.x;
    result.m[1] = actualUp.x;
    result.m[2] = -forward.x;
    result.m[4] = side.y;
    result.m[5] = actualUp.y;
    result.m[6] = -forward.y;
    result.m[8] = side.z;
    result.m[9] = actualUp.z;
    result.m[10] = -forward.z;
    result.m[12] = -Dot(side, eye);
    result.m[13] = -Dot(actualUp, eye);
    result.m[14] = Dot(forward, eye);
    return result;
}

// Perspective projection using Vulkan's 0..1 depth range. The negative Y entry
// keeps the image upright with a normal viewport.
[[nodiscard]] Mat4 Perspective(float fovRadians, float aspectRatio, float nearPlane, float farPlane) {
    Mat4 result{};
    const float tanHalfFov = std::tan(fovRadians * 0.5f);

    result.m[0] = 1.0f / (aspectRatio * tanHalfFov);
    result.m[5] = -1.0f / tanHalfFov;
    result.m[10] = farPlane / (nearPlane - farPlane);
    result.m[11] = -1.0f;
    result.m[14] = -(farPlane * nearPlane) / (farPlane - nearPlane);
    return result;
}

[[nodiscard]] PushConstants MakePushConstants(const Mat4& viewProjection, const SceneObject& object) {
    PushConstants constants{};
    constants.viewProjection = viewProjection.m;
    constants.color = {object.color.x, object.color.y, object.color.z, object.color.w};
    constants.transform = {object.position.x, object.position.y, object.position.z, object.scale};
    constants.params = {static_cast<int32_t>(object.meshType), 0, 0, 0};
    return constants;
}

[[nodiscard]] bool CheckValidationLayerSupport() {
    uint32_t layerCount = 0;
    vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

    std::vector<VkLayerProperties> layers(layerCount);
    vkEnumerateInstanceLayerProperties(&layerCount, layers.data());

    for (const char* required : kValidationLayers) {
        const auto it = std::find_if(layers.begin(), layers.end(), [required](const VkLayerProperties& layer) {
            return std::strcmp(layer.layerName, required) == 0;
        });
        if (it == layers.end()) {
            return false;
        }
    }

    return true;
}

[[nodiscard]] std::vector<const char*> GetRequiredExtensions(const bool validationEnabled) {
    uint32_t count = 0;
    const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&count);
    std::vector<const char*> extensions(glfwExtensions, glfwExtensions + count);

    if (validationEnabled) {
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    return extensions;
}

[[nodiscard]] QueueFamilies FindQueueFamilies(const App& app, const VkPhysicalDevice device) {
    QueueFamilies families;

    uint32_t queueCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueCount, nullptr);

    std::vector<VkQueueFamilyProperties> queues(queueCount);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueCount, queues.data());

    for (uint32_t i = 0; i < queueCount; ++i) {
        if ((queues[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0) {
            families.graphics = i;
        }

        VkBool32 supportsPresent = VK_FALSE;
        vkGetPhysicalDeviceSurfaceSupportKHR(device, i, app.surface, &supportsPresent);
        if (supportsPresent == VK_TRUE) {
            families.present = i;
        }

        if (families.complete()) {
            break;
        }
    }

    return families;
}

[[nodiscard]] bool CheckDeviceExtensions(const VkPhysicalDevice device) {
    uint32_t extensionCount = 0;
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);

    std::vector<VkExtensionProperties> extensions(extensionCount);
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, extensions.data());

    std::set<std::string_view> required(kDeviceExtensions.begin(), kDeviceExtensions.end());
    for (const auto& extension : extensions) {
        required.erase(extension.extensionName);
    }

    return required.empty();
}

[[nodiscard]] SwapchainSupport QuerySwapchainSupport(const App& app, const VkPhysicalDevice device) {
    SwapchainSupport support;

    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, app.surface, &support.capabilities);

    uint32_t formatCount = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(device, app.surface, &formatCount, nullptr);
    if (formatCount > 0) {
        support.formats.resize(formatCount);
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, app.surface, &formatCount, support.formats.data());
    }

    uint32_t presentModeCount = 0;
    vkGetPhysicalDeviceSurfacePresentModesKHR(device, app.surface, &presentModeCount, nullptr);
    if (presentModeCount > 0) {
        support.presentModes.resize(presentModeCount);
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, app.surface, &presentModeCount, support.presentModes.data());
    }

    return support;
}

[[nodiscard]] bool IsDeviceSuitable(const App& app, const VkPhysicalDevice device) {
    const QueueFamilies families = FindQueueFamilies(app, device);
    if (!families.complete() || !CheckDeviceExtensions(device)) {
        return false;
    }

    const SwapchainSupport swapchainSupport = QuerySwapchainSupport(app, device);
    return !swapchainSupport.formats.empty() && !swapchainSupport.presentModes.empty();
}

[[nodiscard]] VkSurfaceFormatKHR ChooseSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& formats) {
    for (const auto& format : formats) {
        if (format.format == VK_FORMAT_B8G8R8A8_SRGB &&
            format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            return format;
        }
    }

    return formats.front();
}

[[nodiscard]] VkPresentModeKHR ChoosePresentMode(const std::vector<VkPresentModeKHR>& presentModes) {
    for (const auto mode : presentModes) {
        if (mode == VK_PRESENT_MODE_MAILBOX_KHR) {
            return mode;
        }
    }

    return VK_PRESENT_MODE_FIFO_KHR;
}

[[nodiscard]] VkExtent2D ChooseExtent(const SwapchainSupport& support) {
    if (support.capabilities.currentExtent.width != UINT32_MAX) {
        return support.capabilities.currentExtent;
    }

    VkExtent2D extent{};
    extent.width = std::clamp(static_cast<uint32_t>(kWindowWidth),
                              support.capabilities.minImageExtent.width,
                              support.capabilities.maxImageExtent.width);
    extent.height = std::clamp(static_cast<uint32_t>(kWindowHeight),
                               support.capabilities.minImageExtent.height,
                               support.capabilities.maxImageExtent.height);
    return extent;
}

[[nodiscard]] uint32_t FindMemoryType(const App& app, const uint32_t typeFilter, const VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memoryProperties{};
    vkGetPhysicalDeviceMemoryProperties(app.physicalDevice, &memoryProperties);

    for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; ++i) {
        const bool supportedByResource = (typeFilter & (1u << i)) != 0;
        const bool hasFlags = (memoryProperties.memoryTypes[i].propertyFlags & properties) == properties;
        if (supportedByResource && hasFlags) {
            return i;
        }
    }

    throw std::runtime_error("Failed to find suitable memory type.");
}

[[nodiscard]] bool HasStencilComponent(const VkFormat format) {
    return format == VK_FORMAT_D32_SFLOAT_S8_UINT || format == VK_FORMAT_D24_UNORM_S8_UINT;
}

[[nodiscard]] VkFormat FindSupportedFormat(const App& app,
                                           const std::vector<VkFormat>& candidates,
                                           const VkImageTiling tiling,
                                           const VkFormatFeatureFlags features) {
    for (const VkFormat format : candidates) {
        VkFormatProperties properties{};
        vkGetPhysicalDeviceFormatProperties(app.physicalDevice, format, &properties);

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

[[nodiscard]] VkFormat FindDepthFormat(const App& app) {
    return FindSupportedFormat(app,
                               {VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT},
                               VK_IMAGE_TILING_OPTIMAL,
                               VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);
}

void CreateImage(App& app,
                 const uint32_t width,
                 const uint32_t height,
                 const VkFormat format,
                 const VkImageUsageFlags usage,
                 const VkMemoryPropertyFlags properties,
                 VkImage& image,
                 VkDeviceMemory& memory) {
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

    if (vkCreateImage(app.device, &imageInfo, nullptr, &image) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create image.");
    }

    VkMemoryRequirements memoryRequirements{};
    vkGetImageMemoryRequirements(app.device, image, &memoryRequirements);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memoryRequirements.size;
    allocInfo.memoryTypeIndex = FindMemoryType(app, memoryRequirements.memoryTypeBits, properties);

    if (vkAllocateMemory(app.device, &allocInfo, nullptr, &memory) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate image memory.");
    }

    vkBindImageMemory(app.device, image, memory, 0);
}

[[nodiscard]] VkImageView CreateImageView(const App& app,
                                          const VkImage image,
                                          const VkFormat format,
                                          VkImageAspectFlags aspectMask) {
    if (HasStencilComponent(format)) {
        aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
    }

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

    VkImageView imageView = VK_NULL_HANDLE;
    if (vkCreateImageView(app.device, &viewInfo, nullptr, &imageView) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create image view.");
    }

    return imageView;
}

void CreateWindow(App& app) {
    if (glfwInit() != GLFW_TRUE) {
        throw std::runtime_error("Failed to initialize GLFW.");
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

    app.window = glfwCreateWindow(kWindowWidth, kWindowHeight, "Vulkan Cubes", nullptr, nullptr);
    if (app.window == nullptr) {
        throw std::runtime_error("Failed to create window.");
    }
}

void CreateInstance(App& app) {
    if (!glfwVulkanSupported()) {
        throw std::runtime_error("GLFW could not find the Vulkan loader.");
    }

#ifndef NDEBUG
    app.validationEnabled = CheckValidationLayerSupport();
#endif

    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "Vulkan Cubes";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "None";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_1;

    const auto extensions = GetRequiredExtensions(app.validationEnabled);

    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;
    createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    createInfo.ppEnabledExtensionNames = extensions.data();

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
        createInfo.pNext = &debugInfo;
    }

    if (vkCreateInstance(&createInfo, nullptr, &app.instance) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create Vulkan instance.");
    }
}

void CreateDebugMessenger(App& app) {
    if (!app.validationEnabled) {
        return;
    }

    VkDebugUtilsMessengerCreateInfoEXT createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                                 VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                             VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                             VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    createInfo.pfnUserCallback = DebugCallback;

    if (vkCreateDebugUtilsMessengerEXT(app.instance, &createInfo, nullptr, &app.debugMessenger) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create debug messenger.");
    }
}

void CreateSurface(App& app) {
    if (glfwCreateWindowSurface(app.instance, app.window, nullptr, &app.surface) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create window surface.");
    }
}

void PickPhysicalDevice(App& app) {
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(app.instance, &deviceCount, nullptr);
    if (deviceCount == 0) {
        throw std::runtime_error("No Vulkan-capable GPU found.");
    }

    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(app.instance, &deviceCount, devices.data());

    for (const auto device : devices) {
        if (IsDeviceSuitable(app, device)) {
            app.physicalDevice = device;
            return;
        }
    }

    throw std::runtime_error("No suitable Vulkan GPU found.");
}

void CreateDevice(App& app) {
    const QueueFamilies families = FindQueueFamilies(app, app.physicalDevice);
    std::set<uint32_t> uniqueFamilies = {families.graphics.value(), families.present.value()};

    constexpr float priority = 1.0f;
    std::vector<VkDeviceQueueCreateInfo> queueInfos;
    for (const auto family : uniqueFamilies) {
        VkDeviceQueueCreateInfo queueInfo{};
        queueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueInfo.queueFamilyIndex = family;
        queueInfo.queueCount = 1;
        queueInfo.pQueuePriorities = &priority;
        queueInfos.push_back(queueInfo);
    }

    VkPhysicalDeviceFeatures features{};

    VkDeviceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueInfos.size());
    createInfo.pQueueCreateInfos = queueInfos.data();
    createInfo.pEnabledFeatures = &features;
    createInfo.enabledExtensionCount = static_cast<uint32_t>(kDeviceExtensions.size());
    createInfo.ppEnabledExtensionNames = kDeviceExtensions.data();

    if (app.validationEnabled) {
        createInfo.enabledLayerCount = static_cast<uint32_t>(kValidationLayers.size());
        createInfo.ppEnabledLayerNames = kValidationLayers.data();
    }

    if (vkCreateDevice(app.physicalDevice, &createInfo, nullptr, &app.device) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create logical device.");
    }

    vkGetDeviceQueue(app.device, families.graphics.value(), 0, &app.graphicsQueue);
    vkGetDeviceQueue(app.device, families.present.value(), 0, &app.presentQueue);
}

void CreateSwapchain(App& app) {
    const SwapchainSupport support = QuerySwapchainSupport(app, app.physicalDevice);
    const VkSurfaceFormatKHR surfaceFormat = ChooseSurfaceFormat(support.formats);
    const VkPresentModeKHR presentMode = ChoosePresentMode(support.presentModes);
    const VkExtent2D extent = ChooseExtent(support);

    uint32_t imageCount = support.capabilities.minImageCount + 1;
    if (support.capabilities.maxImageCount > 0 && imageCount > support.capabilities.maxImageCount) {
        imageCount = support.capabilities.maxImageCount;
    }

    const QueueFamilies families = FindQueueFamilies(app, app.physicalDevice);
    const uint32_t familyIndices[] = {families.graphics.value(), families.present.value()};

    VkSwapchainCreateInfoKHR createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    createInfo.surface = app.surface;
    createInfo.minImageCount = imageCount;
    createInfo.imageFormat = surfaceFormat.format;
    createInfo.imageColorSpace = surfaceFormat.colorSpace;
    createInfo.imageExtent = extent;
    createInfo.imageArrayLayers = 1;
    createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

    if (families.graphics != families.present) {
        createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
        createInfo.queueFamilyIndexCount = 2;
        createInfo.pQueueFamilyIndices = familyIndices;
    } else {
        createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    }

    createInfo.preTransform = support.capabilities.currentTransform;
    createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    createInfo.presentMode = presentMode;
    createInfo.clipped = VK_TRUE;

    if (vkCreateSwapchainKHR(app.device, &createInfo, nullptr, &app.swapchain) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create swapchain.");
    }

    vkGetSwapchainImagesKHR(app.device, app.swapchain, &imageCount, nullptr);
    app.swapchainImages.resize(imageCount);
    vkGetSwapchainImagesKHR(app.device, app.swapchain, &imageCount, app.swapchainImages.data());

    app.swapchainFormat = surfaceFormat.format;
    app.swapchainExtent = extent;
}

void CreateSwapchainImageViews(App& app) {
    app.swapchainImageViews.resize(app.swapchainImages.size());

    for (std::size_t i = 0; i < app.swapchainImages.size(); ++i) {
        app.swapchainImageViews[i] =
            CreateImageView(app, app.swapchainImages[i], app.swapchainFormat, VK_IMAGE_ASPECT_COLOR_BIT);
    }
}

void CreateDepthResources(App& app) {
    app.depthFormat = FindDepthFormat(app);
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

void CreateRenderPass(App& app) {
    VkAttachmentDescription colorAttachment{};
    colorAttachment.format = app.swapchainFormat;
    colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentDescription depthAttachment{};
    depthAttachment.format = app.depthFormat;
    depthAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentReference colorAttachmentRef{};
    colorAttachmentRef.attachment = 0;
    colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkAttachmentReference depthAttachmentRef{};
    depthAttachmentRef.attachment = 1;
    depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorAttachmentRef;
    subpass.pDepthStencilAttachment = &depthAttachmentRef;

    VkSubpassDependency dependency{};
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

    const std::array<VkAttachmentDescription, 2> attachments = {colorAttachment, depthAttachment};

    VkRenderPassCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    createInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
    createInfo.pAttachments = attachments.data();
    createInfo.subpassCount = 1;
    createInfo.pSubpasses = &subpass;
    createInfo.dependencyCount = 1;
    createInfo.pDependencies = &dependency;

    if (vkCreateRenderPass(app.device, &createInfo, nullptr, &app.renderPass) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create render pass.");
    }
}

[[nodiscard]] VkShaderModule LoadShaderModule(const App& app, const std::filesystem::path& path) {
    const auto code = ReadBinaryFile(path);

    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = code.size();
    createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

    VkShaderModule module = VK_NULL_HANDLE;
    if (vkCreateShaderModule(app.device, &createInfo, nullptr, &module) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create shader module for " + path.string());
    }

    return module;
}

void CreatePipeline(App& app) {
    const VkShaderModule vertModule = LoadShaderModule(app, "shaders/triangle.vert.spv");
    const VkShaderModule fragModule = LoadShaderModule(app, "shaders/triangle.frag.spv");

    VkPipelineShaderStageCreateInfo vertStage{};
    vertStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertStage.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vertStage.module = vertModule;
    vertStage.pName = "main";

    VkPipelineShaderStageCreateInfo fragStage{};
    fragStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragStage.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragStage.module = fragModule;
    fragStage.pName = "main";

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

    VkPushConstantRange pushConstantRange{};
    pushConstantRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    pushConstantRange.offset = 0;
    pushConstantRange.size = sizeof(PushConstants);

    VkPipelineLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutInfo.pushConstantRangeCount = 1;
    layoutInfo.pPushConstantRanges = &pushConstantRange;

    if (vkCreatePipelineLayout(app.device, &layoutInfo, nullptr, &app.pipelineLayout) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create pipeline layout.");
    }

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

    const VkResult result = vkCreateGraphicsPipelines(app.device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &app.pipeline);

    vkDestroyShaderModule(app.device, fragModule, nullptr);
    vkDestroyShaderModule(app.device, vertModule, nullptr);

    if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to create graphics pipeline.");
    }
}

void CreateFramebuffers(App& app) {
    app.framebuffers.resize(app.swapchainImageViews.size());

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

void CreateCommandPool(App& app) {
    const QueueFamilies families = FindQueueFamilies(app, app.physicalDevice);

    VkCommandPoolCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    createInfo.queueFamilyIndex = families.graphics.value();

    if (vkCreateCommandPool(app.device, &createInfo, nullptr, &app.commandPool) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create command pool.");
    }
}

void CreateCommandBuffers(App& app) {
    app.commandBuffers.resize(app.framebuffers.size());

    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = app.commandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = static_cast<uint32_t>(app.commandBuffers.size());

    if (vkAllocateCommandBuffers(app.device, &allocInfo, app.commandBuffers.data()) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate command buffers.");
    }

    // A fixed camera keeps the sample focused on rendering, not input handling.
    const Mat4 view = LookAt({8.0f, 6.0f, 8.0f}, {0.0f, 0.5f, 0.0f}, {0.0f, 1.0f, 0.0f});
    const Mat4 projection = Perspective(45.0f * kPi / 180.0f,
                                        static_cast<float>(app.swapchainExtent.width) /
                                            static_cast<float>(app.swapchainExtent.height),
                                        0.1f,
                                        100.0f);
    const Mat4 viewProjection = Multiply(projection, view);

    for (std::size_t i = 0; i < app.commandBuffers.size(); ++i) {
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

        if (vkBeginCommandBuffer(app.commandBuffers[i], &beginInfo) != VK_SUCCESS) {
            throw std::runtime_error("Failed to begin command buffer.");
        }

        std::array<VkClearValue, 2> clearValues{};
        clearValues[0].color.float32[0] = 0.09f;
        clearValues[0].color.float32[1] = 0.11f;
        clearValues[0].color.float32[2] = 0.15f;
        clearValues[0].color.float32[3] = 1.0f;
        clearValues[1].depthStencil.depth = 1.0f;

        VkRenderPassBeginInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.renderPass = app.renderPass;
        renderPassInfo.framebuffer = app.framebuffers[i];
        renderPassInfo.renderArea.offset = {0, 0};
        renderPassInfo.renderArea.extent = app.swapchainExtent;
        renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
        renderPassInfo.pClearValues = clearValues.data();

        vkCmdBeginRenderPass(app.commandBuffers[i], &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdBindPipeline(app.commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, app.pipeline);

        // Each object reuses the same pipeline. Only the push constants and
        // vertex count change between the plane and the cubes.
        for (const SceneObject& object : kSceneObjects) {
            const PushConstants constants = MakePushConstants(viewProjection, object);
            vkCmdPushConstants(app.commandBuffers[i],
                               app.pipelineLayout,
                               VK_SHADER_STAGE_VERTEX_BIT,
                               0,
                               sizeof(PushConstants),
                               &constants);
            vkCmdDraw(app.commandBuffers[i], object.vertexCount, 1, 0, 0);
        }

        vkCmdEndRenderPass(app.commandBuffers[i]);

        if (vkEndCommandBuffer(app.commandBuffers[i]) != VK_SUCCESS) {
            throw std::runtime_error("Failed to record command buffer.");
        }
    }
}

void CreateSyncObjects(App& app) {
    VkSemaphoreCreateInfo semaphoreInfo{};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    app.renderFinishedPerImage.resize(app.swapchainImages.size(), VK_NULL_HANDLE);
    app.imageInFlight.assign(app.swapchainImages.size(), VK_NULL_HANDLE);

    for (std::size_t i = 0; i < app.swapchainImages.size(); ++i) {
        if (vkCreateSemaphore(app.device, &semaphoreInfo, nullptr, &app.renderFinishedPerImage[i]) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create render-finished semaphore.");
        }
    }

    for (std::size_t i = 0; i < kMaxFramesInFlight; ++i) {
        if (vkCreateSemaphore(app.device, &semaphoreInfo, nullptr, &app.imageAvailable[i]) != VK_SUCCESS ||
            vkCreateFence(app.device, &fenceInfo, nullptr, &app.inFlightFences[i]) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create frame synchronization objects.");
        }
    }
}

void DrawFrame(App& app) {
    vkWaitForFences(app.device, 1, &app.inFlightFences[app.currentFrame], VK_TRUE, UINT64_MAX);

    uint32_t imageIndex = 0;
    const VkResult acquireResult = vkAcquireNextImageKHR(app.device,
                                                         app.swapchain,
                                                         UINT64_MAX,
                                                         app.imageAvailable[app.currentFrame],
                                                         VK_NULL_HANDLE,
                                                         &imageIndex);
    if (acquireResult != VK_SUCCESS && acquireResult != VK_SUBOPTIMAL_KHR) {
        throw std::runtime_error("Failed to acquire swapchain image.");
    }

    if (app.imageInFlight[imageIndex] != VK_NULL_HANDLE) {
        vkWaitForFences(app.device, 1, &app.imageInFlight[imageIndex], VK_TRUE, UINT64_MAX);
    }
    app.imageInFlight[imageIndex] = app.inFlightFences[app.currentFrame];

    vkResetFences(app.device, 1, &app.inFlightFences[app.currentFrame]);

    const VkSemaphore waitSemaphores[] = {app.imageAvailable[app.currentFrame]};
    const VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
    const VkSemaphore signalSemaphores[] = {app.renderFinishedPerImage[imageIndex]};

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = waitSemaphores;
    submitInfo.pWaitDstStageMask = waitStages;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &app.commandBuffers[imageIndex];
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = signalSemaphores;

    if (vkQueueSubmit(app.graphicsQueue, 1, &submitInfo, app.inFlightFences[app.currentFrame]) != VK_SUCCESS) {
        throw std::runtime_error("Failed to submit command buffer.");
    }

    VkPresentInfoKHR presentInfo{};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = signalSemaphores;
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = &app.swapchain;
    presentInfo.pImageIndices = &imageIndex;

    const VkResult presentResult = vkQueuePresentKHR(app.presentQueue, &presentInfo);
    if (presentResult != VK_SUCCESS && presentResult != VK_SUBOPTIMAL_KHR) {
        throw std::runtime_error("Failed to present swapchain image.");
    }

    app.currentFrame = (app.currentFrame + 1) % kMaxFramesInFlight;
}

void Cleanup(App& app) {
    if (app.device != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(app.device);

        for (auto semaphore : app.renderFinishedPerImage) {
            if (semaphore != VK_NULL_HANDLE) {
                vkDestroySemaphore(app.device, semaphore, nullptr);
            }
        }

        for (std::size_t i = 0; i < kMaxFramesInFlight; ++i) {
            if (app.imageAvailable[i] != VK_NULL_HANDLE) {
                vkDestroySemaphore(app.device, app.imageAvailable[i], nullptr);
            }
            if (app.inFlightFences[i] != VK_NULL_HANDLE) {
                vkDestroyFence(app.device, app.inFlightFences[i], nullptr);
            }
        }

        if (app.commandPool != VK_NULL_HANDLE) {
            vkDestroyCommandPool(app.device, app.commandPool, nullptr);
        }

        for (auto framebuffer : app.framebuffers) {
            if (framebuffer != VK_NULL_HANDLE) {
                vkDestroyFramebuffer(app.device, framebuffer, nullptr);
            }
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

        for (auto imageView : app.depthImageViews) {
            if (imageView != VK_NULL_HANDLE) {
                vkDestroyImageView(app.device, imageView, nullptr);
            }
        }
        for (auto image : app.depthImages) {
            if (image != VK_NULL_HANDLE) {
                vkDestroyImage(app.device, image, nullptr);
            }
        }
        for (auto memory : app.depthMemories) {
            if (memory != VK_NULL_HANDLE) {
                vkFreeMemory(app.device, memory, nullptr);
            }
        }

        for (auto imageView : app.swapchainImageViews) {
            if (imageView != VK_NULL_HANDLE) {
                vkDestroyImageView(app.device, imageView, nullptr);
            }
        }

        if (app.swapchain != VK_NULL_HANDLE) {
            vkDestroySwapchainKHR(app.device, app.swapchain, nullptr);
        }

        vkDestroyDevice(app.device, nullptr);
    }

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
    glfwTerminate();
}

void Run() {
    App app;

    try {
        CreateWindow(app);

        if (volkInitialize() != VK_SUCCESS) {
            throw std::runtime_error("Failed to initialize volk.");
        }

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
        CreateFramebuffers(app);
        CreateCommandPool(app);
        CreateCommandBuffers(app);
        CreateSyncObjects(app);

        while (!glfwWindowShouldClose(app.window)) {
            glfwPollEvents();
            DrawFrame(app);
        }

        Cleanup(app);
    } catch (...) {
        Cleanup(app);
        throw;
    }
}

}  // namespace

int main() {
    try {
        Run();
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << exception.what() << '\n';
        return 1;
    }
}
