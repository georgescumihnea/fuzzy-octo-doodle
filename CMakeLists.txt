cmake_minimum_required(VERSION 3.30)

project(VulkanCubes LANGUAGES CXX)

# Download small third-party dependencies at configure time.
include(FetchContent)

# Use modern C++ without compiler-specific language extensions.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Keep binaries in predictable `build/.../bin/<Config>` folders.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

foreach(config IN ITEMS Debug Release RelWithDebInfo MinSizeRel)
    string(TOUPPER "${config}" config_upper)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${config_upper} "${CMAKE_BINARY_DIR}/bin/${config}")
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${config_upper} "${CMAKE_BINARY_DIR}/bin/${config}")
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${config_upper} "${CMAKE_BINARY_DIR}/lib/${config}")
endforeach()

if(MSVC)
    # Reasonable warning level for the sample.
    add_compile_options(/W4 /permissive- /EHsc)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# We only need GLFW as a library, not its docs/examples/tests.
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
set(VOLK_INSTALL OFF CACHE BOOL "" FORCE)

# GLFW creates the window. Vulkan-Headers provides official headers.
# volk loads Vulkan function pointers without linking directly to vulkan-1.lib.
FetchContent_Declare(
    glfw
    URL https://github.com/glfw/glfw/archive/refs/tags/3.4.zip
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)

FetchContent_Declare(
    vulkan_headers
    URL https://github.com/KhronosGroup/Vulkan-Headers/archive/refs/tags/vulkan-sdk-1.4.341.0.zip
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)

FetchContent_Declare(
    volk
    URL https://github.com/zeux/volk/archive/refs/tags/vulkan-sdk-1.4.341.0.zip
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)

FetchContent_MakeAvailable(glfw vulkan_headers)
set(VULKAN_HEADERS_INSTALL_DIR "${vulkan_headers_SOURCE_DIR}" CACHE PATH "" FORCE)
FetchContent_MakeAvailable(volk)

# The app is intentionally a single-file 3D sample.
add_executable(vulkan_triangle
    src/main.cpp
)

target_compile_definitions(vulkan_triangle PRIVATE GLFW_INCLUDE_NONE)

if(TARGET volk::volk)
    target_link_libraries(vulkan_triangle PRIVATE glfw volk::volk)
else()
    target_link_libraries(vulkan_triangle PRIVATE glfw volk)
endif()

# Prefer `glslc` from the Vulkan SDK, but fall back to `glslangValidator` if needed.
find_program(GLSLC_EXECUTABLE NAMES glslc HINTS ENV VULKAN_SDK PATH_SUFFIXES Bin bin)
find_program(GLSLANG_VALIDATOR_EXECUTABLE NAMES glslangValidator HINTS ENV VULKAN_SDK PATH_SUFFIXES Bin bin)

set(VULKAN_TRIANGLE_SHADER_COMPILER "")
set(VULKAN_TRIANGLE_SHADER_COMPILER_MODE "")

if(GLSLC_EXECUTABLE)
    set(VULKAN_TRIANGLE_SHADER_COMPILER "${GLSLC_EXECUTABLE}")
    set(VULKAN_TRIANGLE_SHADER_COMPILER_MODE "glslc")
elseif(GLSLANG_VALIDATOR_EXECUTABLE)
    set(VULKAN_TRIANGLE_SHADER_COMPILER "${GLSLANG_VALIDATOR_EXECUTABLE}")
    set(VULKAN_TRIANGLE_SHADER_COMPILER_MODE "glslangValidator")
endif()

set(VULKAN_TRIANGLE_SHADER_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/shaders/triangle.vert"
    "${CMAKE_CURRENT_SOURCE_DIR}/shaders/triangle.frag"
)

if(VULKAN_TRIANGLE_SHADER_COMPILER)
    # Compile shaders into the same folder as the executable so the app can load
    # them with a simple relative path at runtime.
    add_custom_target(compile_shaders ALL
        COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:vulkan_triangle>/shaders"
        COMMAND ${CMAKE_COMMAND}
            -DINPUT_FILE=${CMAKE_CURRENT_SOURCE_DIR}/shaders/triangle.vert
            -DOUTPUT_FILE=$<TARGET_FILE_DIR:vulkan_triangle>/shaders/triangle.vert.spv
            -DCOMPILER=${VULKAN_TRIANGLE_SHADER_COMPILER}
            -DMODE=${VULKAN_TRIANGLE_SHADER_COMPILER_MODE}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/CompileShader.cmake
        COMMAND ${CMAKE_COMMAND}
            -DINPUT_FILE=${CMAKE_CURRENT_SOURCE_DIR}/shaders/triangle.frag
            -DOUTPUT_FILE=$<TARGET_FILE_DIR:vulkan_triangle>/shaders/triangle.frag.spv
            -DCOMPILER=${VULKAN_TRIANGLE_SHADER_COMPILER}
            -DMODE=${VULKAN_TRIANGLE_SHADER_COMPILER_MODE}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/CompileShader.cmake
        DEPENDS ${VULKAN_TRIANGLE_SHADER_SOURCES}
        VERBATIM
    )
else()
    message(WARNING "No shader compiler found. Install the Vulkan SDK to compile shaders.")
    add_custom_target(compile_shaders
        COMMAND ${CMAKE_COMMAND} -E echo "Shader compilation skipped. Install the Vulkan SDK to get glslc or glslangValidator."
        VERBATIM
    )
endif()

# Building the executable also builds the shaders first.
add_dependencies(vulkan_triangle compile_shaders)
