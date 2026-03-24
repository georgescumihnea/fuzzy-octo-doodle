# Vulkan Triangle

This repo is now intentionally minimal:

- one source file: `src/main.cpp`
- two shaders: `shaders/triangle.vert` and `shaders/triangle.frag`
- one CMake target: `vulkan_triangle`

It is a raw Vulkan triangle, not a mini engine.

## Build

From the repo root:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\invoke_in_vs_dev_shell.ps1 -Command "cmake --preset debug"
powershell -ExecutionPolicy Bypass -File .\scripts\invoke_in_vs_dev_shell.ps1 -Command "cmake --build --preset debug --config Debug"
```

Run the app:

```powershell
Set-Location .\build\debug\bin\Debug
.\vulkan_triangle.exe
```

## Shader compile

If the Vulkan SDK is installed, shaders are compiled automatically during the build.

To compile them directly:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\invoke_in_vs_dev_shell.ps1 -Command "cmake --build --preset debug --config Debug --target compile_shaders"
```

## VS Code

- `Ctrl+Shift+B` builds the sample
- `Run Task -> run-app` runs it
- `F5` starts the debugger

## Important

Even stripped down, a Vulkan triangle is still a few hundred lines. That is normal for Vulkan.

If you want a graphics API where a triangle really is a few lines, use OpenGL, raylib, or SDL first. If you want to learn Vulkan specifically, this smaller sample is a reasonable base.
