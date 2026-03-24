# Vulkan Fire Sample

This is a small Vulkan + GLFW sample that renders a simple 3D scene with a free camera and a fire particle effect in the middle. It is a lightweight playground, not a full engine.

Build and run it from the repo root:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\invoke_in_vs_dev_shell.ps1 -Command "cmake --preset debug"
powershell -ExecutionPolicy Bypass -File .\scripts\invoke_in_vs_dev_shell.ps1 -Command "cmake --build --preset debug --config Debug"
Set-Location .\build\debug\bin\Debug
.\vulkan_triangle.exe
```

If the Vulkan SDK is installed, the shaders compile during the build. Controls are simple: `WASD` to move, mouse to look, `Space` and `Shift` for up/down, `Escape` to release the cursor, and left click to capture it again.
