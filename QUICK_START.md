# Quick Start - Stiff-GIPC Build

**For complete instructions, see [BUILD_GUIDE.md](BUILD_GUIDE.md)**

## Essential Commands

```powershell
# 1. Set environment variables
$env:CMAKE_TOOLCHAIN_FILE = "C:\dev\vcpkg\scripts\buildsystems\vcpkg.cmake"
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
$env:CUDAToolkit_ROOT = $env:CUDA_PATH
$env:CUDA_PATH_V12_4 = $env:CUDA_PATH
$env:PATH = "$env:CUDA_PATH\bin;$env:PATH"

# 2. Build
cd Stiff-GIPC
mkdir build -Force
cd build
Remove-Item CMakeCache.txt -ErrorAction SilentlyContinue
Remove-Item -Recurse CMakeFiles -ErrorAction SilentlyContinue

& "C:\Program Files\CMake\bin\cmake.exe" .. `
    -DCMAKE_BUILD_TYPE=Release `
    -DCMAKE_TOOLCHAIN_FILE="$env:CMAKE_TOOLCHAIN_FILE" `
    -DCUDAToolkit_ROOT="$env:CUDA_PATH" `
    -DCUDA_TOOLKIT_ROOT_DIR="$env:CUDA_PATH" `
    -DCMAKE_CUDA_COMPILER="$env:CUDA_PATH\bin\nvcc.exe" `
    -G "Visual Studio 16 2019" `
    -A x64

& "C:\Program Files\CMake\bin\cmake.exe" --build . --config Release
```

**Requirements:** CMake >= 3.18, CUDA 12.4 ✅, Visual Studio 2019+, vcpkg with dependencies

**Executable:** `build\Release\gipc.exe`
