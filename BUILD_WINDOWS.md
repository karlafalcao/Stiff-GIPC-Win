# Building Stiff-GIPC on Windows

## Prerequisites

- **MSVC compiler** — Visual Studio **Build Tools 2022** (no full IDE) or full VS 2022. CUDA on Windows requires it. [Build Tools download](https://visualstudio.microsoft.com/visual-cpp-build-tools/) → select workload **Desktop development with C++**.
- **Ninja** — `winget install Ninja-build.Ninja` (optional; script can use the one bundled with VS if present).
- **CUDA Toolkit** 11.0+ (12.4 recommended) — [NVIDIA CUDA](https://developer.nvidia.com/cuda-downloads)
- **CMake** 3.18+ — [cmake.org](https://cmake.org/download/) or `winget install Kitware.CMake`
- **Git** (for cloning vcpkg)

## Step 1: Install dependencies

**Recommended: use vcpkg** (reliable packages for all dependencies). From the project root:

```powershell
.\setup-dependencies.ps1 -VcpkgInstallPath "c:\Users\karla\coding\Stiff-GIPC\vcpkg"
```

This installs vcpkg (if needed) and the required packages: eigen3, freeglut, glew, nlohmann-json, tbb. Then set the toolchain and build:

```powershell
$env:CMAKE_TOOLCHAIN_FILE = "c:\Users\karla\coding\Stiff-GIPC\vcpkg\scripts\buildsystems\vcpkg.cmake"
.\build.ps1
```

**Chocolatey:** Only **eigen** is available and works. The packages `freeglut`, `glew`, `nlohmann-json`, and `tbb` are either not on the Chocolatey community feed or use broken download URLs, so Chocolatey is not sufficient for a full install. Use vcpkg instead.

**Manual:** Install Eigen3, FreeGLUT, GLEW, nlohmann-json, and TBB yourself and set `CMAKE_PREFIX_PATH` so CMake can find them.

Required libraries:

| Library        | Usage               |
|----------------|---------------------|
| Eigen3         | Matrix math         |
| FreeGLUT       | Visualization (GLUT)|
| GLEW           | OpenGL extensions   |
| nlohmann-json  | JSON parsing        |
| TBB            | Threading           |

## Step 2: Build with CMake

From the project root:

```powershell
.\build.ps1
```

The script uses **Ninja** and the MSVC compiler (no Visual Studio IDE). It finds CUDA and Build Tools/VS, uses the vcpkg toolchain if set, and builds.

Options:

- **Clean configure + build:** `.\build.ps1 -Clean`
- **Debug build:** `.\build.ps1 -Configuration Debug`

Output: `build\gipc.exe`

`build.ps1` auto-detects vcpkg if it’s in the project folder or common locations; you can also set `$env:CMAKE_TOOLCHAIN_FILE` as above.

### Manual configure (without build.ps1)

If you run `cmake` yourself, you **must use the Ninja generator** and pass the CUDA paths. **nvcc needs `cl.exe` in PATH** — either run from **Developer PowerShell for VS** or use the script below so `vcvars64.bat` is run first.

Replace `C:\Users\karla\coding\Stiff-GIPC` if needed. **Use CUDA 12.4** (not 13.1) — see [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for VS 2026 + CUDA 13.1 issues.

**Option A — from regular PowerShell (set CUDA in PowerShell, then run cmake inside cmd with vcvars64):**

CUDA must be in PATH before `cmd` starts so the child process has it; `vcvars64` then adds MSVC (`cl.exe`) in the same session. In `cmd`, `%PATH%` is expanded when the line is parsed, so we do not set PATH inside the same `cmd /c` as `call vcvars64` — we set it in PowerShell instead.

```powershell
$ProjectRoot = "C:\Users\karla\coding\Stiff-GIPC"
$Toolchain = "$ProjectRoot\vcpkg\scripts\buildsystems\vcpkg.cmake"
$cuda = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
if (-not (Test-Path $cuda)) { $cuda = (Get-ChildItem "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA" -Directory | Sort-Object Name -Descending)[0].FullName }

$env:CUDA_PATH = $cuda
$env:PATH = "$cuda\bin;$env:PATH"
$env:CMAKE_TOOLCHAIN_FILE = $Toolchain

$vsWhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$VsPath = & $vsWhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
$vcvars64 = "$VsPath\VC\Auxiliary\Build\vcvars64.bat"
$cmakeExe = (Get-Command cmake -ErrorAction SilentlyContinue).Source
if (-not $cmakeExe) { $cmakeExe = "C:\Program Files\CMake\bin\cmake.exe" }

mkdir $ProjectRoot\build -Force
cd $ProjectRoot\build
Remove-Item CMakeCache.txt, CMakeFiles -Recurse -Force -ErrorAction SilentlyContinue

cmd /c "call `"$vcvars64`" >nul && `"$cmakeExe`" .. -G Ninja -DCMAKE_BUILD_TYPE=Release -DCUDAToolkit_ROOT=`"$cuda`" -DCMAKE_CUDA_COMPILER=`"$cuda\bin\nvcc.exe`" -DCMAKE_TOOLCHAIN_FILE=`"$Toolchain`""
cmd /c "call `"$vcvars64`" >nul && `"$cmakeExe`" --build . --config Release"
```

**Option B — from Developer PowerShell for VS** (Start menu → “Developer PowerShell for VS 2022” or your VS version). Then:

```powershell
$ProjectRoot = "C:\Users\karla\coding\Stiff-GIPC"
$Toolchain = "$ProjectRoot\vcpkg\scripts\buildsystems\vcpkg.cmake"
$cuda = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
$env:CUDA_PATH = $cuda
$env:PATH = "$cuda\bin;$env:PATH"
$env:CMAKE_TOOLCHAIN_FILE = $Toolchain
cd $ProjectRoot\build
Remove-Item CMakeCache.txt, CMakeFiles -Recurse -Force -ErrorAction SilentlyContinue
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release -DCUDAToolkit_ROOT="$cuda" -DCMAKE_CUDA_COMPILER="$cuda\bin\nvcc.exe" -DCMAKE_TOOLCHAIN_FILE="$Toolchain"
cmake --build . --config Release
```

## Troubleshooting

- **Eigen3 / GLEW / GLUT / nlohmann_json / TBB not found:** Use vcpkg (Step 1) or set `CMAKE_PREFIX_PATH` to your install prefix(es). Chocolatey does not provide working packages for freeglut, glew, nlohmann-json, or tbb.
- **CUDA not found:** Install CUDA Toolkit and set `CUDA_PATH` (e.g. `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4`), or ensure it’s on PATH.
- **MSVC compiler not found:** Install [Visual Studio Build Tools 2022](https://visualstudio.microsoft.com/visual-cpp-build-tools/) and select **Desktop development with C++** (no full IDE required).
- **CMake/other errors:** See [TROUBLESHOOTING.md](TROUBLESHOOTING.md).
