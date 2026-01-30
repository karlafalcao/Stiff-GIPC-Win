# Troubleshooting Guide - Stiff-GIPC Build

This guide covers detailed troubleshooting for common build issues. For basic build instructions, see [BUILD_GUIDE.md](BUILD_GUIDE.md).

---

## 🔧 CMake Issues

### CMake Not Found After Installation

**Symptoms:**
- `cmake --version` returns "command not found"
- CMake was installed but PowerShell can't find it

**Solutions:**

1. **Restart your computer** - This ensures PATH variables are fully updated
2. **Open a new terminal window** - Don't use an old one that was open before installation
3. **Verify PATH:**
   ```powershell
   $env:PATH -split ';' | Select-String cmake
   ```
   If nothing appears, CMake is not in your PATH.

4. **Reinstall using the official installer** and make sure to check "Add CMake to the system PATH for all users"

5. **Manually add to PATH:**
   - Press `Win + X` → **System** → **Advanced system settings** → **Environment Variables**
   - Add: `C:\Program Files\CMake\bin` to your PATH
   - Restart your computer

### Multiple CMake Installations

**Symptoms:**
- `cmake --version` shows an older version than expected
- Different versions detected in different terminals

**Solutions:**

1. **Check which CMake is being used:**
   ```powershell
   where.exe cmake
   ```
   This shows all CMake installations in your PATH.

2. **Remove old installations** or update your PATH to prioritize the new one

3. **Use full path** to the specific CMake version you want:
   ```powershell
   & "C:\Program Files\CMake\bin\cmake.exe" --version
   ```

### Permission Denied Errors During CMake Installation

**Symptoms:**
- Installer fails with permission errors
- Chocolatey installation fails

**Solutions:**

1. **Run installer as Administrator:**
   - Right-click the CMake installer `.msi` file
   - Select "Run as administrator"

2. **For Chocolatey:**
   - Make sure you're running PowerShell as Administrator
   - Close any other Chocolatey processes

---

## 🔧 CUDA Issues

### "No CUDA toolset found" (Visual Studio generator)

**Symptoms:**
- CMake configuration fails with: `No CUDA toolset found`
- You ran `cmake ..` without `-G Ninja` and CMake chose the Visual Studio generator

**Cause:** The Visual Studio generator expects CUDA to be installed as a VS “platform toolset”. That integration may be missing or not support your VS version (e.g. VS 2026).

**Solution:** Use the **Ninja** generator and set the CUDA compiler and toolchain with **explicit paths** (so CMake does not receive a literal `$env:...`). Run from **Developer PowerShell for VS** so MSVC is in PATH, and install Ninja: `winget install Ninja-build.Ninja`. Then:

```powershell
$Toolchain = "C:/Users/karla/coding/Stiff-GIPC/vcpkg/scripts/buildsystems/vcpkg.cmake"
$cuda = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
$env:CUDA_PATH = $cuda
$env:PATH = "$cuda\bin;$env:PATH"
cd C:\Users\karla\coding\Stiff-GIPC\build
Remove-Item CMakeCache.txt, CMakeFiles -Recurse -Force -ErrorAction SilentlyContinue
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release -DCUDAToolkit_ROOT="$cuda" -DCMAKE_CUDA_COMPILER="$cuda\bin\nvcc.exe" -DCMAKE_TOOLCHAIN_FILE="$Toolchain"
```

Or use the project build script (it sets PATH, finds Ninja, and runs in a proper environment): `.\build.ps1`

### nvcc: "Cannot find compiler 'cl.exe' in PATH"

**Symptoms:**
- CMake configuration fails with: `nvcc fatal : Cannot find compiler 'cl.exe' in PATH`
- "The CXX compiler identification is unknown" then the CUDA compiler check fails

**Cause:** You ran `cmake` from a **regular** PowerShell or Command Prompt. On Windows, nvcc uses MSVC’s `cl.exe` for host code; `cl.exe` is only in PATH in a **Developer** environment (Developer PowerShell for VS, or after running `vcvars64.bat`).

**Solution:** Either:

1. **Use the project build script** (it runs vcvars64 then cmake): from project root run `.\build.ps1`
2. **Open "Developer PowerShell for Visual Studio"** (Start menu), then run your cmake commands there, or
3. **From regular PowerShell**, use the "Option A" block in [BUILD_WINDOWS.md](BUILD_WINDOWS.md) under "Manual configure" — it runs `vcvars64.bat` via `cmd /c` before cmake and build.

### CUDA Not Found by CMake

**Symptoms:**
- CMake configuration fails with "CUDA not found"
- `nvcc --version` works but CMake can't detect CUDA

**Solutions:**

1. **Set CUDA path explicitly:**
   ```powershell
   $env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
   $env:CUDAToolkit_ROOT = $env:CUDA_PATH
   $env:CUDA_PATH_V12_4 = $env:CUDA_PATH
   $env:PATH = "$env:CUDA_PATH\bin;$env:PATH"
   ```

2. **Add CUDA to PATH permanently:**
   - Add: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin`
   - Restart terminal after adding to PATH

3. **Explicitly pass CUDA path to CMake:**
   ```powershell
   & "C:\Program Files\CMake\bin\cmake.exe" .. `
       -DCUDAToolkit_ROOT="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4" `
       -DCUDA_TOOLKIT_ROOT_DIR="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4" `
       -DCMAKE_CUDA_COMPILER="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin\nvcc.exe"
   ```

### MSBuild Error: "The CUDA Toolkit v12.4 directory '' does not exist"

**Symptoms:**
- CMake configuration succeeds
- Build fails with: `The CUDA Toolkit v12.4 directory '' does not exist`
- This is an MSBuild-specific error

**Solutions:**

1. **Set additional environment variables that MSBuild looks for:**
   ```powershell
   $env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
   $env:CUDAToolkit_ROOT = $env:CUDA_PATH
   $env:CUDA_PATH_V12_4 = $env:CUDA_PATH  # MSBuild specifically looks for this
   $env:PATH = "$env:CUDA_PATH\bin;$env:PATH"
   ```

2. **Add CUDA to PATH before running CMake:**
   ```powershell
   # Set PATH first
   $env:PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin;$env:PATH"
   
   # Then configure
   & "C:\Program Files\CMake\bin\cmake.exe" .. `
       -DCUDAToolkit_ROOT="$env:CUDA_PATH" `
       -DCUDA_TOOLKIT_ROOT_DIR="$env:CUDA_PATH" `
       -DCMAKE_CUDA_COMPILER="$env:CUDA_PATH\bin\nvcc.exe"
   ```

3. **Clean and reconfigure:**
   ```powershell
   Remove-Item CMakeCache.txt -ErrorAction SilentlyContinue
   Remove-Item -Recurse CMakeFiles -ErrorAction SilentlyContinue
   # Then reconfigure with all environment variables set
   ```

### CUDA Version Mismatch

**Symptoms:**
- CMake detects wrong CUDA version
- Multiple CUDA versions installed

**Solutions:**

1. **Check installed CUDA versions:**
   ```powershell
   Get-ChildItem "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA" | Select-Object Name
   ```

2. **Uninstall unwanted CUDA versions** via Windows Settings → Apps

3. **Explicitly set the CUDA version you want** using environment variables

---

## 🔧 Dependency Issues

### Package Not Found Errors

**Symptoms:**
- CMake fails with "Could not find package X"
- vcpkg packages installed but CMake can't find them

**Solutions:**

1. **Verify vcpkg toolchain is set:**
   ```powershell
   echo $env:CMAKE_TOOLCHAIN_FILE
   ```
   Should show: `C:\dev\vcpkg\scripts\buildsystems\vcpkg.cmake`

2. **Check installed packages:**
   ```powershell
   cd C:\dev\vcpkg
   .\vcpkg list
   ```
   Should show: eigen3, freeglut, glew, nlohmann-json, tbb

3. **Reinstall missing packages:**
   ```powershell
   .\vcpkg install <package-name>:x64-windows
   ```

4. **Verify package installation:**
   ```powershell
   .\vcpkg list | Select-String "<package-name>"
   ```

### TBB Not Found (Common Issue)

**Symptoms:**
- CMake error: "Could not find TBB"
- TBB is required but sometimes missing from dependency lists

**Solution:**
```powershell
.\vcpkg install tbb:x64-windows
```

**Note:** TBB (Threading Building Blocks) is required but was missing from the original README.md dependency list.

### vcpkg Installation Fails

**Symptoms:**
- `.\bootstrap-vcpkg.bat` fails
- `.\vcpkg install` commands fail

**Solutions:**

1. **Run PowerShell as Administrator**

2. **Check internet connection** - packages download during installation

3. **Check Git is installed:**
   ```powershell
   git --version
   ```
   If not found, install Git for Windows: https://git-scm.com/download/win

4. **Try reinstalling vcpkg:**
   ```powershell
   Remove-Item -Recurse -Force C:\dev\vcpkg
   git clone https://github.com/Microsoft/vcpkg.git C:\dev\vcpkg
   cd C:\dev\vcpkg
   .\bootstrap-vcpkg.bat
   ```

---

## 🔧 Build Errors

### "target_compile_features no known features for CXX compiler MSVC"

**Symptoms:**
- Warning during CMake configuration
- `target_compile_features no known features for CXX compiler MSVC`

**Status:** ✅ **This is a known warning and can be ignored**

**Explanation:**
- MSVC 2019 has limited support for `target_compile_features`
- The project still builds successfully despite this warning
- This is a CMake/MSVC compatibility issue, not a project issue

**Action:** None required - the build will succeed.

### Compilation Errors in femEnergy.cu

**Symptoms:**
- `identifier "assert" is undefined`
- `integer conversion resulted in a change of sign` warnings/errors

**Status:** ✅ **FIXED** - These fixes have been applied to the codebase

**Fixes Applied:**

1. **Added missing include** (line 12):
   ```cpp
   #include <assert.h>
   ```
   - **Reason:** CUDA 12.4 requires explicit include for `assert()` function

2. **Fixed unsigned comparison** (lines 2370, 2824):
   ```cpp
   // Before: if(adj.y == -1)
   // After:
   if(adj.y == (unsigned int)(-1))
   ```
   - **Reason:** CUDA 12.4 has stricter type checking for unsigned/signed comparisons

**If you still see these errors:**
- Make sure you have the latest code with these fixes
- Check that you're using CUDA 12.4 (not 11.8 or 13.1)

### Build Warnings (Non-Fatal)

**Common Warnings:**

1. **Integer conversion warnings:**
   ```
   warning #68-D: integer conversion resulted in a change of sign
   ```
   - **Status:** ✅ Normal - These are non-fatal
   - **Action:** Can be ignored

2. **Unused variable warnings:**
   ```
   warning #177-D: variable "X" was declared but never referenced
   ```
   - **Status:** ✅ Normal - These are non-fatal
   - **Action:** Can be ignored

3. **Linker warnings:**
   ```
   warning LNK4098: defaultlib 'LIBCMT' conflita com uso de outras bibliotecas
   ```
   - **Status:** ✅ Normal - Common with CUDA projects
   - **Action:** Can be ignored

**All warnings are non-fatal** - the executable is fully functional despite these warnings.

---

## 🔧 Debugging Build Failures

### Redirect Build Output to File

**When to use:** Build errors are truncated or you need to review the full output

**Method:**
```powershell
& "C:\Program Files\CMake\bin\cmake.exe" --build . --config Release 2>&1 | Tee-Object -FilePath build_output.log
```

Then check `build_output.log` for the full error messages.

### Build Specific Target

**When to use:** To isolate which file is causing the error

**Method:**
```powershell
& "C:\Program Files\CMake\bin\cmake.exe" --build . --config Release --target gipc
```

**Note:** Not all projects support building individual targets, but worth trying.

### Check Last Error in Output

**Method:**
```powershell
# Build and filter for errors
& "C:\Program Files\CMake\bin\cmake.exe" --build . --config Release 2>&1 | Select-String -Pattern "error|Error|ERROR" | Select-Object -Last 20
```

### Clean Build

**When to use:** After changing CUDA versions, CMake configuration, or when builds behave unexpectedly

**Method:**
```powershell
# Remove build directory completely
Remove-Item -Recurse -Force build

# Or just clear CMake cache
cd build
Remove-Item CMakeCache.txt -ErrorAction SilentlyContinue
Remove-Item -Recurse CMakeFiles -ErrorAction SilentlyContinue
```

Then reconfigure and rebuild.

---

## 🔧 Version Compatibility Issues

### CUDA 11.8 Compatibility Issues

**Symptoms:**
- Build fails with: `namespace "cuda::std" has no member "equal_to"`
- Errors in `muda/src/muda/cub/device/device_scan.h`

**Root Cause:**
- CUDA 11.8's CUB library doesn't have `cuda::std::equal_to`
- The muda library's CUB wrapper expects features from newer CUDA versions

**Solution:** ⚠️ **Use CUDA 12.4 instead**

CUDA 11.8 is deprecated for this project. Upgrade to CUDA 12.4.

### CUDA 13.1 + Visual Studio 2026 (VS 18) — unsupported / cudafe++ crash

**Symptoms:**
- CMake compiler check fails with: `#error: unsupported Microsoft Visual Studio version! Only the versions between 2019 and 2022 (inclusive) are supported!`
- Or: `'cudafe++' died with status 0xC0000005 (ACCESS_VIOLATION)` when using `-allow-unsupported-compiler`

**Root Cause:**
- CUDA 13.1’s `host_config.h` only allows VS 2019–2022. VS 2026 (18) is not supported.
- With `-allow-unsupported-compiler`, nvcc’s `cudafe++` can crash (ACCESS_VIOLATION) with VS 2026.

**Solution:** ⚠️ **Use CUDA 12.4**

Point the build at CUDA 12.4 (you can keep 13.1 installed). From the project root:

```powershell
$cuda = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
$env:CUDA_PATH = $cuda
$env:PATH = "$cuda\bin;$env:PATH"
cd build
Remove-Item CMakeCache.txt, CMakeFiles -Recurse -Force -ErrorAction SilentlyContinue
# Then re-run your cmake .. -G Ninja ... with -DCMAKE_CUDA_COMPILER="$cuda\bin\nvcc.exe" and -DCUDAToolkit_ROOT="$cuda"
```

If you only have VS 2026 and CUDA 12.4 still fails the compiler check, install **Visual Studio 2022 Build Tools** alongside VS 2026 and use **Developer PowerShell for VS 2022** to build (so `cl.exe` is VS 2022).

### CUDA 13.1 Compatibility Issues (template / code errors)

**Symptoms:**
- Build fails with template specialization errors
- Errors like: `declaration is incompatible with "muda::ComputeGraphVar<...>::operator=(...)"`
- Errors in `graph_var_view.inl`, `graph_buffer_view.inl`, etc.

**Root Cause:**
- CUDA 13.1 has stricter template checking
- muda library hasn't been updated for CUDA 13.1 compatibility

**Solution:** ⚠️ **Use CUDA 12.4 instead**

CUDA 13.1 is deprecated for this project. Use CUDA 12.4.

### Visual Studio Version Issues

**Symptoms:**
- C++17 features not recognized
- Standard library compatibility errors

**Solutions:**

1. **Use Visual Studio 2019 or later** (recommended: 2019 or 2022)

2. **Install C++ workload:**
   - Open Visual Studio Installer
   - Modify your installation
   - Ensure "Desktop development with C++" workload is installed

3. **Visual Studio 2022** has better C++17 and CUDA compatibility than 2019

---

## 🔧 Environment and Path Issues

### Environment Variables Not Persisting

**Symptoms:**
- Environment variables set in one terminal don't work in another
- Need to set variables every time

**Solutions:**

1. **Set permanently via System Properties:**
   - Press `Win + X` → **System** → **Advanced system settings**
   - Click **Environment Variables**
   - Add variables under "User variables" or "System variables"
   - Restart terminal after adding

2. **Create a setup script:**
   ```powershell
   # Save as setup-env.ps1
   $env:CMAKE_TOOLCHAIN_FILE = "C:\dev\vcpkg\scripts\buildsystems\vcpkg.cmake"
   $env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
   $env:CUDAToolkit_ROOT = $env:CUDA_PATH
   $env:CUDA_PATH_V12_4 = $env:CUDA_PATH
   $env:PATH = "$env:CUDA_PATH\bin;$env:PATH"
   ```
   Then run: `. .\setup-env.ps1` before building

### Git Not Recognized

**Symptoms:**
- `git is not recognized as an internal or external command`
- vcpkg bootstrap fails

**Solution:**
- Install Git for Windows: https://git-scm.com/download/win
- Restart terminal after installation
- Verify: `git --version`

---

## 🔧 Specific Error Messages

### "File not found" During Build

**Possible Causes:**
1. Missing source files
2. Incorrect project path
3. Corrupted repository

**Solutions:**
1. Verify repository is complete: `git status`
2. Check you're in the correct directory
3. Try cloning the repository fresh

### Linking Errors

**Symptoms:**
- Compilation succeeds but linking fails
- Missing library errors

**Solutions:**
1. Verify all dependencies are installed via vcpkg
2. Check vcpkg toolchain is set correctly
3. Ensure CUDA libraries are in PATH
4. Try a clean build

### CMake Configuration Fails

**Symptoms:**
- `cmake ..` fails before building
- Dependencies not found

**Solutions:**
1. Verify all prerequisites are installed (CMake, CUDA, vcpkg)
2. Check environment variables are set
3. Review error messages for specific missing components
4. See dependency-specific troubleshooting above

---

## 📊 Detailed Test Results Reference

**Successfully Tested Environment:**
- Windows 11
- CMake 4.2.1
- CUDA 12.4 ✅
- MSVC 19.28.29910.0 (Visual Studio 2019)
- vcpkg with all dependencies installed

**Package Versions (from successful build):**
- Eigen3: 5.0.1
- freeglut: 3.8.0
- GLEW: 2.3.0#1
- nlohmann-json: 3.12.0#2
- TBB: 2022.3.0

**Files That Compiled Successfully:**
- ✅ `femEnergy.cu` - Compiled with minor warnings (non-fatal)
- ✅ `gipc.cu` - Compiled successfully
- ✅ `gl_main.cu` - Compiled successfully
- ✅ All other CUDA and C++ source files - Compiled successfully

**Expected CMake Configuration Output:**
```
-- Found CUDAToolkit: C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/include (found version "12.4.x")
-- Found GLUT: optimized;C:/dev/vcpkg/installed/x64-windows/lib/freeglut.lib
-- Found OpenGL: opengl32
-- Found nlohmann_json: C:/dev/vcpkg/installed/x64-windows/share/nlohmann_json/nlohmann_jsonConfig.cmake (found version "3.12.0")
-- Configuring done
-- Generating done
```

---

## 🆘 Still Having Issues?

If you've tried all the troubleshooting steps above and still have issues:

1. **Check the project's GitHub Issues:**
   - https://github.com/KemengHuang/Stiff-GIPC/issues
   - Search for similar issues
   - Check if there are known fixes or workarounds

2. **Verify your environment matches the tested configuration:**
   - Windows 11
   - CMake 4.2.1
   - CUDA 12.4 (not 11.8 or 13.1)
   - Visual Studio 2019 or 2022
   - All vcpkg dependencies installed

3. **Try a completely clean build:**
   ```powershell
   # Remove everything
   Remove-Item -Recurse -Force build
   
   # Set all environment variables
   $env:CMAKE_TOOLCHAIN_FILE = "C:\dev\vcpkg\scripts\buildsystems\vcpkg.cmake"
   $env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
   $env:CUDAToolkit_ROOT = $env:CUDA_PATH
   $env:CUDA_PATH_V12_4 = $env:CUDA_PATH
   $env:PATH = "$env:CUDA_PATH\bin;$env:PATH"
   
   # Reconfigure and rebuild
   mkdir build
   cd build
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

4. **Create a detailed issue report:**
   - Include your environment (OS, CMake version, CUDA version, Visual Studio version)
   - Include the full error output (use `Tee-Object` to capture it)
   - Include what troubleshooting steps you've already tried

---

**Last Updated:** Based on successful build with CUDA 12.4, CMake 4.2.1, Visual Studio 2019
