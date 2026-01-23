# Setup script for Stiff-GIPC dependencies
# This script helps install vcpkg and required dependencies

Write-Host "Stiff-GIPC Dependency Setup" -ForegroundColor Green
Write-Host "============================" -ForegroundColor Green
Write-Host ""

# Check if vcpkg exists
$vcpkgPaths = @(
    "C:\vcpkg",
    "C:\dev\vcpkg", 
    "C:\tools\vcpkg",
    "$env:USERPROFILE\vcpkg",
    "$env:LOCALAPPDATA\vcpkg"
)

$vcpkgFound = $false
$vcpkgPath = $null

foreach ($path in $vcpkgPaths) {
    if (Test-Path (Join-Path $path "vcpkg.exe")) {
        $vcpkgPath = $path
        $vcpkgFound = $true
        Write-Host "Found vcpkg at: $path" -ForegroundColor Green
        break
    }
}

if (-not $vcpkgFound) {
    Write-Host "vcpkg not found. Let's install it!" -ForegroundColor Yellow
    Write-Host ""
    
    # Ask where to install
    $installPath = Read-Host "Where would you like to install vcpkg? (default: C:\dev\vcpkg)"
    if ([string]::IsNullOrWhiteSpace($installPath)) {
        $installPath = "C:\dev\vcpkg"
    }
    
    $parentDir = Split-Path $installPath -Parent
    if (-not (Test-Path $parentDir)) {
        Write-Host "Creating directory: $parentDir" -ForegroundColor Cyan
        New-Item -ItemType Directory -Path $parentDir -Force | Out-Null
    }
    
    Write-Host "Cloning vcpkg repository..." -ForegroundColor Cyan
    Set-Location $parentDir
    git clone https://github.com/Microsoft/vcpkg.git (Split-Path $installPath -Leaf)
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed to clone vcpkg. Please install git first." -ForegroundColor Red
        exit 1
    }
    
    $vcpkgPath = $installPath
    Write-Host "Bootstraping vcpkg..." -ForegroundColor Cyan
    Set-Location $vcpkgPath
    .\bootstrap-vcpkg.bat
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed to bootstrap vcpkg." -ForegroundColor Red
        exit 1
    }
    
    Write-Host "vcpkg installed successfully!" -ForegroundColor Green
} else {
    Write-Host "Using existing vcpkg installation." -ForegroundColor Green
}

Write-Host ""
Write-Host "Installing required packages..." -ForegroundColor Cyan
Write-Host "This may take several minutes..." -ForegroundColor Yellow
Write-Host ""

Set-Location $vcpkgPath

# Install packages
$packages = @("eigen3", "freeglut", "glew", "nlohmann-json", "tbb")
foreach ($package in $packages) {
    Write-Host "Installing $package..." -ForegroundColor Cyan
    .\vcpkg install "${package}:x64-windows"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Warning: Failed to install $package" -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "Setting up environment..." -ForegroundColor Cyan

# Set toolchain file
$toolchainFile = Join-Path $vcpkgPath "scripts\buildsystems\vcpkg.cmake"
$env:CMAKE_TOOLCHAIN_FILE = $toolchainFile

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "vcpkg toolchain file: $toolchainFile" -ForegroundColor Cyan
Write-Host ""
Write-Host "To make this permanent, add this to your environment variables:" -ForegroundColor Yellow
Write-Host "  CMAKE_TOOLCHAIN_FILE = $toolchainFile" -ForegroundColor Cyan
Write-Host ""
Write-Host "Or run this command in your PowerShell session:" -ForegroundColor Yellow
Write-Host "  `$env:CMAKE_TOOLCHAIN_FILE = `"$toolchainFile`"" -ForegroundColor Cyan
Write-Host ""
Write-Host "Now you can build the project!" -ForegroundColor Green
Write-Host "  cd C:\Users\Pichau\karla\Stiff-GIPC" -ForegroundColor Cyan
Write-Host "  .\build.ps1" -ForegroundColor Cyan
