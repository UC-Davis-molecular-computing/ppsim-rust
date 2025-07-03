# Build script for ppsim-rust using MSYS2 environment
# This script ensures that cargo build uses the MSYS2 environment with necessary Unix tools

Write-Host "Setting up MSYS2 environment for Rust build..." -ForegroundColor Green

# Set MSYS2 environment variables
$env:MSYSTEM = "MINGW64"
$env:PATH = "C:\msys64\mingw64\bin;C:\msys64\usr\bin;" + $env:PATH

# Set compiler environment variables for configure scripts (use Unix-style paths)
$env:CC = "gcc"
$env:CXX = "g++"
$env:CFLAGS = "-O2"
$env:CXXFLAGS = "-O2"

# Show active toolchain
Write-Host "Active Rust toolchain:" -ForegroundColor Yellow
rustup show active-toolchain

# Verify MSYS2 tools are available
Write-Host "Checking for required build tools..." -ForegroundColor Yellow
try {
    $shPath = & C:\msys64\usr\bin\bash.exe --login -c "which sh" 2>$null
    $makePath = & C:\msys64\usr\bin\bash.exe --login -c "which make" 2>$null
    $gccPath = & C:\msys64\usr\bin\bash.exe --login -c "export MSYSTEM=MINGW64 && which gcc" 2>$null
    
    Write-Host "sh: $shPath" -ForegroundColor Cyan
    Write-Host "make: $makePath" -ForegroundColor Cyan
    Write-Host "gcc: $gccPath" -ForegroundColor Cyan
    Write-Host "CC env var: $env:CC" -ForegroundColor Cyan
    Write-Host "CXX env var: $env:CXX" -ForegroundColor Cyan
} catch {
    Write-Host "Warning: Some build tools may not be available" -ForegroundColor Red
}

Write-Host "Starting cargo build..." -ForegroundColor Green
cargo build @args

Write-Host "Build complete." -ForegroundColor Green
