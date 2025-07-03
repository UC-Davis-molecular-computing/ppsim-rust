@echo off
REM Build script for ppsim-rust using MSYS2 environment
REM This script ensures that cargo build uses the MSYS2 environment with necessary Unix tools

echo Setting up MSYS2 environment for Rust build...

REM Set MSYS2 environment variables
set MSYSTEM=MINGW64
set PATH=C:\msys64\mingw64\bin;C:\msys64\usr\bin;%PATH%

REM Ensure we're using the GNU toolchain
rustup show active-toolchain

echo Starting cargo build...
cargo build %*

echo Build complete.
pause
