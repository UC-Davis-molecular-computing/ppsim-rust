# Building Rug Crate on Windows - SUCCESS! 🎉

**Status**: ✅ **CONFIRMED WORKING!** The `rug` crate successfully builds on Windows using MSYS2 with a small patch to the GMP configure script. Build completed in ~13-14 minutes and all high-precision arithmetic functions including `ln_gamma` work perfectly.

## Test Results ✅

Successfully tested on Windows with the following output:
```
🧪 Testing Rug crate with high-precision arithmetic on Windows...
x = 2.5000000000000000000000000000000
ln_gamma(2.5) = 2.8468287047291915963249466968273e-1
High precision number: 1.2345678901234566904321354741114191710948944091796875000000000
gamma(2.5) = 1.3293403881791370204736256125055
✅ SUCCESS! High-precision arithmetic with ln_gamma is working correctly on Windows!
```

## Problem and Solution Summary

The `rug` crate provides Rust bindings for the GMP (GNU Multiple Precision Arithmetic Library) and MPFR (Multiple Precision Floating-Point Reliable Library). On Windows, the `gmp-mpfr-sys` crate builds these libraries from source, but the GMP configure script has a bug that causes it to fail with modern GCC versions (15.1.0+).

**The issue**: GMP's "long long reliability test 1" declares `void g(){}` but then calls `g()` with 6 arguments, causing a compilation error with strict modern compilers.

**The solution**: Use MSYS2 to provide the Unix build environment and patch the GMP configure script to fix the function signature.

**Important**: MSYS2 is required because `gmp-mpfr-sys` needs Unix tools like `sh`, `make`, `autoconf`, etc. Standalone MinGW GCC is not sufficient.

## Working Solution

**Note**: After initial setup, regular PowerShell terminals work fine for `cargo build` and `cargo run` thanks to the `.cargo/config.toml` configuration. MSYS2 is only needed for the initial build or if you need to apply the GMP configure patch.

### 1. Install MSYS2 and Development Tools

```bash
# In MSYS2 terminal
pacman -S mingw-w64-x86_64-gcc mingw-w64-x86_64-gmp mingw-w64-x86_64-mpfr mingw-w64-x86_64-mpc
pacman -S make autoconf automake libtool pkg-config
```

### 2. Configure VSCode for MSYS2

Create `.vscode/settings.json`:
```json
{
    "terminal.integrated.defaultProfile.windows": "MSYS2",
    "terminal.integrated.profiles.windows": {
        "MSYS2": {
            "path": "C:\\msys64\\usr\\bin\\bash.exe",
            "args": ["--login", "-i"],
            "env": {
                "CHERE_INVOKING": "1",
                "MSYSTEM": "MINGW64"
            },
            "icon": "terminal-bash"
        }
    },
    "terminal.integrated.env.windows": {
        "MSYSTEM": "MINGW64",
        "PATH": "C:\\msys64\\mingw64\\bin;C:\\msys64\\usr\\bin;${env:PATH}"
    }
}
```

### 3. Configure Cargo

### VSCode Settings (`.vscode/settings.json`)
```json
{
    "terminal.integrated.defaultProfile.windows": "MSYS2",
    "terminal.integrated.profiles.windows": {
        "MSYS2": {
            "path": "C:\\msys64\\usr\\bin\\bash.exe",
            "args": ["--login", "-i"],
            "env": {
                "CHERE_INVOKING": "1",
                "MSYSTEM": "MINGW64"
            },
            "icon": "terminal-bash"
        }
    },
    "terminal.integrated.env.windows": {
        "MSYSTEM": "MINGW64",
        "PATH": "C:\\msys64\\mingw64\\bin;C:\\msys64\\usr\\bin;${env:PATH}"
    }
}
```

Create `.cargo/config.toml`:
```toml
[env]
MSYSTEM = "MINGW64"
CC = "gcc"
CXX = "g++"
CFLAGS = "-O2"
CXXFLAGS = "-O2"
PATH = "/mingw64/bin:/usr/bin:/bin"
PKG_CONFIG_PATH = "/mingw64/lib/pkgconfig"
LIBRARY_PATH = "/mingw64/lib"
CPATH = "/mingw64/include"

[target.x86_64-pc-windows-gnu]
linker = "C:/msys64/mingw64/bin/gcc.exe"
```

### 4. Set Up Dependencies

Create minimal `Cargo.toml`:
```toml
[dependencies]
rug = { version = "1.26", features = ["float"], default-features = false }
```

### 5. Apply the GMP Configure Patch

**Important**: When you run `cargo build`, it will fail on the first attempt due to the GMP configure bug. You need to patch the configure script:

1. Run `cargo build` - it will fail
2. Find the GMP configure script in: `target/debug/build/gmp-mpfr-sys-*/out/build/gmp-src/configure`
3. Apply the patch:

```bash
# In MSYS2 terminal
sed -i 's/void g(){}/void g(int i, t1* src, t1 n, t1* got, t1* want, int x){}/g' target/debug/build/gmp-mpfr-sys-*/out/build/gmp-src/configure
```

4. Run `cargo build` again - it should now work!

## Build Process

The build process involves:

1. **Configure phase**: GMP runs extensive compatibility tests (creates/deletes many `conftest.c` files)
2. **Compilation phase**: Builds GMP (10-30 minutes), then MPFR, then MPC
3. **Linking phase**: Creates the final Rust library

**Note**: The build can take 30+ minutes. Seeing `conftest.c` files being created and deleted is normal during the configure phase.

## Verification

Once built, you can use `rug` for high-precision arithmetic:

```rust
use rug::Float;

fn main() {
    let x = Float::with_val(100, 2.5);
    let ln_gamma_x = x.ln_gamma();
    println!("ln_gamma(2.5) = {}", ln_gamma_x);
}
```

## Why This Works

- MSYS2 provides a Unix-like environment that GMP expects
- The GNU toolchain (MinGW-w64) is compatible with GMP's build system
- Patching the configure script fixes the function signature bug
- Building from source ensures optimal performance for your specific hardware
