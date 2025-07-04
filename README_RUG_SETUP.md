# Rug Crate Setup on Windows

This document provides step-by-step instructions for setting up the [rug crate](https://crates.io/crates/rug) (arbitrary-precision arithmetic with high-precision `ln_gamma`) on Windows.

## Overview

The rug crate provides Rust bindings for the GNU Multiple Precision Arithmetic Library (GMP), GNU MPFR Library, and GNU MPC Library. On Windows, this requires:

1. **MSYS2** environment (provides Unix tools and libraries)
2. **GNU toolchain** (`x86_64-pc-windows-gnu`)
3. **System libraries** (GMP, MPFR, MPC from MSYS2)

**Important**: The MSVC toolchain (`x86_64-pc-windows-msvc`) is **not supported** by the `gmp-mpfr-sys` crate that rug depends on.

## Prerequisites

- Windows 10/11
- Rust installed via [rustup](https://rustup.rs/)
- Administrative privileges (for MSYS2 installation)

## Step 1: Install MSYS2

1. Download MSYS2 from https://www.msys2.org/
2. Run the installer and follow the installation wizard
3. When prompted, install to the default location: `C:\msys64`
4. After installation, update the package database:
   ```bash
   # In MSYS2 terminal
   pacman -Syu
   ```
5. Close the terminal when prompted and reopen it
6. Complete the update:
   ```bash
   # In MSYS2 terminal
   pacman -Su
   ```

## Step 2: Install Required MSYS2 Packages

Open an MSYS2 terminal and install the necessary packages:

TODO: I think we don't need `mingw-w64-x86_64-mpc`, but it was installed when I tried this originally, so try installing it with pacman as well if the below does not work.

```bash
# Install compiler tools
pacman -S mingw-w64-x86_64-toolchain mingw-w64-x86_64-gcc mingw-w64-x86_64-make

# Install GMP, MPFR, and MPC libraries
pacman -S mingw-w64-x86_64-gmp mingw-w64-x86_64-mpfr 

# Install pkg-config (optional but recommended)
pacman -S mingw-w64-x86_64-pkg-config
```

## Step 3: Configure Rust Toolchain

Create or update `rust-toolchain.toml` in your project root:

```toml
[toolchain]
channel = "stable-x86_64-pc-windows-gnu"
```

If you don't have a `rust-toolchain.toml` file, create one:

```powershell
# In PowerShell, from your project root
@"
[toolchain]
channel = "stable-x86_64-pc-windows-gnu"
"@ | Out-File -FilePath "rust-toolchain.toml" -Encoding utf8
```

**Note**: This explicitly sets the GNU toolchain (`x86_64-pc-windows-gnu`) as the default for this project. This is critical because the MSVC toolchain does not work with rug/gmp-mpfr-sys.

## Step 4: Configure Cargo Environment

Create the `.cargo` directory and `config.toml` file:

```powershell
# In PowerShell, from your project root
New-Item -ItemType Directory -Force -Path ".cargo"
```

Create or update `.cargo/config.toml`:

```toml
[env]
# Tell gmp-mpfr-sys to use system libraries
GMP_MPFR_SYS_USE_SYSTEM = "1"

# Point to MSYS2 library locations
PKG_CONFIG_PATH = "C:/msys64/mingw64/lib/pkgconfig"
LIBRARY_PATH = "C:/msys64/mingw64/lib"
CPATH = "C:/msys64/mingw64/include"

# Optional: Explicitly set linker for GNU target
[target.x86_64-pc-windows-gnu]
linker = "gcc"
```

## Step 5: Update Cargo.toml

Add the rug dependency to your `Cargo.toml`:

```toml
[dependencies]
rug = { version = "1", default-features = false, features = ["float"] }

# Enable system libraries feature for gmp-mpfr-sys
[dependencies.gmp-mpfr-sys]
version = "1.6"
features = ["use-system-libs"]
```

## Step 6: Add MSYS2 to PATH

Add the MSYS2 bin directory to your system PATH:

1. Open System Properties → Advanced → Environment Variables
2. Edit the system PATH variable
3. Add: `C:\msys64\mingw64\bin`
4. Restart your terminal/IDE

Alternatively, set it temporarily in PowerShell:
```powershell
$env:PATH = "C:\msys64\mingw64\bin;" + $env:PATH
```

## Step 7: Build and Test

Now you can build your project:

```powershell
# Clean any previous builds
cargo clean

# Build the project
cargo build

# Run tests
cargo test
```

## Example Usage

Here's a simple example using rug for high-precision arithmetic:

```rust
use rug::{Float, ops::Pow};

fn main() {
    // Create a high-precision float with 256 bits of precision
    let mut x = Float::with_val(256, 2.0);
    
    // Calculate 2^100 with high precision
    x.pow_assign(100);
    println!("2^100 = {}", x);
    
    // High-precision ln_gamma calculation
    let mut gamma_input = Float::with_val(256, 10.5);
    let ln_gamma_result = gamma_input.ln_gamma();
    println!("ln_gamma(10.5) = {}", ln_gamma_result);
}
```

## Troubleshooting

### Build Fails with "sh: command not found"

**Solution**: Ensure MSYS2's bin directory is in your PATH and you're using the GNU toolchain.

```powershell
# Check if sh is available
where.exe sh

# Should show: C:\msys64\usr\bin\sh.exe
```

### "Cannot find library" errors

**Solution**: Verify the MSYS2 libraries are installed and environment variables are set correctly.

```powershell
# Check if libraries exist
Test-Path "C:\msys64\mingw64\lib\libgmp.a"
Test-Path "C:\msys64\mingw64\lib\libmpfr.a"
Test-Path "C:\msys64\mingw64\lib\libmpc.a"

# All should return True
```

### Wrong toolchain errors

**Solution**: Ensure you're using the GNU toolchain, not MSVC.

```powershell
# Check current default toolchain
rustup show

# Should show: stable-x86_64-pc-windows-gnu (default)
```

If not, set it explicitly:
```powershell
rustup default stable-x86_64-pc-windows-gnu
```

### VSCode Integration

If using VSCode, ensure your settings don't override the toolchain. Check `.vscode/settings.json`:

```json
{
    "rust-analyzer.server.extraEnv": {
        "PATH": "C:\\msys64\\mingw64\\bin;C:\\msys64\\usr\\bin"
    }
}
```

## Performance Notes

This setup provides:
- **Fast builds**: Uses precompiled system libraries instead of building from source
- **No patching required**: Works with stable Rust and current GMP/MPFR versions
- **Production ready**: Stable, well-tested configuration

## Alternative Approaches (Not Recommended)

- **MSVC toolchain**: Not supported by `gmp-mpfr-sys`
- **Standalone MinGW**: Requires complex manual setup and library compilation
- **Source builds**: Much slower and requires patching for newer GCC versions

## References

- [rug crate documentation](https://docs.rs/rug/)
- [gmp-mpfr-sys documentation](https://docs.rs/gmp-mpfr-sys/)
- [MSYS2 website](https://www.msys2.org/)
- [GNU MP Library](https://gmplib.org/)
- [GNU MPFR Library](https://www.mpfr.org/)
