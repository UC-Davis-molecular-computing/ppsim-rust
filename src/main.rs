#![feature(f128)]

// Define the flame module for the binary
#[cfg(feature = "flm")]
pub use flame;

#[cfg(not(feature = "flm"))]
pub mod flame {
    pub fn start(_name: &str) {}
    pub fn end(_name: &str) {}
}

mod util;

use crate::util::ln_gamma_manual_high_precision;
use num_bigint::BigInt;
use std::fmt;

// Wrapper type for f128 that implements Display
#[derive(Debug, Clone, Copy)]
pub struct F128Display(pub f128);

impl fmt::Display for F128Display {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", f128_to_decimal(self.0))
    }
}

impl From<f128> for F128Display {
    fn from(x: f128) -> Self {
        F128Display(x)
    }
}

// Convenience function to wrap f128 values
fn display_f128(x: f128) -> F128Display {
    F128Display(x)
}

// Macro for easy f128 printing
macro_rules! println_f128 {
    ($($arg:tt)*) => {
        println!($($arg)*);
    };
}

// Much simpler: macro specifically for f128 printing that looks like println!
macro_rules! println_f128 {
    // No arguments case
    () => {
        std::println!();
    };
    // Just format string case  
    ($fmt:literal) => {
        std::println!($fmt);
    };
    // The main case: format string with f128 arguments
    ($fmt:literal, $($arg:expr),* $(,)?) => {
        std::println!($fmt, $(f128_to_decimal($arg)),*);
    };
}

fn f128_to_decimal(x: f128) -> String {
    // Handle special cases first
    if x.is_nan() {
        return "NaN".to_string();
    }
    if x.is_infinite() {
        return if x.is_sign_positive() {
            "inf".to_string()
        } else {
            "-inf".to_string()
        };
    }
    if x == 0.0 {
        return "0.0".to_string();
    }

    // Extract IEEE 754 binary128 components
    let bits = x.to_bits();
    let sign = (bits >> 127) != 0;
    let exponent = ((bits >> 112) & 0x7FFF) as i32;
    let mantissa = bits & 0x0000_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF;

    // Handle sign
    let sign_str = if sign { "-" } else { "" };

    // IEEE 754 binary128 has:
    // - 1 sign bit
    // - 15 exponent bits (bias = 16383)
    // - 112 mantissa bits

    let bias = 16383;
    let actual_exponent = exponent - bias;

    // Build the significand (1.mantissa for normal numbers)
    let mut significand = BigInt::from(1_u128 << 112); // Implicit leading 1
    significand += BigInt::from(mantissa);

    // Calculate the actual value: significand * 2^(actual_exponent - 112)
    let power_of_2 = actual_exponent - 112;

    let mut result = significand;

    if power_of_2 >= 0 {
        // Multiply by 2^power_of_2
        result <<= power_of_2;
        format!("{}{}.0", sign_str, result)
    } else {
        // Divide by 2^(-power_of_2)
        // This is where we need to do decimal division
        let divisor = BigInt::from(1_u128) << (-power_of_2);

        // Perform long division to get decimal representation
        let quotient = &result / &divisor;
        let remainder = &result % &divisor;

        if remainder == BigInt::from(0) {
            format!("{}{}.0", sign_str, quotient)
        } else {
            // Calculate decimal places
            let mut decimal_digits = String::new();
            let mut current_remainder = remainder * 10;

            for _ in 0..50 {
                // Limit to 50 decimal places
                let digit: BigInt = &current_remainder / &divisor;
                decimal_digits.push_str(&digit.to_string());
                current_remainder = (&current_remainder % &divisor) * 10;

                if current_remainder == BigInt::from(0) {
                    break;
                }
            }

            format!("{}{}.{}", sign_str, quotient, decimal_digits)
        }
    }
}

fn main() {
    let x: f128 = 8.0;
    let ln = ln_gamma_manual_high_precision(x);

    // Method 1: Direct wrapper construction
    println!("ln_gamma({}) = {}", F128Display(x), F128Display(ln));

    // Method 2: Using convenience function
    println!("ln_gamma({}) = {}", display_f128(x), display_f128(ln));

    // Method 3: Using our custom macro - looks just like println!
    println_f128!("ln_gamma({}) = {}", x, ln);
    
    // This is as close as we can get to `println!("{}", x)` 
    // while maintaining type safety and not requiring unstable features
    println_f128!("x = {}", x);
    println_f128!("ln = {}", ln);
}
