#![feature(f128)]

use num_bigint::BigInt;

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
    let x: f128 = 8.5829034758923475897234582348975908234758927348905723890589;

    println!("x = {}", f128_to_decimal(x));
}
