#![feature(f128)]

use num_bigint::BigInt;

const MAX_FACTORIAL: usize = 126;

pub fn create_log_fact_cache() -> [f64; MAX_FACTORIAL] {
    let mut cache = [0.0; MAX_FACTORIAL];

    let mut i: usize = 1;
    let mut ln_fact: f64 = 0.0;
    while i < MAX_FACTORIAL {
        // using the identity ln(k!) = ln((k-1)!) + ln(k)
        ln_fact += (i as f64).ln();
        cache[i] = ln_fact; // ln(0!) = 0 is a special case but we already populated with 0.0
        i += 1;
    }
    cache
}

use lazy_static::lazy_static;
lazy_static! {
    static ref LOGFACT: [f64; MAX_FACTORIAL] = create_log_fact_cache();
}

fn binomial_as_f64(n: u64, k: u64) -> f64 {
    if k > n {
        0.0
    } else {
        (0.5 + (ln_factorial(n) - ln_factorial(k) - ln_factorial(n - k)).exp()).floor()
    }
}
const HALFLN2PI: f64 = 0.9189385332046728;

pub fn ln_factorial(k: u64) -> f64 {
    if k < MAX_FACTORIAL as u64 {
        let ret = LOGFACT[k as usize];
        return ret;
    }
    // Use the Stirling approximation for large x
    let k = k as f64;
    let ret =
        (k + 0.5) * k.ln() - k + (HALFLN2PI + (1.0 / k) * (1.0 / 12.0 - 1.0 / (360.0 * k * k)));
    ret
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
    println!(
        "binomial_to_f64 is {:?}",
        binomial_as_f64(19053614116978, 2)
    )
}
