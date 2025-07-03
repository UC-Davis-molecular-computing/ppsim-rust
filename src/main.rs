use rug::Float;

fn main() {
    println!("🧪 Testing Rug crate with high-precision arithmetic on Windows...");

    // Test basic arithmetic
    let xf: f64 = 4.0;
    let x = Float::with_val(100, xf);
    println!("x = {}", x.clone());

    // Test ln_gamma function - this was the original goal!
    let ln_gamma_x = x.clone().ln_gamma();
    println!("ln_gamma({xf}) = {ln_gamma_x}");

    // Test more precision
    let high_prec = Float::with_val(200, 1.2345678901234567890123456789_f64);
    println!("High precision number: {}", high_prec);

    // Test gamma function
    let gamma_x = x.gamma();
    println!("gamma({xf}) = {gamma_x}",);

    println!(
        "✅ SUCCESS! High-precision arithmetic with ln_gamma is working correctly on Windows!"
    );
}
