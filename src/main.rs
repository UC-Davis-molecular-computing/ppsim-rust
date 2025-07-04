use rug::Float;

#[macro_use]
extern crate timeit;

fn main() {
    println!("🧪 Testing Rug crate with high-precision arithmetic on Windows...");

    let mut x = Float::with_val(116, 8.0);
    for _ in 0..100 {
        println!("{:?}", x);
        x.ln_gamma_mut();
    }
    let mut x = Float::with_val(116, 8.0);
    let secs = timeit_loops!(100, {
        x.ln_gamma_mut();
    });
    let us = secs * 1_000_000.0;
    println!("Time taken for 100 iterations: {:.6} us", us);
    // println!("ln_gamma({xf}) = {ln_gamma_x}");
}
