use rebop::define_system;

define_system! {
    r_fox_eats r_rabbit_multiplies r_fox_dies;
    LV { R, F }
    fox_eats_rabbit   : R + F => 2 F  @ r_fox_eats
    rabbit_multiplies : R     => 2 R  @ r_rabbit_multiplies
    fox_dies          : F     =>      @ r_fox_dies
}

fn main() {
    println!("Running Lotka-Volterra model with rebop");
    let mut problem = LV::new();
    let pop_exponent = 3;
    let n = 10isize.pow(pop_exponent);
    problem.r_fox_eats = 1.0 / n as f64;
    problem.r_rabbit_multiplies = 1.0;
    problem.r_fox_dies = 1.0;
    problem.R = n / 2;
    problem.F = n / 2;
    println!("time,R,F");
    for t in 0..250 {
        problem.advance_until(t as f64);
        println!("{},{},{}", problem.t, problem.R, problem.F);
    }
}
