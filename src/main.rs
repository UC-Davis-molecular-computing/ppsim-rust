use ppsim_rust::simulator::SimulatorMultiBatch;

fn main() {
    let delta = vec![
        vec![(0, 0), (1, 1), (2, 2)],
        vec![(0, 0), (1, 1), (2, 2)],
        vec![(0, 0), (1, 1), (2, 2)],
    ];
    let n = 100usize;
    let a_init = (0.51 * n as f64) as usize;
    let b_init = n - a_init;
    let init_config = vec![a_init as usize, b_init as usize, 0];
    let transition_order = "symmetric".to_string();
    let gillespie = false;
    let seed = Some(1);
    let sim = SimulatorMultiBatch::from_delta_deterministic(
        delta,
        init_config,
        transition_order,
        gillespie,
        seed,
    );
    println!("n={n}");
    let r = 2;
    let u = 0.713;
    let has_bounds = true;
    let pp = true;
    let l = sim.sample_coll(r, u, has_bounds, pp);
    println!("l={l}");
}
