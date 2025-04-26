use ppsim_rust::urn::Urn;

fn main() {
    let seed: u64 = 0;
    let config = vec![1, 9];
    let mut urn = Urn::new(config, Some(seed));
    let mut v = vec![0, 0];
    let n = 2;
    let nz = urn.sample_vector(n, &mut v).unwrap();
    println!("Sampled vector: {:?}", v);
    println!("Number of nonzero entries: {:?}", nz);
}
