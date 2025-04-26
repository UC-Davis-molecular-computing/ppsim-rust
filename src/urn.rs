use rand::rngs::SmallRng;
use rand::Rng;
use rand::SeedableRng;
use statrs::distribution::Hypergeometric;

use crate::flame;

// We precompuate log(k!) for k = 0, 1, ..., MAX_FACTORIAL-1
// technically MAX_FACTORIAL is the SIZE of array, not the max k for which we compute ln(k!),
// starting at ln(0!), so this goes up to ln((MAX_FACTORIAL-1)!).
// We use this cache because numpy does, out of paranoia, but in practice it's actually really not
// any faster than using the Stirling approximation, and the Stirling approximation is surprisingly
// accurate even for small values of k. I'm not sure why numpy uses this cache.
const MAX_FACTORIAL: usize = 126;

pub fn create_log_fact_cache() -> [f64; MAX_FACTORIAL] {
    let mut cache = [0.0; MAX_FACTORIAL];

    let mut i: usize = 1;
    let mut ln_fact: f64 = 0.0;
    while i < MAX_FACTORIAL {
        // using the identity ln(k!) = ln((k-1)!) + ln(k)
        ln_fact += (i as f64).ln();
        cache[i] = ln_fact; // ln(0!)=0 is a special case but we already populated with 0.0
        i += 1;
    }
    cache
}

use lazy_static::lazy_static;
lazy_static! {
    static ref LOGFACT: [f64; MAX_FACTORIAL] = create_log_fact_cache();
}

const HALFLN2PI: f64 = 0.9189385332046728;

// pub static mut num_lookup: usize = 0;
// pub static mut num_stirling: usize = 0;

fn log_factorial(k: u64) -> f64 {
    // for (x, lg) in LOGFACT.iter().enumerate() {
    //     println!("log_fact({}) = {}", x, lg);
    //     panic!();
    // }
    if k < MAX_FACTORIAL as u64 {
        // unsafe { num_lookup += 1 };
        // flame::start("log_factorial lookup");
        let ret = LOGFACT[k as usize];
        // flame::end("log_factorial lookup");
        return ret;
    }
    // unsafe { num_stirling += 1 };
    // flame::start("log_factorial stirling");
    // Use the Stirling approximation for large x
    let k = k as f64;
    let ret =
        (k + 0.5) * k.ln() - k + (HALFLN2PI + (1.0 / k) * (1.0 / 12.0 - 1.0 / (360.0 * k * k)));
    // flame::end("log_factorial stirling");
    ret
}

type State = usize;

/// Samples a discrete uniform random number in the range [low, high].
/// Note inclusive on both ends
pub fn sample_discrete_uniform(rng: &mut SmallRng, low: usize, high: usize) -> usize {
    // <DiscreteUniform as rand::distributions::Distribution<i64>>::sample(&discrete_uniform, rng)
    //     as usize
    rng.gen_range(low as i64..=high as i64) as usize
}

/// Data structure for a multiset that supports fast random sampling.
pub struct Urn {
    pub config: Vec<State>,
    pub order: Vec<usize>,
    pub size: usize,
    rng: SmallRng,
}

impl Urn {
    /// Create a new Urn object.
    pub fn new(config: Vec<State>, seed: Option<u64>) -> Self {
        let size = config.iter().sum();
        let order = (0..config.len()).collect();
        let rng = if let Some(s) = seed {
            SmallRng::seed_from_u64(s)
        } else {
            SmallRng::from_entropy()
        };

        let mut urn = Urn {
            config,
            order,
            size,
            rng,
        };

        urn.sort();

        urn
    }

    /// Updates self.order.
    ///
    /// Uses insertion sort to maintain that
    ///   config[order[0]] >= config[order[1]] >= ... >= config[order[q]].
    /// This method is used to have O(q) time when order is almost correct.
    pub fn sort(&mut self) {
        for i in 1..self.config.len() {
            // See if the entry at order[i] needs to be moved earlier.
            // Recursively, we have ensured that order[0], ..., order[i-1] have the correct order.
            let o_i = self.order[i];
            // j will be the index where order[i] should be inserted to.
            let mut j = i;
            while j > 0 && self.config[o_i] > self.config[self.order[j - 1]] {
                j -= 1;
            }

            // Index at order[i] will get moved to order[j], and all indices order[j], ..., order[i-1] get right shifted
            // First do the right shift, moving order[i-k] for k = 1, ..., i-j
            for k in 1..(i - j + 1) {
                self.order[i + 1 - k] = self.order[i - k];
            }
            self.order[j] = o_i;
        }
    }

    /// Samples and removes one element, returning its index.
    pub fn sample_one(&mut self) -> Result<State, String> {
        if self.size <= 0 {
            return Err("Cannot sample from empty urn".to_string());
        }

        // Generate random integer in [0, self.size-1]
        let x = sample_discrete_uniform(&mut self.rng, 0, self.size - 1);

        let mut i = 0;
        let mut x: i64 = x as i64;
        let mut index = 0;

        while x >= 0 {
            index = self.order[i];
            x -= self.config[index] as i64;
            i += 1;
        }

        // Decrement the count for the sampled element
        self.config[index] -= 1;
        self.size -= 1;

        Ok(index)
    }

    /// Adds one element at index.
    pub fn add_to_entry(&mut self, index: usize, amount: i64) {
        self.config[index] = (self.config[index] as i64 + amount) as State;
        self.size = (self.size as i64 + amount) as usize;
    }

    /// Samples n elements, writing them into the vector v.
    ///
    /// This method is implemented only to make testing easier, but sample_vector_impl
    /// should be called within Rust code, since it does not allocate new memory as this does.
    ///
    /// Args:
    ///     n: number of elements to sample
    ///     v: the array to write the output vector in
    ///         (this is faster than re-initializing an output array)
    ///         
    /// Returns:
    ///     nz: the number of nonzero entries
    ///         v[self.order[i]] for i in range(nz) can then loop over only
    ///             the nonzero entries of the vector
    pub fn sample_vector(&mut self, n: usize, v: &mut [State]) -> Result<usize, String> {
        let mut n = n as i64;
        let mut i: usize = 0;
        let mut total: usize = self.size;
        for j in 0..v.len() {
            v[j] = 0;
        }

        while n > 0 && i < self.config.len() - 1 {
            let index = self.order[i];
            let successes = self.config[index];
            let h = hypergeometric_sample(total, successes, n as usize, &mut self.rng)?;
            total -= self.config[index];

            v[index] = h as usize;
            n -= h as i64;
            self.size -= h as usize;
            self.config[index] -= h as usize;
            i += 1;
        }

        if n != 0 {
            debug_assert!(n > 0);
            v[self.order[i]] = n as usize;
            self.config[self.order[i]] -= n as usize;
            self.size -= n as usize;
            i += 1;
        }

        Ok(i)
    }

    /// Adds a vector of elements to the urn.
    pub fn add_vector(&mut self, vector: &Vec<State>) {
        for i in 0..self.config.len() {
            let count = vector[i];
            self.config[i] += count;
            self.size += count;
        }
    }

    /// Set the counts back to zero.
    pub fn reset(&mut self) {
        for i in 0..self.config.len() {
            self.config[i] = 0;
        }
        self.size = 0;
    }

    /// This mimics creating a new Urn, but instead we replace
    /// the current config with the new one. This is useful because
    /// we can save the RNG. Otherwise we would have to create a
    /// new Urn with the same RNG and this avoid borrowship issues.
    pub fn reset_config(&mut self, config: &Vec<State>) {
        for i in 0..self.config.len() {
            self.config[i] = config[i];
        }
        self.size = config.iter().sum();
        self.order = (0..config.len()).collect();
        self.sort();
    }
}

fn hypergeometric_sample(
    popsize: usize,
    good: usize,
    sample: usize,
    rng: &mut SmallRng,
) -> Result<usize, String> {
    // println!("popsize = {popsize}, good = {good}, sample = {sample}");
    let h: usize;
    if sample >= 10 && sample <= good + popsize - 10 {
        flame::start("hypergeometric_hrua");
        h = hypergeometric_hrua(popsize, good, sample, rng)?;
        flame::end("hypergeometric_hrua");
    } else {
        flame::start("hypergeometric_slow");
        // This is the simpler implementation for small samples.
        let hypergeometric_result = Hypergeometric::new(popsize as u64, good as u64, sample as u64);
        if hypergeometric_result.is_err() {
            return Err(String::from(format!(
                "Hypergeometric distribution creation error: {:?}",
                hypergeometric_result.unwrap_err(),
            )));
        }
        let hypergeometric = hypergeometric_result.unwrap();
        let h64: u64 = rng.sample(hypergeometric);
        flame::end("hypergeometric_slow");
        h = h64 as usize;
    }
    Ok(h)
}

// adapted from numpy's implementation of the hypergeometric distribution (as of April 2025)
// https://github.com/numpy/numpy/blob/b76bb2329032809229e8a531ba3179c34b0a3f0a/numpy/random/src/distributions/random_hypergeometric.c#L119
const D1: f64 = 1.7155277699214135; // 2*sqrt(2/e)
const D2: f64 = 0.8989161620588988; // 3 - 2*sqrt(3/e)
fn hypergeometric_hrua(
    popsize: usize,
    good: usize,
    sample: usize,
    rng: &mut SmallRng,
) -> Result<usize, String> {
    if good > popsize {
        return Err("good must be less than or equal to popsize".to_string());
    }
    if sample > popsize {
        return Err("sample must be less than or equal to popsize".to_string());
    }
    let bad = popsize - good;
    let computed_sample = sample.min(popsize - sample);
    let mingoodbad = good.min(bad);
    let maxgoodbad = good.max(bad);

    /*
     *  Variables that do not match Stadlober (1989)
     *    Here               Stadlober
     *    ----------------   ---------
     *    mingoodbad            M
     *    popsize               N
     *    computed_sample       n
     */
    let p = mingoodbad as f64 / popsize as f64;
    let q = maxgoodbad as f64 / popsize as f64;

    let mu = computed_sample as f64 * p; // mean of the distribution

    let a = mu + 0.5;

    let var = ((popsize - computed_sample) as f64 * computed_sample as f64 * p * q
        / (popsize as f64 - 1.0)) as f64; // variance of the distribution

    let c = var.sqrt() + 0.5;

    /*
     *  h is 2*s_hat (See Stadlober's thesis (1989), Eq. (5.17); or
     *  Stadlober (1990), Eq. 8).  s_hat is the scale of the "table mountain"
     *  function that dominates the scaled hypergeometric PMF ("scaled" means
     *  normalized to have a maximum value of 1).
     */
    let h = D1 * c + D2;

    let m =
        ((computed_sample + 1) as f64 * (mingoodbad + 1) as f64 / (popsize + 2) as f64) as usize;

    let g = log_factorial(m as u64)
        + log_factorial((mingoodbad - m) as u64)
        + log_factorial((computed_sample - m) as u64)
        + log_factorial((maxgoodbad + m - computed_sample) as u64);

    /*
     *  b is the upper bound for random samples:
     *  ... min(computed_sample, mingoodbad) + 1 is the length of the support.
     *  ... floor(a + 16*c) is 16 standard deviations beyond the mean.
     *
     *  The idea behind the second upper bound is that values that far out in
     *  the tail have negligible probabilities.
     *
     *  There is a comment in a previous version of this algorithm that says
     *      "16 for 16-decimal-digit precision in D1 and D2",
     *  but there is no documented justification for this value.  A lower value
     *  might work just as well, but I've kept the value 16 here.
     */
    let b = (computed_sample.min(mingoodbad) + 1).min((a + 16.0 * c).floor() as usize);

    let mut k: usize;
    loop {
        let u = rng.gen::<f64>();
        let v = rng.gen::<f64>(); // "U star" in Stadlober (1989)
        let x = a + h * (v - 0.5) / u;

        // fast rejection:
        if x < 0.0 || x >= b as f64 {
            continue;
        }

        k = x.floor() as usize;

        let gp = log_factorial(k as u64)
            + log_factorial((mingoodbad - k) as u64)
            + log_factorial((computed_sample - k) as u64)
            + log_factorial((maxgoodbad + k - computed_sample) as u64);

        let t = g - gp;

        // fast acceptance:
        if (u * (4.0 - u) - 3.0) <= t {
            break;
        }

        // fast rejection:
        if u * (u - t) >= 1.0 {
            continue;
        }

        if 2.0 * u.ln() <= t {
            // acceptance
            break;
        }
    }

    if good > bad {
        k = computed_sample - k;
    }

    if computed_sample < sample {
        k = good - k;
    }

    Ok(k)
}
