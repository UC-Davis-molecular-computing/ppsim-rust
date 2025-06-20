use rand::rngs::SmallRng;
use rand::Rng;

use rand_distr::Distribution;

#[allow(unused_imports)]
use crate::flame;

// #[cfg(feature = "ue")]
pub fn ln_gamma(x: f64) -> f64 {
    // println!("ue version called");
    ln_gamma_special(x)
}

// #[cfg(not(feature = "ue"))]
// pub fn ln_gamma(x: f64) -> f64 {
//     // println!("non ue version called");
//     ln_gamma_manual(x)
// }

#[cfg(feature = "ue")]
pub fn ln_factorial(x: usize) -> f64 {
    // println!("ue version called");
    ln_factorial_statrs(x as u64)
}

#[cfg(not(feature = "ue"))]
pub fn ln_factorial(x: usize) -> f64 {
    // println!("non ue version called");
    ln_factorial_manual(x as u64)
}

#[cfg(feature = "ue")]
pub fn hypergeometric_sample(
    popsize: usize,
    good: usize,
    draws: usize,
    rng: &mut SmallRng,
) -> Result<usize, String> {
    // println!("hypergeometric_sample_statrs");
    hypergeometric_sample_statrs(popsize, good, draws, rng)
}

#[cfg(not(feature = "ue"))]
pub fn hypergeometric_sample(
    popsize: usize,
    good: usize,
    draws: usize,
    rng: &mut SmallRng,
) -> Result<usize, String> {
    // println!("hypergeometric_sample_manual");
    hypergeometric_sample_manual(popsize, good, draws, rng)
}

#[cfg(feature = "ue")]
pub fn multinomial_sample(n: usize, pix: &Vec<f64>, result: &mut [usize], rng: &mut SmallRng) {
    multinomial_sample_statrs(n, pix, result, rng);
}

#[cfg(not(feature = "ue"))]
pub fn multinomial_sample(n: usize, pix: &Vec<f64>, result: &mut [usize], rng: &mut SmallRng) {
    multinomial_sample_manual(n, pix, result, rng);
}

/////////////////////////////////////////////////////////////////////////////////
// ln_gamma

// As always, if the statrs crate implements something, it's slower than alternatives.
// Here we use the special crate instead, which is about 40% faster.
// const ln_gamma: fn(f64) -> f64 = statrs::function::gamma::ln_gamma;
// UPDATE: The custom implementation below, adapted from R, is slight faster, maybe 10-20%,
// than the implementation in the special crate.

pub fn ln_gamma_special(x: f64) -> f64 {
    // special implements these as methods that can be called on f64's if we use special::Gamma,
    // but that gives a Rust warning about possibly the method name being used in Rust in the future.
    // We can call the method directly like this, but for some reason it returns a pair,
    // and the output of ln_gamma is the first element.
    special::Gamma::ln_gamma(x).0
}

// adapted from C source for R standard library
// https://github.com/SurajGupta/r-source/blob/a28e609e72ed7c47f6ddfbb86c85279a0750f0b7/src/nmath/lgamma.c#L44
// Since we only call it on non-negative integers, we can simplify some of the code that was handling
// negative integers and NaN values.
const M_LN_SQRT_2PI: f64 = 0.918938533204672741780329736406; // log(sqrt(2*pi)) == log(2*pi)/2
const XMAX_LN_GAMMA: f64 = 2.5327372760800758e+305;

pub fn ln_gamma_manual(x: f64) -> f64 {
    let ret: f64;
    if x <= 10.0 {
        return special::Gamma::gamma(x as f64).ln();
    }
    if x > XMAX_LN_GAMMA {
        return f64::INFINITY;
    }
    let lnx = x.ln();
    // In C code this only happens if x > 0, but since our x is usize,
    // we do this unconditionally.
    if x > 1e17 {
        ret = x * (lnx - 1.0);
    } else if x > 4934720.0 {
        ret = M_LN_SQRT_2PI + (x - 0.5) * lnx - x;
    } else {
        ret = M_LN_SQRT_2PI + (x - 0.5) * lnx - x + lgammacor(x);
    }
    ret
}

const ALGMCS: [f64; 15] = [
    0.1666389480451863247205729650822,
    -0.1384948176067563840732986059135e-4,
    0.9810825646924729426157171547487e-8,
    -0.1809129475572494194263306266719e-10,
    0.6221098041892605227126015543416e-13,
    -0.3399615005417721944303330599666e-15,
    0.2683181998482698748957538846666e-17,
    -0.2868042435334643284144622399999e-19,
    0.3962837061046434803679306666666e-21,
    -0.6831888753985766870111999999999e-23,
    0.1429227355942498147573333333333e-24,
    -0.3547598158101070547199999999999e-26,
    0.1025680058010470912000000000000e-27,
    -0.3401102254316748799999999999999e-29,
    0.1276642195630062933333333333333e-30,
];
const NALGM: usize = 5;
const XBIG_LGAMMACOR: f64 = 94906265.62425156;
const XMAX_LGAMMACOR: f64 = 3.745194030963158e306;

fn lgammacor(x: f64) -> f64 {
    assert!(x >= 10.0);
    assert!(x < XMAX_LGAMMACOR);
    if x < 10.0 {
        return 0.0;
    } else if x >= XBIG_LGAMMACOR {
        panic!("x = {x} must be less than {XBIG_LGAMMACOR}");
    } else if x < XBIG_LGAMMACOR {
        let tmp = 10.0 / x;
        return chebyshev_eval(tmp * tmp * 2.0 - 1.0) / x;
    }
    1.0 / (x * 12.0)
}

fn chebyshev_eval(x: f64) -> f64 {
    let n = NALGM;
    let a = ALGMCS;

    if n < 1 || n > 1000 {
        panic!("n must be between 1 and 1000, got {n}");
    }
    if x < -1.1 || x > 1.1 {
        panic!("x must be between -1.1 and 1.1, got {x}");
    }

    let twox = x * 2.0;
    let mut b0 = 0.0;
    let mut b1 = 0.0;
    let mut b2 = 0.0;

    for i in (1..=n).rev() {
        b2 = b1;
        b1 = b0;
        b0 = twox * b1 - b2 + a[n - i];
    }
    (b0 - b2) * 0.5
}

/////////////////////////////////////////////////////////////////////////////////
// log_factorial

pub fn ln_factorial_statrs(k: u64) -> f64 {
    statrs::function::factorial::ln_factorial(k)
}

// We precompuate log(k!) for k = 0, 1, ..., MAX_FACTORIAL-1
// technically MAX_FACTORIAL is the SIZE of array, not the max k for which we compute ln(k!),
// starting at ln(0!), so this goes up to ln((MAX_FACTORIAL-1)!).
// We use this cache because numpy does, out of paranoia, but in practice it's actually really not
// any faster than using the Stirling approximation, and the Stirling approximation is surprisingly
// accurate even for small values of k. I'm not sure why numpy uses this cache, maybe
// because Numerical Recipes in C recommends that: https://numerical.recipes/book.html
// since it defers to log_gamma for large, which might be slower than the Stirling approximation
// used for integer k in log_factorial.
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

const HALFLN2PI: f64 = 0.9189385332046728;

pub fn ln_factorial_manual(k: u64) -> f64 {
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

/////////////////////////////////////////////////////////////////////////////////
// hypergeometric_sample

use statrs::distribution::Hypergeometric;

pub fn hypergeometric_sample_statrs(
    popsize: usize,
    good: usize,
    draws: usize,
    rng: &mut SmallRng,
) -> Result<usize, String> {
    let hypergeometric_result = Hypergeometric::new(popsize as u64, good as u64, draws as u64);
    if hypergeometric_result.is_err() {
        return Err(String::from(format!(
            "Hypergeometric distribution creation error: {:?}",
            hypergeometric_result.unwrap_err(),
        )));
    }
    let hypergeometric = hypergeometric_result.unwrap();
    let h64: u64 = rng.sample(hypergeometric);
    let h = h64 as usize;
    Ok(h)
}

// adapted from numpy's implementation of the hypergeometric distribution (as of April 2025)
// https://github.com/numpy/numpy/blob/b76bb2329032809229e8a531ba3179c34b0a3f0a/numpy/random/src/distributions/random_hypergeometric.c#L246
pub fn hypergeometric_sample_manual(
    popsize: usize,
    good: usize,
    draws: usize,
    rng: &mut SmallRng,
) -> Result<usize, String> {
    let h: usize;
    if draws >= 10 && draws <= good + popsize - 10 {
        h = hypergeometric_hrua(popsize, good, draws, rng)?;
    } else {
        // This is the simpler implementation for small samples.
        let hypergeometric_result = Hypergeometric::new(popsize as u64, good as u64, draws as u64);
        if hypergeometric_result.is_err() {
            return Err(String::from(format!(
                "Hypergeometric distribution creation error: {:?}",
                hypergeometric_result.unwrap_err(),
            )));
        }
        let hypergeometric = hypergeometric_result.unwrap();
        let h64: u64 = rng.sample(hypergeometric);
        h = h64 as usize;
    }
    Ok(h)
}

// adapted from numpy's implementation of the hypergeometric_hrua algorithm
// https://github.com/numpy/numpy/blob/b76bb2329032809229e8a531ba3179c34b0a3f0a/numpy/random/src/distributions/random_hypergeometric.c#L119
const D1: f64 = 1.7155277699214135; // 2*sqrt(2/e)
const D2: f64 = 0.8989161620588988; // 3 - 2*sqrt(3/e)
pub fn hypergeometric_hrua(
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

    let g = ln_factorial(m)
        + ln_factorial(mingoodbad - m)
        + ln_factorial(computed_sample - m)
        + ln_factorial(maxgoodbad + m - computed_sample);

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

        let gp = ln_factorial(k)
            + ln_factorial(mingoodbad - k)
            + ln_factorial(computed_sample - k)
            + ln_factorial(maxgoodbad + k - computed_sample);

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

/////////////////////////////////////////////////////////////////////////////////
// multinomial_sample

use nalgebra::DVector;
use statrs::distribution::Multinomial;

pub fn multinomial_sample_statrs(
    n: usize,
    pix: &Vec<f64>,
    result: &mut [usize],
    rng: &mut SmallRng,
) {
    assert_eq!(pix.len(), result.len());
    let multinomial = Multinomial::new(pix.clone(), n as u64).unwrap();
    let sample: DVector<u64> = rng.sample(multinomial);

    assert_eq!(sample.len(), result.len());
    for i in 0..result.len() {
        result[i] = sample[i] as usize;
    }
}

pub fn binomial_sample(n: usize, p: f64, mut rng: &mut SmallRng) -> usize {
    let binomial_distribution = rand_distr::Binomial::new(n as u64, p).unwrap();
    let sample = binomial_distribution.sample(&mut rng);
    sample as usize
}

// port of numpy's multinomial sample to Rust, using rand_distr::Binomial as the underlying binomial sampler
// https://github.com/numpy/numpy/blob/4961a1414bba2222016f29a03dcf75e6034a13f7/numpy/random/src/distributions/distributions.c#L1726
pub fn multinomial_sample_manual(
    n: usize,
    pix: &Vec<f64>,
    result: &mut [usize],
    rng: &mut SmallRng,
) {
    assert_eq!(pix.len(), result.len());
    let mut remaining_p = 1.0;
    let d = pix.len(); // in numpy C code, pix is just a pointer so they need pix's array length too
    let mut dn = n;
    // Original Cython implementation zeroed out the result array initially, but
    // since we are overwriting the array, we only zero out the entries if we break out of the loop early.
    for j in 0..(d - 1) {
        result[j] = binomial_sample(dn as usize, pix[j] / remaining_p, rng);
        dn -= result[j];
        if dn <= 0 {
            // erase old values in remainder of result array
            for i in (j + 1)..d {
                result[i] = 0;
            }
            break;
        }
        remaining_p -= pix[j];
    }
    if dn > 0 {
        result[d - 1] = dn as usize;
    }
}
