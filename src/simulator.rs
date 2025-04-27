use std::collections::HashMap;
use std::io::Write;
use std::time::{Duration, Instant};

use crate::flame;

use numpy::PyArrayMethods;
use numpy::PyUntypedArrayMethods;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use nalgebra::DVector;
use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3};
use rand::rngs::SmallRng;
use rand::Rng;
use rand::SeedableRng;
use statrs::distribution::{Geometric, Multinomial, Uniform};
use statrs::function::gamma::ln_gamma;

use crate::urn::Urn;

type State = usize;

//TODO: consider using ndarrays instead of multi-dimensional vectors
// I think native Rust arrays won't work because their size needs to be known at compile time.
#[pyclass]
pub struct SimulatorMultiBatch {
    /// The population size (sum of values in urn.config).
    #[pyo3(get, set)]
    pub n: usize,
    /// The current number of elapsed interaction steps.
    #[pyo3(get, set)]
    pub t: usize,
    /// The total number of states (length of urn.config).
    pub q: usize,
    /// A q x q array of pairs (c,d) representing the transition function.
    /// delta[a][b] gives contains the two output states for a
    /// deterministic transition a,b --> c,d.
    #[pyo3(get, set)]
    pub delta: Vec<Vec<(State, State)>>,
    /// A q x q boolean array where null_transitions[i][j] says if these states have a null interaction.
    #[pyo3(get, set)]
    pub null_transitions: Vec<Vec<bool>>,
    /// A boolean that is true if there are any random transitions.
    pub is_random: bool,
    /// A q x q array of pairs random_transitions[i][j] = (`num_outputs`, `first_idx`).
    /// `num_outputs` is the number of possible outputs if transition i,j --> ... is random,
    /// otherwise it is 0. `first_idx` gives the starting index to find
    /// the outputs in the array `self.random_outputs` if it is random.
    #[pyo3(get, set)] // XXX: for testing
    pub random_transitions: Vec<Vec<(State, State)>>,
    /// A 1D array of pairs containing all (out1,out2) outputs of random transitions,
    /// whose indexing information is contained in random_transitions.
    /// For example, if there are random transitions
    /// 3,4 --> 5,6 and 3,4 --> 7,8 and 3,4 --> 3,2, then
    /// `random_transitions[3][4] = (3, first_idx)` for some `first_idx`, and
    /// `random_outputs[first_idx]   = (5,6)`,
    /// `random_outputs[first_idx+1] = (7,8)`, and
    /// `random_outputs[first_idx+2] = (3,2)`.
    #[pyo3(get, set)] // XXX: for testing
    pub random_outputs: Vec<(usize, usize)>,
    /// An array containing all random transition probabilities,
    /// whose indexing matches random_outputs.
    #[pyo3(get, set)] // XXX: for testing
    pub transition_probabilities: Vec<f64>,
    /// The maximum number of random outputs from any random transition.
    pub random_depth: usize,
    /// A pseudorandom number generator.
    rng: SmallRng,
    /// An :any:`Urn` object that stores the configuration (as urn.config) and has methods for sampling.
    /// This is the equivalent of C in the pseudocode for the batching algorithm in the
    /// original Berenbrink et al. paper.
    urn: Urn,
    /// An additional :any:`Urn` where agents are stored that have been
    /// updated during a batch. Called `C'` in the pseudocode for the batching algorithm.
    updated_counts: Urn,
    /// Precomputed log(n).
    logn: f64,
    /// Minimum number of interactions that must be simulated in each
    /// batch. Collisions will be repeatedly sampled up until batch_threshold
    /// interaction steps, then all non-colliding pairs of 'delayed agents' are
    /// processed in parallel.
    batch_threshold: usize,
    /// Array which stores sampled counts of initiator agents
    /// (row sums of the 'D' matrix from the paper).
    row_sums: Vec<usize>,
    /// Array which stores the counts of responder agents for each type of
    /// initiator agent (one row of the 'D' matrix from the paper).
    row: Vec<usize>,
    // Cython implementation maintained this array to avoid reallocation sicne the numpy c_distributions
    // implementation of multinomial writes into a user-provided array.
    // But in the Rust implementation, we use the statrs Multinomial struct, which creates a new Vector
    // whenever sampled from, so there's no point in maintaining this array.
    // m: Vec<usize>,
    /// A boolean determining if we are currently doing Gillespie steps.
    #[pyo3(get, set)]
    pub do_gillespie: bool,
    /// A boolean determining if the configuration is silent (all interactions are null).
    #[pyo3(get, set)]
    pub silent: bool,
    /// A list of reactions, as (input, input, output, output).
    #[pyo3(get, set)]
    pub reactions: Vec<(State, State, State, State)>,
    /// An array holding indices into `self.reactions` of all currently enabled
    /// (i.e., applicable; positive counts of reactants) reactions.
    #[pyo3(get, set)]
    pub enabled_reactions: Vec<usize>,
    /// The number of meaningful indices in `self.enabled_reactions`.
    #[pyo3(get, set)]
    pub num_enabled_reactions: usize,
    /// An array of length `self.reactions.len()` holding the propensities of each reaction.
    propensities: Vec<f64>, // these are used only when doing Gillespie steps and are all 0 otherwise
    /// The probability of each reaction.
    #[pyo3(get, set)]
    pub reaction_probabilities: Vec<f64>,
    /// The probability of a non-null interaction must be below this
    /// threshold to keep doing Gillespie steps.
    gillespie_threshold: f64,
    /// Precomputed values to speed up the function sample_coll(r, u).
    /// This is a 2D array of size (`coll_table_r_values.len()`, `coll_table_u_values.len()`).
    coll_table: Vec<Vec<usize>>,
    /// Values of r, giving one axis of coll_table.
    coll_table_r_values: Vec<usize>,
    /// Values of u, giving the other axis of coll_table.
    coll_table_u_values: Vec<f64>,
    /// Used to populate coll_table_r_values.
    r_constant: usize,

    // Used for testing; not needed for simulation.
    #[pyo3(get, set)]
    collision_counts: HashMap<usize, usize>,
}

#[pymethods]
impl SimulatorMultiBatch {
    #[new]
    #[pyo3(signature = (init_config, delta, null_transitions, random_transitions, random_outputs, transition_probabilities, seed=None))]
    pub fn new(
        init_config: PyReadonlyArray1<State>,
        delta: PyReadonlyArray3<State>,
        null_transitions: PyReadonlyArray2<bool>,
        random_transitions: PyReadonlyArray3<usize>,
        random_outputs: PyReadonlyArray2<State>,
        transition_probabilities: PyReadonlyArray1<f64>,
        seed: Option<u64>,
    ) -> Self {
        let config = init_config.to_vec().unwrap();
        let n = config.iter().sum();
        let q = config.len() as State;

        debug_assert_eq!(delta.shape()[0], q as usize, "delta shape mismatch");
        debug_assert_eq!(delta.shape()[1], q as usize, "delta shape mismatch");
        debug_assert_eq!(delta.shape()[2], 2 as usize, "delta shape mismatch");
        let mut delta_vec = Vec::with_capacity(q);
        for i in 0..q {
            let mut delta_inner_vec = Vec::with_capacity(q);
            for j in 0..q {
                let out1 = *delta.get([i, j, 0]).unwrap();
                let out2 = *delta.get([i, j, 1]).unwrap();
                delta_inner_vec.push((out1, out2));
            }
            delta_vec.push(delta_inner_vec);
        }
        let delta = delta_vec;

        let mut null_transitions_vec = Vec::with_capacity(q);
        for i in 0..q {
            let mut null_inner_vec = Vec::with_capacity(q);
            for j in 0..q {
                let is_null = *null_transitions.get([i, j]).unwrap();
                null_inner_vec.push(is_null);
            }
            null_transitions_vec.push(null_inner_vec);
        }
        let null_transitions = null_transitions_vec;

        let mut random_transitions_vec = Vec::with_capacity(q);
        for i in 0..q {
            let mut random_inner_vec = Vec::with_capacity(q);
            for j in 0..q {
                let num = *random_transitions.get([i, j, 0]).unwrap();
                let idx = *random_transitions.get([i, j, 1]).unwrap();
                random_inner_vec.push((num, idx));
            }
            random_transitions_vec.push(random_inner_vec);
        }
        let random_transitions = random_transitions_vec;
        // is_random is true if any num in pair (num, idx) in random_transitions is non-zero
        let mut is_random = false;
        // random_depth is the maximum number of outputs for any randomized transition
        let mut random_depth = 1;
        for random_transitions_inner in &random_transitions {
            for &(num, _) in random_transitions_inner {
                if num != 0 {
                    is_random = true;
                    random_depth = random_depth.max(num);
                }
            }
        }

        let random_outputs_length = random_outputs.shape()[0];
        debug_assert_eq!(
            random_outputs.shape()[1],
            2 as usize,
            "random_outputs shape mismatch"
        );
        let mut random_outputs_vec = Vec::with_capacity(random_outputs_length);
        for i in 0..random_outputs_length {
            let out1 = *random_outputs.get([i, 0]).unwrap();
            let out2 = *random_outputs.get([i, 1]).unwrap();
            random_outputs_vec.push((out1, out2));
        }
        let random_outputs = random_outputs_vec;

        let transition_probabilities = transition_probabilities.to_vec().unwrap();
        debug_assert_eq!(
            random_outputs.len(),
            transition_probabilities.len(),
            "random_outputs and transition_probabilities length mismatch"
        );

        let t = 0;
        let rng = if let Some(s) = seed {
            SmallRng::seed_from_u64(s)
        } else {
            SmallRng::from_entropy()
        };

        let urn = Urn::new(config.clone(), seed);
        let updated_counts = Urn::new(vec![0; q], seed);
        let row_sums = vec![0; q];
        let row = vec![0; q];
        let silent = false;
        let do_gillespie = false;

        let mut reactions: Vec<(usize, usize, usize, usize)> = vec![];
        let mut reaction_probabilities = vec![];
        for i in 0..q {
            for j in 0..=i {
                // check if interaction is symmetric
                let mut symmetric = false;
                // Check that entries in delta array match
                let (mut o1, mut o2) = delta[i][j];
                if o1 > o2 {
                    (o1, o2) = (o2, o1);
                }
                let (mut o1_p, mut o2_p) = delta[j][i];
                if o1_p > o2_p {
                    (o1_p, o2_p) = (o2_p, o1_p);
                }
                if o1 == o1_p && o2 == o2_p {
                    // Check if those really were matching deterministic transitions
                    if !is_random
                        || (random_transitions[i][j].0 == 0 && random_transitions[j][i].0 == 0)
                    {
                        symmetric = true;
                    } else if is_random
                        && random_transitions[i][j].0 == random_transitions[j][i].0
                        && random_transitions[i][j].0 > 0
                    {
                        let (a, b) = (random_transitions[i][j].1, random_transitions[j][i].1);
                        symmetric = true;
                        for k in 0..random_transitions[i][j].0 {
                            let (mut o1, mut o2) = random_outputs[a + k];
                            if o1 > o2 {
                                (o1, o2) = (o2, o1);
                            }
                            let (mut o1_p, mut o2_p) = random_outputs[b + k];
                            if o1_p > o2_p {
                                (o1_p, o2_p) = (o2_p, o1_p);
                            }
                            if o1 != o1_p || o2 != o2_p {
                                symmetric = false;
                                // break;
                            }
                        }
                    }
                }
                // Other cases are not symmetric, such as a different number of random outputs based on order
                let indices = if symmetric {
                    vec![(i, j, 1.0)]
                } else {
                    // if interaction is not symmetric, each distinct order gets added as reactions with half probability
                    vec![(i, j, 0.5), (j, i, 0.5)]
                };
                for (a, b, p) in indices.iter() {
                    let a = *a;
                    let b = *b;
                    let p = *p;
                    if !null_transitions[a][b] {
                        let (num_outputs, start_idx) = random_transitions[a][b];
                        if is_random && num_outputs > 0 {
                            for k in 0..num_outputs {
                                let output = random_outputs[start_idx + k];
                                if output != (a, b) {
                                    reactions.push((a, b, output.0, output.1));
                                    reaction_probabilities
                                        .push(transition_probabilities[start_idx + k] * p);
                                }
                            }
                        } else {
                            reactions.push((a, b, o1, o2));
                            reaction_probabilities.push(p);
                        }
                    }
                }
            }
        }

        // next three fields are only used with Gillespie steps;
        // they will be set accordingly if we switch to Gillespie
        let propensities = vec![0.0; reactions.len()];
        let enabled_reactions = vec![0; reactions.len()];
        let num_enabled_reactions = 0;

        // below here we give meaningless default values to the other fields and rely on
        // set_n_parameters and get_enabled_reactions to set them to the correct values
        let logn = 0.0;
        let batch_threshold = 0;
        let gillespie_threshold = 0.0;
        let coll_table = vec![vec![0; 1]; 1];
        let coll_table_r_values = vec![0; 1];
        let coll_table_u_values = vec![0.0; 1];
        let r_constant = 0;

        let collision_counts = HashMap::new();

        let mut sim = SimulatorMultiBatch {
            n,
            t,
            q,
            delta,
            null_transitions,
            is_random,
            random_transitions,
            random_outputs,
            transition_probabilities,
            random_depth,
            rng,
            urn,
            updated_counts,
            logn,
            batch_threshold,
            row_sums,
            row,
            do_gillespie,
            silent,
            reactions,
            enabled_reactions,
            num_enabled_reactions,
            propensities,
            reaction_probabilities,
            gillespie_threshold,
            coll_table,
            coll_table_r_values,
            coll_table_u_values,
            r_constant,
            collision_counts,
        };
        sim.set_n_parameters();
        sim.update_enabled_reactions();
        sim
    }

    #[getter]
    pub fn config(&self) -> Vec<State> {
        self.urn.config.clone()
    }

    /// Run the simulation for a specified number of steps or until max time is reached
    #[pyo3(signature = (t_max, max_wallclock_time=3600.0))]
    pub fn run(&mut self, t_max: usize, max_wallclock_time: f64) -> PyResult<()> {
        if self.silent {
            return Err(PyValueError::new_err("Simulation is silent; cannot run."));
        }
        let max_wallclock_milliseconds: u64 = (max_wallclock_time * 1_000.0).ceil() as u64;
        let duration = Duration::from_millis(max_wallclock_milliseconds);
        let start_time = Instant::now();
        while self.t < t_max && start_time.elapsed() < duration {
            if self.silent {
                return Ok(());
            } else if self.do_gillespie {
                self.gillespie_step(t_max);
            } else {
                self.multibatch_step(t_max);
            }
        }
        Ok(())
    }

    /// Run the simulation until it is silent, i.e., no reactions are applicable.
    #[pyo3()]
    pub fn run_until_silent(&mut self) {
        while !self.silent {
            if self.do_gillespie {
                self.gillespie_step(0);
            } else {
                self.multibatch_step(0);
            }
        }
    }

    /// Reset the simulation with a new configuration
    /// Sets all parameters necessary to change the configuration.
    /// Args:
    ///     config: The configuration array to reset to.
    ///     t: The new value of :any:`t`. Defaults to 0.
    #[pyo3(signature = (config, t=0))]
    pub fn reset(&mut self, config: PyReadonlyArray1<State>, t: usize) -> PyResult<()> {
        self.collision_counts.clear();
        let config = config.to_vec().unwrap();
        self.urn.reset_config(&config);
        let n: usize = config.iter().sum();
        if n != self.n {
            self.n = n;
            self.set_n_parameters();
        }
        self.t = t;
        self.update_enabled_reactions();
        self.do_gillespie = false;
        Ok(())
    }

    #[pyo3(signature = (filename=None))]
    pub fn write_profile(&self, filename: Option<String>) -> PyResult<()> {
        let spans = flame::spans();
        if spans.is_empty() {
            println!("No profiling data available since flame_profiling feature disabled.");
            return Ok(());
        }

        let mut content = String::new();
        content.push_str("Flame Profile Report\n");
        content.push_str("===================\n");

        // Process the span tree recursively
        let mut span_data_map: HashMap<String, SpanData> = HashMap::new();
        for span in &spans {
            process_span(&mut span_data_map, span);
        }

        write_span_data(&mut content, &span_data_map, 0);

        // content.push_str(&format!("\nTotal time: {}ms\n", total_time_ms));

        if filename.is_none() {
            println!("{}", content);
        } else {
            let filename = filename.unwrap();
            let mut file = std::fs::File::create(filename)?;
            file.write_all(content.as_bytes())?;
        }

        Ok(())
    }
}

fn write_span_data(content: &mut String, span_data_map: &HashMap<String, SpanData>, depth: usize) {
    let indent = "  ".repeat(depth);
    let mut span_datas: Vec<&SpanData> = span_data_map.values().collect();
    span_datas.sort_by_key(|span_data| span_data.ns);
    span_datas.reverse();
    let mut name_length = 0;
    for span_data in &span_datas {
        name_length = name_length.max(span_data.name.len());
    }
    for span_data in span_datas {
        content.push_str(&format!(
            "{}{:name_length$}: {} ms\n",
            indent,
            span_data.name,
            span_data.ns / 1_000_000
        ));
        write_span_data(content, &span_data.children, depth + 1);
    }
}

struct SpanData {
    name: String,
    ns: u64,
    children: HashMap<String, SpanData>,
}

impl SpanData {
    fn new(name: String) -> Self {
        SpanData {
            name,
            ns: 0,
            children: HashMap::new(),
        }
    }
}

// Helper function to process spans recursively
fn process_span(span_data_map: &mut HashMap<String, SpanData>, span: &flame::Span) {
    let span_name = span.name.to_string();
    if !span_data_map.contains_key(&span_name) {
        span_data_map.insert(span_name.clone(), SpanData::new(span_name.clone()));
    }

    let span_data = span_data_map.get_mut(&span_name).unwrap();
    span_data.ns += span.delta;

    // Process children recursively
    for child in &span.children {
        process_span(&mut span_data.children, child);
    }
}

impl SimulatorMultiBatch {
    fn multibatch_step(&mut self, t_max: usize) -> () {
        self.updated_counts.reset();
        for i in 0..self.urn.order.len() {
            self.updated_counts.order[i] = self.urn.order[i];
        }

        // start with count 2 of delayed agents (guaranteed for the next interaction)
        let mut num_delayed: usize = 2;

        let now = Instant::now();
        let t1 = now.elapsed().as_secs_f64();

        // batch will go for at least batch_threshold interactions, unless passing t_max
        let mut end_step = self.t + self.batch_threshold;
        if t_max > 0 {
            end_step = end_step.min(t_max);
        }

        let uniform = Uniform::standard();

        flame::start("process collisions");

        let mut num_collisions = 0;
        while self.t + num_delayed / 2 < end_step {
            num_collisions += 1;

            flame::start("sample_coll");
            let mut u = self.rng.sample(uniform);
            let l = self.sample_coll(num_delayed + self.updated_counts.size, u, true);
            assert!(l > 0, "sample_coll must return at least 1");
            // add (l-1) // 2 pairs of delayed agents, the lth agent a was already picked, so has a collision
            num_delayed += 2 * ((l - 1) / 2);
            flame::end("sample_coll");

            // If the sampled collision happens after t_max, then include delayed agents up until t_max
            //   and do not perform the collision.
            if t_max > 0 && self.t + num_delayed / 2 >= t_max {
                assert!(t_max > self.t);
                num_delayed = (t_max - self.t) * 2;
                break;
            }

            let mut a: State;

            flame::start("process collision");
            // sample if a was a delayed or an updated agent
            u = self.rng.sample(uniform);
            // delayed with probability num_delayed / (num_delayed + num_updated)
            if (u * ((num_delayed + self.updated_counts.size) as f64)) <= num_delayed as f64 {
                // if a was delayed, need to first update a with its first interaction before the collision
                // c is the delayed partner that a interacted with, so add this interaction
                a = self.urn.sample_one().unwrap();
                let mut c = self.urn.sample_one().unwrap();
                (a, c) = self.unordered_delta(a, c);
                self.t += 1;
                // c is moved from delayed to updated, a is currently uncounted
                self.updated_counts.add_to_entry(c, 1);
                num_delayed -= 2;
            } else {
                // if a was updated, we simply sample a and remove it from updated counts
                a = self.updated_counts.sample_one().unwrap();
            }

            let mut b: State;

            if l % 2 == 0 {
                // when l is even, the collision must with a formally untouched agent
                b = self.urn.sample_one().unwrap();
            } else {
                // when l is odd, the collision is with the next agent, either untouched, delayed, or updated
                u = self.rng.sample(uniform);
                if ((u * ((self.n - 1) as f64)) as usize) < self.updated_counts.size {
                    // b is an updated agent, simply remove it
                    b = self.updated_counts.sample_one().unwrap();
                } else {
                    // we simply remove b from C if b is untouched
                    b = self.urn.sample_one().unwrap();
                    // if b was delayed, we have to do the past interaction
                    if ((u * (self.n - 1) as f64) as usize) < self.updated_counts.size + num_delayed
                    {
                        let mut c = self.urn.sample_one().unwrap();
                        (b, c) = self.unordered_delta(b, c);
                        self.t += 1;
                        self.updated_counts.add_to_entry(c, 1);
                        num_delayed -= 2;
                    }
                }
            }

            (a, b) = self.unordered_delta(a, b);
            self.t += 1;
            self.updated_counts.add_to_entry(a, 1);
            self.updated_counts.add_to_entry(b, 1);
            flame::end("process collision");
        }

        self.collision_counts
            .entry(num_collisions)
            .and_modify(|e| *e += 1)
            .or_insert(1);

        flame::end("process collisions");

        let t2 = now.elapsed().as_secs_f64();

        flame::start("process batch");

        self.do_gillespie = true; // if entire batch are null reactions, stays true and switches to gillspie

        let i_max = self
            .urn
            .sample_vector(num_delayed / 2, &mut self.row_sums)
            .unwrap();
        // println!("i_max = {i_max}");
        for i in 0..i_max {
            let o_i = self.urn.order[i];
            let j_max = self
                .urn
                .sample_vector(self.row_sums[o_i], &mut self.row)
                .unwrap();
            for j in 0..j_max {
                let o_j = self.urn.order[j];
                if self.is_random && self.random_transitions[o_i][o_j].0 > 0 {
                    // don't switch to gillespie because we did a random transition
                    // TODO: this might not switch to gillespie soon enough in certain cases
                    // better to test if the random transition is null or not
                    self.do_gillespie = false;
                    let (num_outputs, first_idx) = self.random_transitions[o_i][o_j];
                    // updates the first num_outputs entries of sample to hold a multinomial,
                    // giving the number of times for each random transition
                    let probabilities =
                        self.transition_probabilities[first_idx..first_idx + num_outputs].to_vec();
                    flame::start("multinomial sample");
                    let multinomial =
                        Multinomial::new(probabilities, self.row[o_j] as u64).unwrap();
                    let sample: DVector<u64> = self.rng.sample(multinomial);
                    flame::end("multinomial sample");
                    debug_assert_eq!(sample.len(), self.row[o_j], "sample length mismatch");
                    for c in 0..num_outputs {
                        let idx = first_idx + c;
                        let (out1, out2) = self.random_outputs[idx];
                        self.updated_counts.add_to_entry(out1, sample[c] as i64);
                        self.updated_counts.add_to_entry(out2, sample[c] as i64);
                    }
                } else {
                    if self.do_gillespie {
                        // if transition is non-null, we will set do_gillespie = False
                        self.do_gillespie = self.null_transitions[o_i][o_j];
                    }
                    // We are directly adding to updated_counts.config rather than using the function
                    //   updated_counts.add_to_entry for speed. None of the other urn features of updated_counts will
                    //   be used until it is reset in the next loop, so this is fine.
                    self.updated_counts.config[self.delta[o_i][o_j].0] += self.row[o_j];
                    self.updated_counts.config[self.delta[o_i][o_j].1] += self.row[o_j];
                }
            }
        }

        self.t += num_delayed / 2;
        // TODO: this is the only part scaling when the number of states (but not reached states) blows up
        self.urn.add_vector(&self.updated_counts.config);

        flame::end("process batch");

        let t3 = now.elapsed().as_secs_f64();

        const TIMING_PRINTS: bool = false;

        if TIMING_PRINTS {
            println!(
                "********\nself.batch_threshold before: {}",
                self.batch_threshold
            );
        }

        // Dynamically update batch threshold, by comparing the times t2 - t1 of the collision sampling and
        //   the time t_3 - t_2 of the batch processing. Batch_threshold is adjusted to try to ensure
        //   t_2 - t_1 = t_3 - t_2
        // self.batch_threshold = ((t3 - t2) / (t2 - t1)).powf(0.1) as usize * self.batch_threshold;
        // Keep the batch threshold within some fixed bounds.
        self.batch_threshold = self.batch_threshold.min(2 * self.n / 3);
        self.batch_threshold = self.batch_threshold.max(3);

        if TIMING_PRINTS {
            println!(
            "ratio: {:.1}, batching time: {:.1} us; collision time: {:.1} us, self.batch_threshold after: {}",
            (t3 - t2) / (t2 - t1),
            (t3 - t2) * 1_000_000.0,
            (t2 - t1) * 1_000_000.0,
            self.batch_threshold
        );

            if self.t > 1_000_000 {
                panic!();
            }
        }

        self.urn.sort();

        // update enabled_reactions if switching to gillespie

        self.update_enabled_reactions();
    }

    /// Chooses sender/receiver, then applies delta to input states a, b.
    fn unordered_delta(&mut self, a_p: State, b_p: State) -> (State, State) {
        let coin = self.rng.gen_bool(0.5);
        let mut a = a_p;
        let mut b = b_p;
        // swap roles of a, b and swap return order if coin is true
        if coin {
            (b, a) = (a, b);
        }
        let o1: State;
        let o2: State;
        if self.is_random && self.random_transitions[a][b].0 > 0 {
            // find the appropriate random output by linear search
            let mut k = self.random_transitions[a][b].1;
            let uniform = Uniform::standard();
            let mut u = self.rng.sample(uniform) - self.transition_probabilities[k];
            while u > 0.0 {
                k += 1;
                u -= self.transition_probabilities[k];
            }
            (o1, o2) = self.random_outputs[k];
        } else {
            (o1, o2) = self.delta[a][b];
        }
        // swap outputs if coin is true
        if coin {
            (o2, o1)
        } else {
            (o1, o2)
        }
    }

    /// Perform a Gillespie step.
    /// Samples the time until the next non-null interaction and updates.
    /// Args:
    /// num_steps:
    ///     If positive, the maximum value of :any:`t` that will be reached.
    ///     If the sampled time is greater than num_steps, then it will instead
    ///     be set to num_steps and no reaction will be performed.
    ///     (Because of the memoryless property of the geometric, this gives a
    ///     faithful simulation up to step num_steps).
    fn gillespie_step(&mut self, t_max: usize) -> () {
        let total_propensity = self.get_total_propensity();
        if total_propensity == 0.0 {
            self.silent = true;
            return;
        }
        let n: f64 = self.n as f64;
        let success_probability = total_propensity / (n * (n - 1.0) / 2.0);
        let mut enabled_reactions_changed = false;

        if success_probability > self.gillespie_threshold {
            self.do_gillespie = false;
        }
        let geometric = Geometric::new(success_probability).unwrap();
        let uniform = Uniform::new(0.0, total_propensity).unwrap();
        // add a geometric number of steps, based on success probability
        let steps: u64 = self.rng.sample(geometric);
        self.t += steps as usize;
        if t_max > 0 && self.t > t_max {
            self.t = t_max;
            return;
        }
        // sample the successful reaction r, currently just using linear search
        let mut x = self.rng.sample(uniform);
        let mut i = 0;
        while x > 0.0 {
            x -= self.propensities[self.enabled_reactions[i]];
            i += 1;
        }

        let r = &self.reactions[self.enabled_reactions[i - 1]];
        // updated with the successful reaction r
        // if any products were not already present, will update enabled_reactions
        if self.urn.config[r.2] == 0 || self.urn.config[r.3] == 0 {
            enabled_reactions_changed = true;
        }
        // this is a bit wasteful, but want to make sure the urn data structure stays intact
        self.urn.add_to_entry(r.0, -1);
        self.urn.add_to_entry(r.1, -1);
        self.urn.add_to_entry(r.2, 1);
        self.urn.add_to_entry(r.3, 1);
        // if any reactants are now absent, will update enabled_reactions
        if enabled_reactions_changed || self.urn.config[r.0] == 0 || self.urn.config[r.1] == 0 {
            self.update_enabled_reactions();
        }
    }

    /// Updates propensity vector, and returns total propensity:
    /// the probability the next interaction is non-null.
    fn get_total_propensity(&mut self) -> f64 {
        let mut total_propensity = 0.0;
        for j in 0..self.num_enabled_reactions {
            let i = self.enabled_reactions[j];
            let a = self.urn.config[self.reactions[i].0] as f64;
            let b = self.urn.config[self.reactions[i].1] as f64;
            if self.reactions[i].0 == self.reactions[i].1 {
                self.propensities[i] = (a * (a - 1.0) / 2.0) * self.reaction_probabilities[i];
            } else {
                self.propensities[i] = a * b * self.reaction_probabilities[i];
            }
            total_propensity += self.propensities[i];
        }
        total_propensity
    }

    /// Updates :any:`enabled_reactions`, :any:`num_enabled_reactions`, and :any:`silent`.
    fn update_enabled_reactions(&mut self) -> () {
        // flame::start("update_enabled_reactions");
        self.num_enabled_reactions = 0;
        for i in 0..self.reactions.len() {
            let (reactant_1, reactant_2) = (self.reactions[i].0, self.reactions[i].1);
            if (reactant_1 == reactant_2 && self.urn.config[reactant_1] >= 2)
                || (reactant_1 != reactant_2
                    && self.urn.config[reactant_1] >= 1
                    && self.urn.config[reactant_2] >= 1)
            {
                self.enabled_reactions[self.num_enabled_reactions] = i;
                self.num_enabled_reactions += 1;
            }
        }
        self.silent = self.num_enabled_reactions == 0;
        // flame::end("update_enabled_reactions");
    }

    /// Initialize all parameters that depend on the population size n.
    fn set_n_parameters(&mut self) -> () {
        self.logn = (self.n as f64).ln();
        // theoretical optimum for batch_threshold is Theta(sqrt(n / logn) * q) agents / batch
        self.batch_threshold = ((self.n as f64 / self.logn).sqrt()
            * (self.q as f64).min((self.n as f64).powf(0.7)))
            as usize;
        // first rough approximation for probability of successful reaction where we want to do gillespie
        self.gillespie_threshold = 2.0 / (self.n as f64).sqrt();

        // build table for precomputed coll(n, r, u) values
        // Note num_attempted_r_values may be too large; we break early if r >= n.
        let mut num_r_values = (10.0 * self.logn) as usize;
        let num_u_values = num_r_values;

        self.r_constant = (((1.5 * self.batch_threshold as f64).floor() as u64)
            / (((num_r_values - 2) * (num_r_values - 2)) as u64))
            .max(1) as usize;

        self.coll_table_r_values = vec![];
        for idx in 0..num_r_values - 1 {
            let r = 2 + self.r_constant * idx * idx;
            if r >= self.n {
                break;
            }
            self.coll_table_r_values.push(r);
        }
        self.coll_table_r_values.push(self.n);
        num_r_values = self.coll_table_r_values.len();

        self.coll_table_u_values = vec![0.0; num_u_values];
        for i in 0..num_u_values {
            self.coll_table_u_values[i] = i as f64 / (num_u_values as f64 - 1.0);
        }

        debug_assert_eq!(
            self.coll_table_r_values.len(),
            num_r_values,
            "self.coll_table_r_values length mismatch",
        );
        debug_assert_eq!(
            self.coll_table_u_values.len(),
            num_u_values,
            "self.coll_table_u_values length mismatch",
        );

        self.coll_table = vec![vec![0; num_u_values]; num_r_values];
        for r_idx in 0..num_r_values {
            for u_idx in 0..num_u_values {
                let r = self.coll_table_r_values[r_idx];
                let u = self.coll_table_u_values[u_idx];
                self.coll_table[r_idx][u_idx] = self.sample_coll(r, u, false);
            }
        }
    }

    /// Sample a collision event from the urn
    /// Returns a sample l ~ coll(n, r) from the collision length distribution.
    /// See Lemma 3 in the source paper https://arxiv.org/pdf/2005.03584.
    /// The distribution gives the number of agents needed to pick an agent twice,
    /// when r unique agents have already been selected.
    /// Inversion sampling with binary search is used, based on the formula
    ///     P(l > t) = (n - r)! / (n - r - t)! / (n^t).
    /// We sample a uniform random variable u, and find the value t such that
    ///     P(l > t) < U < P(l > t - 1).
    /// Taking logarithms and using the lgamma function, this required formula becomes
    ///     P(l > t) < U
    ///      <-->
    ///     lgamma(n - r + 1) - lgamma(n - r - t + 1) - t * log(n) < log(u).
    /// We will do binary search with bounds t_lo, t_hi that maintain the invariant
    ///     P(l > t_hi) < U and P(l > t_lo) >= U.
    /// Once we get t_lo = t_hi - 1, we can then return t = t_hi as the output.
    ///
    /// A value of fixed outputs for u, r will be precomputed, which gives a lookup table for starting values
    /// of t_lo, t_hi. This function will first get called to give coll(n, r_i, u_i) for a fixed range of values
    /// r_i, u_i. Then actual samples of coll(n, r, u) will find values r_i <= r < r_{i+1} and u_j <= u < u_{j+1}.
    /// By monotonicity in u, r, we can then set t_lo = coll(n, r_{i+i}, u_{j+1}) and t_hi = coll(n, r_i, u_j).
    ///
    /// Args:
    ///     r: The number of agents which have already been chosen.
    ///     u: A uniform random variable.
    ///     has_bounds: Has the table for precomputed values of r, u already been computed?
    ///         (This will be false while the function is being called to populate the table.)
    ///
    /// Returns:
    ///     The number of sampled agents to get the first collision (including the collided agent).
    fn sample_coll(&self, r: usize, u: f64, has_bounds: bool) -> usize {
        let mut t_lo: usize;
        let mut t_hi: usize;
        let logu = u.ln();
        debug_assert!(self.n + 1 - r > 0);
        let diff = (self.n + 1 - r) as f64;
        let lhs = ln_gamma(diff) - logu;
        // The condition P(l < t) < U becomes
        //     lhs < lgamma(n - r - t + 1) + t * log(n)

        if has_bounds {
            // Look up bounds from coll_table.
            // For r values, we invert the definition of self.coll_table_r_values:
            //   np.array([2 + self.r_constant * (i ** 2) for i in range(self.num_r_values - 1)] + [self.n])
            let i = (((r - 2) as f64).sqrt() / self.r_constant as f64) as usize;
            let i = i.min(self.coll_table_r_values.len() - 2);

            // for u values we similarly invert the definition: np.linspace(0, 1, num_u_values)
            let j = (u * (self.coll_table_u_values.len() - 1) as f64) as usize;

            debug_assert!(self.coll_table_r_values[i] <= r);
            debug_assert!(r <= self.coll_table_r_values[i + 1]);
            debug_assert!(self.coll_table_u_values[j] <= u);
            debug_assert!(u <= self.coll_table_u_values[j + 1]);
            t_lo = self.coll_table[i + 1][j + 1];
            t_hi = self.coll_table[i][j].min(self.n - r + 1);
        } else {
            // When building the table, we start with bounds that always hold.
            if r >= self.n {
                return 1;
            }
            t_lo = 0;
            t_hi = self.n - r;
        }

        // We maintain the invariant that P(l > t_lo) >= u and P(l > t_hi) < u
        // Equivalently, lhs >= lgamma(n - r - t_lo + 1) + t_lo * logn and
        //               lhs <  lgamma(n - r - t_hi + 1) + t_hi * logn
        while t_lo < t_hi - 1 {
            let t_mid = (t_lo + t_hi) / 2;
            if lhs < ln_gamma((self.n - r + 1 - t_mid) as f64) + (t_mid as f64) * self.logn {
                t_hi = t_mid;
            } else {
                t_lo = t_mid;
            }
        }

        t_hi
    }
}
