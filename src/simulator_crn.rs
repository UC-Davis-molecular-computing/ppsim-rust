use std::collections::{HashMap};
use std::io::Write;
use std::time::{Duration, Instant};

use crate::flame;

use numpy::PyArrayMethods;
use numpy::PyUntypedArrayMethods;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
// use ndarray::prelude::*;
use ndarray::ArrayD;

use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArrayDyn};
use rand::rngs::SmallRng;
use rand::Rng;
use rand::SeedableRng;
// use statrs::distribution::{Geometric, Uniform};

use crate::simulator_abstract::Simulator;

use crate::urn::Urn;
#[allow(unused_imports)]
use crate::util::{ln_factorial, ln_gamma, multinomial_sample};

type State = usize;

//TODO: consider using ndarrays instead of multi-dimensional vectors
// I think native Rust arrays won't work because their size needs to be known at compile time.
#[pyclass(extends = Simulator)]
pub struct SimulatorCRNMultiBatch {
    /// The population size (sum of values in urn.config).
    #[pyo3(get, set)]
    pub n: usize,
    /// The current number of elapsed interaction steps.
    #[pyo3(get, set)]
    pub t: usize,
    /// The total number of states (length of urn.config).
    pub q: usize,
    /// The order of reactions, i.e. the number of reactants.
    #[pyo3(get, set)]
    pub o: usize,
    /// The generativity of reactions, i.e. the number of products minus the number of reactants.
    #[pyo3(get, set)]
    pub g: isize,
    /// An (o + 1)-dimensional array. The first o dimensions represent reactants. After indexing through
    /// the first o dimensions, the last dimension always has size two, with elements (`num_outputs`, `first_idx`).
    /// `num_outputs` is the number of possible outputs if transition i,j --> ... is random,
    /// otherwise it is 0. `first_idx` gives the starting index to find
    /// the outputs in the array `self.random_outputs` if it is random.
    /// #[pyo3(get, set)] // XXX: for testing
    pub random_transitions: ArrayD<usize>,
    /// A 1D array of tuples containing all outputs of random transitions,
    /// whose indexing information is contained in random_transitions.
    /// For example, if there are random transitions
    /// 3,4 --> 5,6,7 and 3,4 --> 7,7,8 and 3,4 --> 3,2,1, then
    /// `random_transitions[3][4] = (3, first_idx)` for some `first_idx`, and
    /// `random_outputs[first_idx]   = (5,6,7)`,
    /// `random_outputs[first_idx+1] = (7,7,8)`, and
    /// `random_outputs[first_idx+2] = (3,2,1)`.
    #[pyo3(get, set)] // XXX: for testing
    pub random_outputs: Vec<Vec<State>>,
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
    #[allow(dead_code)]
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
    #[allow(dead_code)]
    row_sums: Vec<usize>,
    /// Array which stores the counts of responder agents for each type of
    /// initiator agent (one row of the 'D' matrix from the paper).
    #[allow(dead_code)]
    row: Vec<usize>,
    /// Vector holding multinomial samples when doing randomized transitions.
    #[allow(dead_code)]
    m: Vec<usize>,
    /// A boolean determining if we are currently doing Gillespie steps.
    #[pyo3(get, set)]
    pub do_gillespie: bool,
    /// A boolean determining if the configuration is silent (all interactions are null).
    #[pyo3(get, set)]
    pub silent: bool,
    // /// A list of reactions, as (input, input, output, output). TODO: re-add these when re-adding the ability to do gillespie.
    // #[pyo3(get, set)]
    // pub reactions: Vec<(State, State, State, State)>,
    // /// An array holding indices into `self.reactions` of all currently enabled
    // /// (i.e., applicable; positive counts of reactants) reactions.
    // #[pyo3(get, set)]
    // pub enabled_reactions: Vec<usize>,
    // /// The number of meaningful indices in `self.enabled_reactions`.
    // #[pyo3(get, set)]
    // pub num_enabled_reactions: usize,
    // /// An array of length `self.reactions.len()` holding the propensities of each reaction.
    // propensities: Vec<f64>, // these are used only when doing Gillespie steps and are all 0 otherwise
    /// The probability of each reaction. 
    /// #[pyo3(get, set)]
    ///  pub reaction_probabilities: Vec<f64>,
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
    /// If true, unconditionally use the Gillespie algorithm.
    gillespie_always: bool,
    /// Length-q^o vector that has all possible subsets of [q]^o, for easy indexing over reactions.
    #[allow(dead_code)]
    reaction_indices: Vec<Vec<usize>>,
}

// fn py_print(py: Python, msg: &str) {
//     let sys = py.import("sys").unwrap();
//     let stdout = sys.getattr("stdout").unwrap();
//     stdout.call_method1("write", (format!("{}\n", msg),)).unwrap();
//     stdout.call_method0("flush").unwrap();
// }

#[pymethods]
impl SimulatorCRNMultiBatch {
    /// Initializes the main data structures for SimulatorCRNMultiBatch.
    /// We take numpy arrays as input because that's how n-dimensional arrays are represented in python.
    /// We convert those numpy arrays into rust ndarrray::ArrayD for storage.
    ///
    /// Args:
    ///     init_array: A length-q integer array of counts representing the initial configuration.
    ///     delta: A 2D q x q x 2 array representing the transition function. TODO remove if I can, might not be able to for consistency with other simulators.
    ///         Delta[i, j] gives contains the two output states.
    ///     random_transitions: A q^o x 2 array. That is, it has o+1 dimensions, all but the last have length q,
    ///         and the last dimension always has length two.
    ///         Entry [r, 0] is the number of possible outputs if transition on reactant set r is random, 
    ///         otherwise it is 0. Entry [r, 1] gives the starting index to find the outputs in the array random_outputs if it is random.
    ///     random_outputs: A ? x (o + g) array containing all outputs of random transitions,
    ///         whose indexing information is contained in random_transitions.
    ///     transition_probabilities: A 1D length-? array containing all random transition probabilities,
    ///         whose indexing matches random_outputs.
    ///     seed (optional): An integer seed for the pseudorandom number generator.
    #[new]
    #[pyo3(signature = (init_config, _delta, random_transitions, random_outputs, transition_probabilities, transition_order, gillespie=false, seed=None))]
    pub fn new(
        init_config: PyReadonlyArray1<State>,
        _delta: PyReadonlyArrayDyn<State>,
        random_transitions: PyReadonlyArrayDyn<usize>,
        random_outputs: PyReadonlyArray2<State>,
        transition_probabilities: PyReadonlyArray1<f64>,
        transition_order: String,
        gillespie: bool,
        seed: Option<u64>,
    ) -> (Self, Simulator) {
        
        let init_config = init_config.to_vec().unwrap();
        let q: usize = init_config.len() as State;

        let num_inputs = random_transitions.shape().len() - 1;
        let num_outputs = random_transitions.shape()[num_inputs];

        for i in 0..num_inputs {
            assert_eq!(random_transitions.shape()[i], q, "random_transitions shape mismatch");
        }
        let random_transitions: ArrayD<usize> = random_transitions.as_array().to_owned();

        let random_outputs_length = random_outputs.shape()[0];
        assert_eq!(
            random_outputs.shape()[1],
            num_outputs,
            "random_outputs shape mismatch"
        );
        let mut random_outputs_vec = Vec::with_capacity(random_outputs_length);
        for i in 0..random_outputs_length {
            let mut output_element = Vec::new();
            for j in 0..num_outputs {
                output_element.push(*random_outputs.get([i,j]).unwrap());

            }
            random_outputs_vec.push(output_element);
        }
        let random_outputs = random_outputs_vec;

        let transition_probabilities = transition_probabilities.to_vec().unwrap();
        assert_eq!(
            random_outputs.len(),
            transition_probabilities.len(),
            "random_outputs and transition_probabilities length mismatch"
        );

        (SimulatorCRNMultiBatch::from_delta_random(
            init_config,
            random_transitions,
            random_outputs,
            transition_probabilities,
            transition_order,
            gillespie,
            seed,
        ), Simulator::default())
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
        let max_wallclock_milliseconds = (max_wallclock_time * 1_000.0).ceil() as u64;
        let duration = Duration::from_millis(max_wallclock_milliseconds);
        let start_time = Instant::now();
        while self.t < t_max && start_time.elapsed() < duration {
            // println!("self.gillespie_always = {}", self.gillespie_always);
            if self.gillespie_always {
                self.do_gillespie = true;
            }
            if self.silent {
                return Ok(());
            } else if self.do_gillespie {
                flame::start("gillespie step");
                self.gillespie_step(t_max);
                flame::end("gillespie step");
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
            if self.gillespie_always {
                self.do_gillespie = true;
            }
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
        let config = config.to_vec().unwrap();
        self.urn.reset_config(&config);
        let n: usize = config.iter().sum();
        if n != self.n {
            self.n = n;
            self.set_n_parameters();
        }
        self.t = t;
        self.update_enabled_reactions();
        self.do_gillespie = self.gillespie_always;
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
        content.push_str("====================\n");

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

    #[pyo3(signature = (r, u, has_bounds=false))]
    pub fn sample_collision(&self, r: usize, u: f64, has_bounds: bool) -> usize {
        self.sample_coll(r, u, has_bounds)
    }

    /// Sample from birthday-like distribution "directly". This is the number of times
    /// we can do the following before seeing something painted red: sample o objects without replacement,
    /// remove them and add o + g red objects, given that r objects out of n are initially red.
    #[pyo3()]
    pub fn sample_collision_directly(&mut self, n: usize, r: usize) -> usize {
        let mut idx = 0usize;
        assert!(r < n, "r must be less than n");
        assert!(n < usize::MAX, "n must be less than usize::MAX");
        let mut num_seen = r;
        let mut pop_size = n;
        if r == 0 && self.o.wrapping_add(self.g as usize) == 0 {
            // Super duper edge case: this means that the CRN only consumes things, no reaction
            // has anything on the right-hand side. In this case, there are never any collisions.
            return n / self.o;
        }
        loop {
            for _ in 0..self.o {
                let sample = self.rng.gen_range(0..pop_size);
                if sample < num_seen {
                    return idx;
                }
                pop_size -= 1;
            }
            idx += 1;
            pop_size += self.o.wrapping_add(self.g as usize);
            num_seen += self.o.wrapping_add(self.g as usize);
        }
    }
}

// Helper function to get all q^o possible sets of reactants, i.e. all subsets of [q]^o.
fn all_reaction_indices(q:usize, o:usize) -> Vec<Vec<usize>> {
    let output: Vec<Vec<usize>> = Vec::with_capacity(q.pow(o.try_into().unwrap()));
    // for i in 0..q.pow(o.try_into().unwrap()) {
    //     let mut elem = Vec::new();
    //     for j in 0..o {
    //         elem.push(i % q.pow(j.try_into().unwrap()));
    //     }
    //     output.push(elem);
    // }
    output
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

// const CAP_BATCH_THRESHOLD: bool = true;

impl SimulatorCRNMultiBatch {
    fn multibatch_step(&mut self, _t_max: usize) -> () {
        unimplemented!()
        // let max_batch_threshold = self.n / 4;
        // if CAP_BATCH_THRESHOLD && self.batch_threshold > max_batch_threshold {
        //     self.batch_threshold = max_batch_threshold;
        // }
        // self.updated_counts.reset();

        // for i in 0..self.urn.order.len() {
        //     self.updated_counts.order[i] = self.urn.order[i];
        // }

        // // start with count 2 of delayed agents (guaranteed for the next interaction)
        // let mut num_delayed: usize = 2;

        // // let now = Instant::now();
        // // let t1 = now.elapsed().as_secs_f64();

        // // batch will go for at least batch_threshold interactions, unless passing t_max
        // let mut end_step = self.t + self.batch_threshold;
        // if t_max > 0 {
        //     end_step = end_step.min(t_max);
        // }

        // let uniform = Uniform::standard();

        // flame::start("process collisions");

        // while self.t + num_delayed / 2 < end_step {
        //     let mut u = self.rng.sample(uniform);

        //     let pp = true;
        //     let has_bounds = false;
        //     flame::start("sample_coll");
        //     let l = self.sample_coll(num_delayed + self.updated_counts.size, u, has_bounds, pp);
        //     flame::end("sample_coll");

        //     assert!(l > 0, "sample_coll must return at least 1");

        //     // add (l-1) // 2 pairs of delayed agents, the lth agent a was already picked, so has a collision
        //     num_delayed += 2 * ((l - 1) / 2);

        //     // If the sampled collision happens after t_max, then include delayed agents up until t_max
        //     //   and do not perform the collision.
        //     if t_max > 0 && self.t + num_delayed / 2 >= t_max {
        //         assert!(t_max > self.t);
        //         num_delayed = (t_max - self.t) * 2;
        //         break;
        //     }

        //     /*
        //     Definitions from paper https://arxiv.org/abs/2005.03584

        //     - *untouched* agents did not interact in the current epoch (multibatch step).
        //       Hence, all agents are labeled untouched at the beginning of an epoch.

        //     - *updated* agents took part in at least one interaction that was already evaluated.
        //       Thus, updated agents are already assigned their most recent state.

        //     - *delayed* agents took part in exactly one interaction that was not yet evaluated.
        //       Thus, delayed agents are still in the same state they had at the beginning of the
        //       epoch, but are scheduled to interact at a later point in time. We additionally
        //       require that their interaction partner is also labeled delayed.
        //      */

        //     let mut initiator: State; // initiator, called a in Cython implementation
        //     let mut responder: State; // responder, called b in Cython implementation

        //     flame::start("process collision");

        //     // sample if initiator was delayed or updated
        //     u = self.rng.sample(uniform);
        //     // initiator is delayed with probability num_delayed / (num_delayed + num_updated)
        //     let initiator_delayed =
        //         u * ((num_delayed + self.updated_counts.size) as f64) < num_delayed as f64;
        //     if initiator_delayed {
        //         // if initiator is delayed, need to first update it with its first interaction before the collision
        //         // c is the delayed partner that initiator interacted with, so add this interaction
        //         initiator = self.urn.sample_one().unwrap();
        //         let mut c = self.urn.sample_one().unwrap();
        //         (initiator, c) = self.unordered_delta(initiator, c);
        //         self.t += 1;
        //         // c is moved from delayed to updated, initiator is currently uncounted;
        //         // we've updated initiator state, but don't put it in updated_counts because
        //         // we'd just need to take it back out to do the initiator/responder interaction
        //         self.updated_counts.add_to_entry(c, 1);
        //         num_delayed -= 2;
        //     } else {
        //         // if initiator is updated, we simply sample initiator and remove it from updated_counts
        //         initiator = self.updated_counts.sample_one().unwrap();
        //     }

        //     // sample responder
        //     if l % 2 == 0 {
        //         // when l is even, the collision must with a formerly untouched agent
        //         responder = self.urn.sample_one().unwrap();
        //     } else {
        //         // when l is odd, the collision is with the next agent, either untouched, delayed, or updated
        //         u = self.rng.sample(uniform);
        //         if (u * ((self.n - 1) as f64)) < self.updated_counts.size as f64 {
        //             // responder is an updated agent, simply remove it
        //             responder = self.updated_counts.sample_one().unwrap();
        //         } else {
        //             // responder is untouched or delayed; we remove responder from self.urn in either case
        //             responder = self.urn.sample_one().unwrap();
        //             // if responder is delayed, we also have to do the past interaction
        //             if (u * (self.n - 1) as f64) < (self.updated_counts.size + num_delayed) as f64 {
        //                 let mut c = self.urn.sample_one().unwrap();
        //                 (responder, c) = self.unordered_delta(responder, c);
        //                 self.t += 1;
        //                 self.updated_counts.add_to_entry(c, 1);
        //                 num_delayed -= 2;
        //             }
        //         }
        //     }

        //     (initiator, responder) = self.unordered_delta(initiator, responder);
        //     self.t += 1;
        //     self.updated_counts.add_to_entry(initiator, 1);
        //     self.updated_counts.add_to_entry(responder, 1);

        //     flame::end("process collision");
        // }

        // flame::end("process collisions");

        // flame::start("process batch");

        // self.do_gillespie = true; // if entire batch are null reactions, stays true and switches to gillspie

        // let i_max = self
        //     .urn
        //     .sample_vector(num_delayed / 2, &mut self.row_sums)
        //     .unwrap();

        // for i in 0..i_max {
        //     let o_i = self.urn.order[i];
        //     let j_max = self
        //         .urn
        //         .sample_vector(self.row_sums[o_i], &mut self.row)
        //         .unwrap();

        //     for j in 0..j_max {
        //         let o_j = self.urn.order[j];
        //         if self.is_random && self.random_transitions[o_i][o_j].0 > 0 {
        //             // don't switch to gillespie because we did a random transition
        //             // TODO: this might not switch to gillespie soon enough in certain cases
        //             // better to test if the random transition is null or not
        //             self.do_gillespie = false;
        //             let (num_outputs, first_idx) = self.random_transitions[o_i][o_j];
        //             // updates the first num_outputs entries of sample to hold a multinomial,
        //             // giving the number of times for each random transition
        //             let probabilities =
        //                 self.transition_probabilities[first_idx..first_idx + num_outputs].to_vec();
        //             flame::start("multinomial sample");
        //             multinomial_sample(self.row[o_j], &probabilities, &mut self.m, &mut self.rng);
        //             flame::end("multinomial sample");
        //             assert_eq!(
        //                 self.m.iter().sum::<usize>(),
        //                 self.row[o_j],
        //                 "sample sum mismatch"
        //             );
        //             for c in 0..num_outputs {
        //                 let idx = first_idx + c;
        //                 let (out1, out2) = self.random_outputs[idx];
        //                 self.updated_counts.add_to_entry(out1, self.m[c] as i64);
        //                 self.updated_counts.add_to_entry(out2, self.m[c] as i64);
        //             }
        //         } else {
        //             if self.do_gillespie {
        //                 // if transition is non-null, we will set do_gillespie = False
        //                 self.do_gillespie = self.null_transitions[o_i][o_j];
        //             }
        //             // We are directly adding to updated_counts.config rather than using the function
        //             //   updated_counts.add_to_entry for speed. None of the other urn features of updated_counts will
        //             //   be used until it is reset in the next loop, so this is fine.
        //             self.updated_counts.config[self.delta[o_i][o_j].0] += self.row[o_j];
        //             self.updated_counts.config[self.delta[o_i][o_j].1] += self.row[o_j];
        //         }
        //     }
        // }

        // self.t += num_delayed / 2;
        // // TODO: this is the only part scaling when the number of states (but not reached states) blows up
        // self.urn.add_vector(&self.updated_counts.config);

        // flame::end("process batch");

        // self.urn.sort();
        // self.update_enabled_reactions();
    }

    /// Chooses sender/receiver, then applies delta to input states a, b.
    // fn unordered_delta(&mut self, a: State, b: State) -> (State, State) {
    //     unimplemented!()
    //     // let heads = self.rng.gen_bool(0.5); // fair coin flip
    //     // let mut i1 = a;
    //     // let mut i2 = b;
    //     // // swap roles of a, b and swap return order if heads is true
    //     // if heads {
    //     //     (i2, i1) = (i1, i2);
    //     // }
    //     // let o1: State;
    //     // let o2: State;
    //     // if self.is_random && self.random_transitions[i1][i2].0 > 0 {
    //     //     // find the appropriate random output by linear search
    //     //     let mut k = self.random_transitions[i1][i2].1;
    //     //     let uniform = Uniform::standard();
    //     //     let mut u = self.rng.sample(uniform) - self.transition_probabilities[k];
    //     //     while u > 0.0 {
    //     //         k += 1;
    //     //         u -= self.transition_probabilities[k];
    //     //     }
    //     //     (o1, o2) = self.random_outputs[k];
    //     // } else {
    //     //     (o1, o2) = self.delta[i1][i2];
    //     // }
    //     // // swap outputs if heads is true
    //     // if heads {
    //     //     (o2, o1)
    //     // } else {
    //     //     (o1, o2)
    //     // }
    // }

    /// Perform a Gillespie step.
    /// Samples the time until the next non-null interaction and updates.
    /// Args:
    /// num_steps:
    ///     If positive, the maximum value of :any:`t` that will be reached.
    ///     If the sampled time is greater than num_steps, then it will instead
    ///     be set to num_steps and no reaction will be performed.
    ///     (Because of the memoryless property of the geometric, this gives a
    ///     faithful simulation up to step num_steps).
    fn gillespie_step(&mut self, _t_max: usize) -> () {
        unimplemented!()
        // // println!("gillespie_step at interaction {}", self.t);
        // let total_propensity = self.get_total_propensity();
        // if total_propensity == 0.0 {
        //     self.silent = true;
        //     return;
        // }
        // let n_choose_2 = (self.n * (self.n - 1) / 2) as f64;
        // let success_probability = total_propensity / n_choose_2;

        // if success_probability > self.gillespie_threshold {
        //     self.do_gillespie = false;
        // }
        // let geometric = Geometric::new(success_probability).unwrap();
        // let uniform = Uniform::new(0.0, total_propensity).unwrap();
        // // add a geometric number of steps, based on success probability
        // let steps: u64 = self.rng.sample(geometric);
        // self.t += steps as usize;
        // if t_max > 0 && self.t > t_max {
        //     self.t = t_max;
        //     return;
        // }
        // // sample the successful reaction r, currently just using linear search
        // let mut x: f64 = self.rng.sample(uniform);
        // let mut i = 0;
        // while x > 0.0 {
        //     x -= self.propensities[self.enabled_reactions[i]];
        //     i += 1;
        // }
        // let (r1, r2, p1, p2) = self.reactions[self.enabled_reactions[i - 1]];

        // // update with the successful reaction r1+r2 --> p1+p2
        // // if any products were not already present, or any reactants went absent, will update enabled_reactions
        // let new_products = self.urn.config[p1] == 0 || self.urn.config[p2] == 0;
        // let absent_reactants = self.urn.config[r1] == 0 || self.urn.config[r2] == 0;
        // if new_products || absent_reactants {
        //     self.update_enabled_reactions();
        // }

        // // now apply the reaction
        // self.urn.add_to_entry(r1, -1);
        // self.urn.add_to_entry(r2, -1);
        // self.urn.add_to_entry(p1, 1);
        // self.urn.add_to_entry(p2, 1);
    }

    /// Updates propensity vector, and returns total propensity:
    /// the probability the next interaction is non-null.
    // fn get_total_propensity(&mut self) -> f64 {
    //     unimplemented!()
    //     // let mut total_propensity = 0.0;
    //     // for j in 0..self.num_enabled_reactions {
    //     //     let i = self.enabled_reactions[j];
    //     //     let a = self.urn.config[self.reactions[i].0] as f64;
    //     //     let b = self.urn.config[self.reactions[i].1] as f64;
    //     //     if self.reactions[i].0 == self.reactions[i].1 {
    //     //         self.propensities[i] = (a * (a - 1.0) / 2.0) * self.reaction_probabilities[i];
    //     //     } else {
    //     //         self.propensities[i] = a * b * self.reaction_probabilities[i];
    //     //     }
    //     //     total_propensity += self.propensities[i];
    //     // }
    //     // total_propensity
    // }

    /// Updates :any:`enabled_reactions`, :any:`num_enabled_reactions`, and :any:`silent`.
    fn update_enabled_reactions(&mut self) -> () {
        unimplemented!()
        // // flame::start("update_enabled_reactions");
        // self.num_enabled_reactions = 0;
        // for i in 0..self.reactions.len() {
        //     let (reactant_1, reactant_2) = (self.reactions[i].0, self.reactions[i].1);
        //     if (reactant_1 == reactant_2 && self.urn.config[reactant_1] >= 2)
        //         || (reactant_1 != reactant_2
        //             && self.urn.config[reactant_1] >= 1
        //             && self.urn.config[reactant_2] >= 1)
        //     {
        //         self.enabled_reactions[self.num_enabled_reactions] = i;
        //         self.num_enabled_reactions += 1;
        //     }
        // }
        // self.silent = self.num_enabled_reactions == 0;
        // // flame::end("update_enabled_reactions");
    }

    /// Initialize all parameters that depend on the population size n.
    fn set_n_parameters(&mut self) -> () {
        self.logn = (self.n as f64).ln();
        // theoretical optimum for batch_threshold is Theta(sqrt(n / logn) * q) agents / batch
        // let batch_constant = 2_i32.pow(2) as usize;
        let batch_constant = 1 as usize;
        self.batch_threshold = batch_constant
            * ((self.n as f64 / self.logn).sqrt() * (self.q as f64).min((self.n as f64).powf(0.7)))
                as usize;
        // println!("batch_threshold = {}", self.batch_threshold);
        self.batch_threshold = self.n / 2;
        // first rough approximation for probability of successful reaction where we want to do gillespie
        self.gillespie_threshold = 2.0 / (self.n as f64).sqrt();

        // build table for precomputed coll(n, r, u) values
        // Note num_attempted_r_values may be too large; we break early if r >= n.
        // let mut num_r_values = (10.0 * self.logn) as usize;
        let mut num_r_values = (5.0 * self.logn) as usize;
        let num_u_values = num_r_values;

        self.r_constant = (((1.5 * self.batch_threshold as f64) as usize)
            / ((num_r_values - 2) * (num_r_values - 2)))
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

        assert_eq!(
            self.coll_table_r_values.len(),
            num_r_values,
            "self.coll_table_r_values length mismatch",
        );
        assert_eq!(
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
    /// Returns a sample l ~ coll(n, r, o, g) from the collision length distribution.
    /// See TODO: add reference to paper once it's on arxiv.
    /// The distribution gives the number of reactions that will occur before a collision.
    /// Inversion sampling with binary search is used, based on the formula
    ///     P(l >= t) = (n-r)! / (n-r-to)! * prod_{j=0}^{o-1} [(n-g-j)!(g) / (n+g(t-1)-j)!(g)].
    /// !(g) denotes a multifactorial: n!(g) = n * (n - g) * (n - 2g) * ..., until these terms become nonpositive. 
    /// This is the formula when g > 0; when g = 0 or g < 0, the formulas are slightly different 
    /// (see the full formula for coll(n,r,o,g) in the paper), but the method is the same:
    /// We sample a uniform random variable u, and find the value t such that
    ///     P(l >= t) < U < P(l >= t - 1).
    /// Taking logarithms and using the ln_gamma function, this required formula becomes
    ///     P(l >= t) < U
    ///       <-->
    ///     ln_gamma(n-r+1) - ln_gamma(n-r-to+1) + sum_{j=0}^{o-1} [log((n-g-j)!(g)) - log((n+g(t-1)-j)!(g))] < log(U).
    /// which can be rewritten by using the fact that gamma(x) = (x - 1) * gamma(x-1) even for non-integer x,
    /// by factoring out a factor of g from every term in the multifactorial. 
    /// To this end, if we let a and b denote the number of terms in these multifactorial products,
    /// that is, let a = ceil((n-g-j)/g) and b = ceil((n+g(t-1)-j)/g),
    ///     ln_gamma(n-r+1) - ln_gamma(n-r-to+1) + sum_{j=0}^{o-1} [log(g^a * gamma((n-j)/g) / gamma((n-ag-j)/g)) - log(g^b * gamma((n+gt-j)/g) / gamma((n+g(t-b)-j)/g))] < log(U).
    ///     ln_gamma(n-r+1) - ln_gamma(n-r-to+1) + sum_{j=0}^{o-1} [a*log(g) + ln_gamma((n-j)/g) - ln_gamma((n-ag-j)/g) - b*log(g) - ln_gamma((n+gt-j)/g) + ln_gamma((n+g(t-b)-j)/g)] < log(U).
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
    /// Returns:
    ///     The number of interactions that will happen before the next collision.
    ///     Note that this is a different convention from :any:`SimulatorMultiBatch`, which 
    ///     returns the index at which an agent collision occurs.
    pub fn sample_coll(&self, r: usize, u: f64, _has_bounds: bool) -> usize {
        let mut t_lo: usize;
        let mut t_hi: usize;
        assert!(self.n + 1 - r > 0);

        // We compute all the terms of the distribution that don't depend on t and collect them in lhs.
        // Below, we will compute all the terms that do depend on t, and collect them in rhs.
        let logu = u.ln();
        let diff = self.n + 1 - r;
        let ln_gamma_diff = ln_factorial(diff - 1);

        let mut lhs = ln_gamma_diff - logu;
        
        if self.g > 0 {
            for j in 0..self.o {
                // Calculates a = ceil((n-g-j)/g)
                let num_static_terms: f64 = (((self.n - j) as f64 - self.g as f64) / self.g as f64).ceil();
                lhs += num_static_terms * (self.g as f64).ln();
                lhs += ln_gamma((self.n - j) as f64 / self.g as f64);
                lhs -= ln_gamma((self.n as f64 - (num_static_terms * self.g as f64) - j as f64) / self.g as f64);
            }
        } else if self.g < 0 {
            let unsigned_g = (-1 * self.g) as usize;
            for j in 0..self.o {
                // Calculates a = ceil((n-j)/|g|)
                let num_static_terms: f64 = ((self.n - j) as f64 / unsigned_g as f64).ceil();
                lhs -= num_static_terms * (unsigned_g as f64).ln();
                lhs -= ln_gamma((self.n + unsigned_g - j) as f64 / unsigned_g as f64);
                lhs += ln_gamma((self.n as f64 - ((num_static_terms - 1.0) * unsigned_g as f64) - j as f64) / unsigned_g as f64);
            }
        } else {
            // Nothing to do here. There are no other static terms in the g = 0 case.
        }
        

        // const PRINT: bool = true;
        // let logn_minus_1 = ((self.n - 1) as f64).ln();

        // if has_bounds {
        //     if PRINT {
        //         use stybulate::{Cell, Headers, Style, Table};
        //         let mut headers: Vec<String> = vec!["".to_string()];
        //         let mut first_row = vec![Cell::from(" r\\u")];
        //         for i in 0..self.coll_table_u_values.len() {
        //             // headers.push(format!("{:.2}", self.coll_table_u_values[i]));
        //             headers.push(i.to_string());
        //             first_row.push(Cell::from(&format!("{:.2}", self.coll_table_u_values[i])));
        //         }
        //         let mut table_data: Vec<Vec<Cell>> = vec![first_row];
        //         for i in 0..self.coll_table.len() {
        //             let first = format!("{i:2}:{}", self.coll_table_r_values[i].to_string());
        //             let mut row = vec![Cell::from(&first)];
        //             for j in 0..self.coll_table[i].len() {
        //                 let entry = format!("{}", self.coll_table[i][j]);
        //                 row.push(Cell::from(&entry));
        //             }
        //             table_data.push(row);
        //         }
        //         let headers_ref: Vec<&str> = headers.iter().map(|s| s.as_str()).collect();
        //         let _table = Table::new(Style::Plain, table_data, Some(Headers::from(headers_ref)));
        //         println!("coll_table:\n{}", _table.tabulate());
        //     }

        //     // Look up bounds from coll_table.
        //     // For r values, we invert the definition of self.coll_table_r_values:
        //     //   [2 + self.r_constant * (i ** 2) for i in range(self.num_r_values - 1)] + [self.n]
        //     let i = (((r - 2) as f64) / self.r_constant as f64).sqrt() as usize;
        //     let i = i.min(self.coll_table_r_values.len() - 2);

        //     // for u values we similarly invert the definition: np.linspace(0, 1, num_u_values)
        //     let j = (u * (self.coll_table_u_values.len() - 1) as f64) as usize;

        //     t_lo = self.coll_table[i + 1][j + 1];
        //     t_hi = self.coll_table[i][j];
        //     t_hi = t_hi.min(self.n - r + 1);
        //     if t_lo == 1 && t_hi == 1 {
        //         t_lo = 1;
        //         t_hi = 2;
        //     }

        //     if PRINT {
        //         println!(
        //             "n = {}, self.r_constant={}, u={u:.3}, r={r} i={i} j={j} t_lo={t_lo}, t_hi={t_hi} (((r - 2) as f64) / self.r_constant as f64).sqrt()={:.3}",
        //             self.n,
        //             self.r_constant,
        //             (((r - 2) as f64) / self.r_constant as f64).sqrt()
        //         );
        //         println!(
        //             "lhs = {lhs:.3}, logn={:.1}, t_lo*logn={:.1}, log_factorial(n-r-t_lo)={:.1}, log_factorial(n-r-t_lo) + t_lo*logn={:.3}",
        //             self.logn,
        //             t_lo as f64 * self.logn,
        //             ln_factorial(self.n - r - t_lo),
        //             ln_factorial(self.n - r - t_lo) + (t_lo as f64 * self.logn)
        //         );
        //     }
        //     assert!(t_lo < t_hi);
        //     assert!(self.coll_table_r_values[i] <= r);
        //     assert!(r <= self.coll_table_r_values[i + 1]);
        //     assert!(self.coll_table_u_values[j] <= u);
        //     assert!(u <= self.coll_table_u_values[j + 1]);

        //     if t_lo < t_hi - 1 {
        //         assert!(lhs >= ln_factorial(self.n - r - t_lo - 1) + (t_lo as f64 * logn_minus_1));
        //         assert!(lhs < ln_factorial(self.n - r - t_hi) + (t_hi as f64 * self.logn));
        //     }
        // } else {
        //     // When building the table, we start with bounds that always hold.
        //     if r >= self.n {
        //         return 0;
        //     }
        //     t_lo = 0;
        //     t_hi = 1 + ((self.n - r) / self.o);
        // }

        // When building the table, we start with bounds that always hold.
        if r >= self.n {
            return 0;
        }
        t_lo = 0;
        t_hi = 1 + ((self.n - r) / self.o);

        // We maintain the invariant that P(l >= t_lo) >= u and P(l >= t_hi) < u
        while t_lo < t_hi - 1 {
            let t_mid = (t_lo + t_hi) / 2;

            let mut rhs: f64 = ln_gamma((self.n - r - (t_mid * self.o)) as f64 + 1.0);
            if self.g > 0 {
                for j in 0..self.o {
                    // Calculates b = ceil((n+g(t-1)-j)/g)
                    let num_dynamic_terms = (((self.n + (self.g as usize * (t_mid - 1)) - j) as f64) / self.g as f64).ceil();
                    rhs += num_dynamic_terms * (self.g as f64).ln();
                    rhs += ln_gamma((self.n + (self.g as usize * t_mid) - j) as f64 / self.g as f64);
                    rhs -= ln_gamma((self.n as f64 + (self.g as f64 * (t_mid as f64 - num_dynamic_terms)) - j as f64) / self.g as f64);
                } 
            } else if self.g < 0 {
                let unsigned_g = (-1 * self.g) as usize;
                for j in 0..self.o {
                    // Calculates b = ceil((n+|g|t-j)/|g|)
                    let num_dynamic_terms = (((self.n + (unsigned_g as usize * t_mid) - j) as f64) / unsigned_g as f64).ceil();
                    // println!("At point 1, {rhs}");
                    rhs -= num_dynamic_terms * (unsigned_g as f64).ln();
                    // println!("At point 2, {rhs}");
                    rhs -= ln_gamma((self.n + (unsigned_g as usize * (t_mid + 1)) - j) as f64 / unsigned_g as f64);
                    // println!("At point 3, {rhs}");
                    rhs += ln_gamma((self.n as f64 + (unsigned_g as f64 * (t_mid as f64 - num_dynamic_terms + 1.0)) - j as f64) / unsigned_g as f64);
                    // println!("At point 4, {rhs}");
                } 
            } else {
                // g = 0 case is much simpler.
                for j in 0..self.o {
                    rhs += t_mid as f64 * ((self.n - j) as f64).ln();
                } 
            }
            
            // println!("They are {lhs} and {rhs}.");
            if lhs < rhs {
                t_hi = t_mid;
            } else {
                t_lo = t_mid;
            }
        }

        // Return t_lo instead of t_hi (as in simulator_pp_multibatch) because the CDF here is written
        // in terms of p(l >= t) instead of p(l > t).
        t_lo
    }

    ///     init_array: A 2D length-q integer array of counts representing the initial configuration.
    ///     delta: A 2D q x q x 2 array representing the transition function.
    ///         Delta[i, j] gives contains the two output states.
    ///     null_transitions: A 2D q x q boolean array where entry [i, j] says if these states have a null interaction.
    ///     random_transitions: A 2D q x q x 2 array. Entry [i, j, 0] is the number of possible outputs if
    ///         transition [i, j] is random, otherwise it is 0. Entry [i, j, 1] gives the starting index to find
    ///         the outputs in the array random_outputs if it is random.
    ///     random_outputs: A ? x 2 array containing all (out1,out2) outputs of random transitions,
    ///         whose indexing information is contained in random_transitions.
    ///     transition_probabilities: A 1D length-? array containing all random transition probabilities,
    ///         whose indexing matches random_outputs.
    ///     seed (optional): An integer seed for the pseudorandom number generator.

    /// This is an easier to use constructor taking native Rust types instead of numpy arrays,
    /// but otherwise it works similarly to the `new` constructor.
    pub fn from_delta_random(
        init_config: Vec<usize>,
        random_transitions: ArrayD<usize>,
        random_outputs: Vec<Vec<State>>,
        transition_probabilities: Vec<f64>,
        _transition_order: String,
        gillespie: bool,
        seed: Option<u64>,
    ) -> Self {

        let random_transitions = random_transitions.clone();

        let config = init_config.clone();
        let n = config.iter().sum();
        let q = config.len() as State;

        // let v = random_transitions[[3,4,5]];
        // let w = [3,4,5];
        let o = random_transitions.shape().len() - 1;
        // TODO Techncially g can be negative, so shouldn't be usize. 
        let g = random_transitions.shape()[o] as isize - o as isize;
        let reaction_indices = all_reaction_indices(q, o);

        // random_depth is the maximum number of outputs for any randomized transition
        let random_depth = 1;

        // TODO uncomment this
        // for random_transitions_inner in random_transitions.axis_iter(Axis(o)) {
        //     random_depth = random_depth.max(random_transitions_inner[0]);
        // }

        // assert_eq!(
        //     random_outputs.len(),
        //     transition_probabilities.len(),
        //     "random_outputs and transition_probabilities length mismatch"
        // );

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
        let m = vec![0; random_depth];
        let silent = false;
        let do_gillespie = false; // this changes during run
        let gillespie_always = gillespie; // this never changes; if True we always do Gillespie steps

        // next three fields are only used with Gillespie steps;
        // they will be set accordingly if we switch to Gillespie
        // let propensities = vec![0.0; reactions.len()];
        // let enabled_reactions = vec![0; reactions.len()];
        // let num_enabled_reactions = 0;

        // below here we give meaningless default values to the other fields and rely on
        // set_n_parameters and get_enabled_reactions to set them to the correct values
        let logn = 0.0;
        let batch_threshold = 0;
        let gillespie_threshold = 0.0;
        let coll_table = vec![vec![0; 1]; 1];
        let coll_table_r_values = vec![0; 1];
        let coll_table_u_values = vec![0.0; 1];
        let r_constant = 0;

        let sim = SimulatorCRNMultiBatch {
            n,
            t,
            q,
            o,
            g,
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
            m,
            do_gillespie,
            silent,
            gillespie_threshold,
            coll_table,
            coll_table_r_values,
            coll_table_u_values,
            r_constant,
            gillespie_always,
            reaction_indices,
        };
        // sim.set_n_parameters();
        // sim.update_enabled_reactions();
        sim
    }
}
