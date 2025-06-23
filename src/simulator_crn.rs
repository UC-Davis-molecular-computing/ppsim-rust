use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::time::{Duration, Instant};
use std::vec;

use crate::flame;

use numpy::PyArrayMethods;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
// use ndarray::prelude::*;
use ndarray::{ArrayD, Axis};
use pyo3::types::PyNone;

use numpy::{PyReadonlyArray1};
use rand::rngs::SmallRng;
use rand::Rng;
use rand::SeedableRng;
#[allow(unused_imports)]
use statrs::distribution::{Geometric, Uniform};

use itertools::Itertools;
use statrs::function::factorial::binomial;

use crate::simulator_abstract::Simulator;

use crate::urn::Urn;
use crate::util::{ln_factorial, ln_gamma, multinomial_sample};

type State = usize;
type RateConstant = f64;
type StateList = Vec<State>;
type Reaction = (StateList, StateList, RateConstant);
type ProductsAndRateConstant = (StateList, RateConstant);
// A map from each state that appears to how many times that state appears in this set of reactants.

// We remember the CRN we started with, because we may need to recompute reaction 
// probabilities between batches if we change the count of K. This should also hopefully
// make it easier to interface with the Simulator class from both python and rust.
// This struct is named to emphasize that it stores the CRN *after* the uniform transformation
// is applied (so all reactions should have equal order and equal generativity). Rate constants
// are stored here *before* adjusting for the count of K.
pub struct UniformCRN {
    // Reaction order.
    pub o: usize,
    // Generativity.
    pub g: usize,
    // Number of species, including K and W.
    pub q: usize,
    // The specific States that represent K and W.
    pub k: State,
    pub w: State,
    // The CRN's reactions. If multiple reactions share the same reactants, they are stored in
    // the same Reaction object, for ease of iterating over reactions.
    pub reactions: Vec<CombinedReaction>,
}

// A struct combining all reactions with the same reactants into a single piece.
pub struct CombinedReaction {
    pub reactants: StateList,
    pub outputs: Vec<ProductsAndRateConstant>,
}

impl UniformCRN {
    // Make sure that a set of reactions is uniform and the reactions valid, and combine reactions 
    // that share the same set of reactants for easier iteration.
    fn verify_and_combine_reactions(reactions: Vec<Reaction>, k: State, w: State) -> UniformCRN {
        if reactions.len() == 0 {
            return UniformCRN {o: 0, g: 0, q: 0, k: 0, w: 0, reactions: Vec::new()}
        }
        let first_reaction = &reactions[0];
        let o = first_reaction.0.len();
        let g = first_reaction.1.len() - o;
        let mut all_species_seen: HashSet<State> = HashSet::from([k,w]);
        let mut highest_species_seen = k.max(w);
        let mut collected_reactions: HashMap<StateList, Vec<ProductsAndRateConstant>> = HashMap::new();
        for reaction in reactions {
            assert!(reaction.0.len() == o, "All reactions must have the same number of inputs");
            assert!(reaction.1.len() - reaction.0.len() == g, "All reactions must have the same number of outputs");
            for reactant in &reaction.0 {
                all_species_seen.insert(*reactant);
                highest_species_seen = highest_species_seen.max(*reactant);
            }
            for product in &reaction.1 {
                all_species_seen.insert(*product);
                highest_species_seen = highest_species_seen.max(*product);
            }
            collected_reactions.entry(reaction.0).or_default().push((reaction.1, reaction.2));
        }
        let q = highest_species_seen + 1;
        assert!(q == all_species_seen.len(), "Species must be indexed using contiguous integers starting from 0");
        let reactions_out: Vec<CombinedReaction> = collected_reactions
            .keys()
            .map(|x| CombinedReaction { reactants: x.to_vec(), outputs: collected_reactions[x].clone() })
            .collect();
        return UniformCRN { o: o, g: g, q: q, k: k, w: w, reactions: reactions_out }
    }
    // Build or rebuild random_transitions, random_outputs, and transition_probabilities.
    // We need to rebuild these tables because reaction propensities depend on the count of K,
    // which we may want to change throughout the execution.
    // Returns a tuple of these three objects in that order.
    fn construct_transition_arrays(&self, k_count: usize) -> (ArrayD<usize>, Vec<StateList>, Vec<f64>) {
        flame::start("construct_transition_arrays");
        let mut max_total_adjusted_rate_constant: f64 = 0.0;
        // Iterate through reactions, adjusting rate constants to account for how many K
        // are being added, and for symmetry that results from the scheduler having different
        // orders it can pick, so that the adjusted CRN keeps the original dynamics.
        for reaction in &self.reactions {
            let mut total_rate_constant = 0.0;
            let reactants = &reaction.reactants;
            let correction_factor = self.correction_factor(reactants, k_count);
            for output in &reaction.outputs {            
                let adjusted_rate_constant: f64 = output.1 * correction_factor;
                total_rate_constant += adjusted_rate_constant;
            }
            max_total_adjusted_rate_constant = max_total_adjusted_rate_constant.max(total_rate_constant);
        }
        // random_transitions has q+1 dimensions, the first q of which have length o,
        // and the last of which has length 2.
        let mut shape_vec = vec![self.q; self.o];
        shape_vec.push(2);
        let mut random_transitions = ArrayD::<usize>::zeros(shape_vec);
        let mut random_outputs: Vec<Vec<State>> = Vec::new();
        let mut random_probabilities: Vec<f64> = Vec::new();
        // Add any non-null reactions. Null reactions don't need any special handling.
        let mut cur_output_index = 0;
        for reaction in &self.reactions {
            
            // Add info from this reaction to all possible permutations of reactants.
            for output in &reaction.outputs {
                let probability = output.1 * self.correction_factor(&reaction.reactants, k_count) / max_total_adjusted_rate_constant;
                random_outputs.push(output.0.clone());
                random_probabilities.push(probability);
                for permutation in reaction.reactants.iter().permutations(self.o) {
                    let mut view = random_transitions.view_mut();
                    // This loop indexes into random_transitions.
                    for dimension in 0..self.o {
                        view = view.index_axis_move(Axis(0), *permutation[dimension]);    
                    }
                    // Make sure that this is one-dimensional.
                    let mut inner_view = view.into_dimensionality::<ndarray::Ix1>().unwrap();
                    // Increment the number of possible outputs for these reactants.
                    inner_view[0] += 1;
                    inner_view[1] = cur_output_index;
                    
                }
            }
            cur_output_index += reaction.outputs.len();
        }        
        assert_eq!(
            random_outputs.len(),
            random_probabilities.len(),
            "random_outputs and transition_probabilities length mismatch"
        );
        flame::end("construct_transition_arrays");
        return (random_transitions, random_outputs, random_probabilities);
    }

    fn correction_factor(&self, reactants: &Vec<State>, k_count: usize) -> f64 {
        let mut correction_factor = Self::symmetry_degree(reactants) as f64;
        let k_multiplicity = reactants.iter().filter(|&s| *s == self.k).count();
        // We've artificially sped up this reaction by |K| * (|K| - 1) * ...
        // This loop undoes that artificial speedup.
        for i in 0..k_multiplicity {
            correction_factor /= (k_count - i) as f64;
        }
        return correction_factor;
    }


    // Determine the degree of symmetry of a reaction, i.e., for any given ordering of its reactants,
    // the number of reorderings that are redundant. Obtained as the product of the factorial of
    // the count of each reactant.
    fn symmetry_degree(reactants: &Vec<State>) -> usize {
        let mut factor = 1;
        let mut frequencies: HashMap<usize, usize> = HashMap::new();
        for reactant in reactants {
            *frequencies.entry(*reactant).or_default() += 1;
        }
        for (_, frequency) in frequencies {
            for i in 2..frequency + 1 {
                factor *= i;
            }
        }
        return factor
    }
    // An iterator that iterates over all possible reactant vectors.
    // It might be more efficient to implement some custom iterator since really all this needs
    // to do is count up through all o-digit numbers in base q.
    // fn reactant_iterator(self) -> impl Iterator<Item = Vec<State>> {
    //     return itertools::repeat_n(0..self.q, self.o).multi_cartesian_product();
    // }
}

//TODO: consider using ndarrays instead of multi-dimensional vectors
// I think native Rust arrays won't work because their size needs to be known at compile time.
#[pyclass(extends = Simulator)]
pub struct SimulatorCRNMultiBatch {
    /// The CRN with a list of reactions, so we can recompute probabilities when the
    /// count of K is updated between batches.
    pub crn: UniformCRN,
    /// The population size (sum of values in urn.config).
    #[pyo3(get, set)]
    pub n_including_extra_species: usize,
    /// The population size of all species except k and w.
    #[pyo3(get, set)]
    pub n: usize,
    /// The current number of elapsed interaction steps that actually simulated something in 
    /// the original CRN, rather than being a "null reaction".
    #[pyo3(get, set)]
    pub t: usize,
    /// The current number of elapsed interaction steps in this CRN, including null reactions.
    #[pyo3(get, set)]
    pub t_including_nulls: usize,
    /// The total number of states (length of urn.config).
    pub q: usize,
    /// An (o + 1)-dimensional array. The first o dimensions represent reactants. After indexing through
    /// the first o dimensions, the last dimension always has size two, with elements (`num_outputs`, `first_idx`).
    /// `num_outputs` is the number of possible outputs if transition i,j --> ... is non-null,
    /// otherwise it is 0. `first_idx` gives the starting index to find
    /// the outputs in the array `self.random_outputs` if it is non-null.
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
    pub random_outputs: Vec<StateList>,
    /// An array containing all random transition probabilities,
    /// whose indexing matches random_outputs.
    /// May add up to less than 1 for a given reaction, in which case the remainder is assumed null.
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
    /// Struct which stores the result of hypergeometric sampling.
    array_sums: NDBatchResult,
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
    // The probability of each reaction. 
    // #[pyo3(get, set)]
    // // pub reaction_probabilities: Vec<f64>,
    // // The probability of a non-null interaction must be below this
    // // threshold to keep doing Gillespie steps.
    // gillespie_threshold: f64,
    // // Precomputed values to speed up the function sample_coll(r, u).
    // // This is a 2D array of size (`coll_table_r_values.len()`, `coll_table_u_values.len()`).
    // coll_table: Vec<Vec<usize>>,
    // // Values of r, giving one axis of coll_table.
    // coll_table_r_values: Vec<usize>,
    // // Values of u, giving the other axis of coll_table.
    // coll_table_u_values: Vec<f64>,
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
    ///         Entry [r, 0] is the number of possible outputs if transition on reactant set r is non-null, 
    ///         otherwise it is 0. Entry [r, 1] gives the starting index to find the outputs in the array random_outputs if it is non-null.
    ///     random_outputs: A ? x (o + g) array containing all outputs of random transitions,
    ///         whose indexing information is contained in random_transitions.
    ///     transition_probabilities: A 1D length-? array containing all random transition probabilities,
    ///         whose indexing matches random_outputs.
    ///     seed (optional): An integer seed for the pseudorandom number generator.
    #[new]
    #[pyo3(signature = (init_config, _delta, _random_transitions, _random_outputs, _transition_probabilities, _transition_order, _gillespie, seed, reactions, k, w))]
    pub fn new(
        init_config: PyReadonlyArray1<State>,
        _delta: Py<PyNone>,
        _random_transitions: Py<PyNone>,
        _random_outputs: Py<PyNone>,
        _transition_probabilities: Py<PyNone>,
        _transition_order: Py<PyNone>,
        _gillespie: Py<PyNone>,
        seed: Option<u64>,
        reactions: Vec<Reaction>,
        k: State,
        w: State,
    ) -> (Self, Simulator) {
        let crn = UniformCRN::verify_and_combine_reactions(reactions, k, w);
        let init_config = init_config.to_vec().unwrap();


        let config = init_config.clone();
        let n = config.iter().sum();
        let n_including_extra_species = n;
        let q = config.len() as State;

        // random_depth is the maximum number of outputs for any randomized transition
        let random_depth = crn.reactions
            .iter()
            .map(|x| x.outputs.len())
            .fold(0, |acc, x| acc.max(x));
        
        let t = 0;
        let t_including_nulls = 0;
        let rng = if let Some(s) = seed {
            SmallRng::seed_from_u64(s)
        } else {
            SmallRng::from_entropy()
        };

        let urn = Urn::new(config.clone(), seed);
        let updated_counts = Urn::new(vec![0; q], seed);
        let array_sums = make_batch_result(crn.o, q);
        let row = vec![0; q];
        // The +1 here is to sample how many reactions are null.
        let m = vec![0; random_depth + 1];
        let silent = false;
        let do_gillespie = false; // this changes during run

        // next three fields are only used with Gillespie steps;
        // they will be set accordingly if we switch to Gillespie
        // let propensities = vec![0.0; reactions.len()];
        // let enabled_reactions = vec![0; reactions.len()];
        // let num_enabled_reactions = 0;

        // below here we give meaningless default values to the other fields and rely on
        // set_n_parameters and get_enabled_reactions to set them to the correct values
        // let gillespie_threshold = 0.0;
        // let coll_table = vec![vec![0; 1]; 1];
        // let coll_table_r_values = vec![0; 1];
        // let coll_table_u_values = vec![0.0; 1];

        // The following will be initialized during reset_k_count() below.
        let random_transitions = ArrayD::<usize>::zeros(Vec::new());
        let random_outputs = Vec::new();
        let transition_probabilities= Vec::new();
        
        let mut simulator = SimulatorCRNMultiBatch {
            crn,
            n,
            n_including_extra_species,
            t,
            t_including_nulls,
            q,
            random_transitions,
            random_outputs,
            transition_probabilities,
            random_depth,
            rng,
            urn,
            updated_counts,
            array_sums,
            row,
            m,
            do_gillespie,
            silent,
            // gillespie_threshold,
            // coll_table,
            // coll_table_r_values,
            // coll_table_u_values,
        };
        
        simulator.reset_k_count();
        (simulator, Simulator::default())
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
            if self.silent {
                return Ok(());
            } else if self.do_gillespie {
                flame::start("gillespie step");
                self.gillespie_step(t_max);
                flame::end("gillespie step");
            } else {
                self.batch_step(t_max);
                self.reset_k_count();
                self.recycle_waste();
            }
            // TODO this should be set more generally
            if self.n_including_extra_species == 0 {
                self.silent = true;
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
                self.batch_step(0);
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
        self.n = config.iter().sum();
        self.reset_k_count();
        self.n_including_extra_species = self.urn.size;
        self.t = t;
        self.t_including_nulls = t;
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
        loop {
            for _ in 0..self.crn.o {
                let sample = self.rng.gen_range(0..pop_size);
                if sample < num_seen {
                    return idx;
                }
                pop_size -= 1;
            }
            idx += 1;
            pop_size += self.crn.o + self.crn.g;
            num_seen += self.crn.o + self.crn.g;
        }
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

// A struct for holding the q^o results of multidimensional hypergeometric sampling from an urn.
// Recursive because of unknown nesting depth. To efficiently sample the k reactions of a batch,
// we first need to sample all the codimension-1 sums, that is, how many reactions have each species
// as its first reactant. Then, for each of those, we need to sample the codimension-2 sums, e.g.,
// how many reactions with A as their first reactant have each reactant as their second. And so on,
// recursively down to o dimensions. 
// When sampling, the each struct will store its codimension-1 sums in values. Then, for each
// of those values, it will recursively sample that many subreactions into its subresults.
// This could be implemented in a more memory-efficient way where the subresult is just a single
// NDBatchResult instead of a Vec, and only one result is stored at a time during the recursion.
// I'm not sure if it's a better implementation; it's definitely better if we expect q^o to
// potentially be large enough that we couldn't store it all at once, though at that point
// it's unlikely that this is the right algorithm for simulation.
struct NDBatchResult {
    // Tells you what level of recursion you're on. 
    // For the top level, dimensions = o (number of reactants). 
    // For the bottom level, dimensions = 1.
    dimensions: usize, 
    q: usize,
    o: usize,
    // For iterating over results
    curr_species: State,
    // Initialized to 0, then sampled into via urn.sample_vector().
    pub counts: Vec<usize>,
    // If dimensions > 1, this is a vector of subresults. If dimensions = 1, it is empty.
    pub subresults: Option<Vec<NDBatchResult>>,
}

impl NDBatchResult {
    fn populate_empty(&mut self) {
        if self.dimensions > 1 {
            for _ in 0..self.q {
                let mut subresult = NDBatchResult{
                    dimensions: self.dimensions - 1, 
                    q: self.q, 
                    o: self.o,
                    curr_species: 0,
                    counts: vec![0; self.q], 
                    subresults: {if self.dimensions == 2 {None} else {Some(Vec::with_capacity(self.q))}}};
                subresult.populate_empty();
                self.subresults.as_mut().unwrap().push(subresult);
            }
        }
    }
    fn sample_batch_result(&mut self, reactions: usize, urn: &mut Urn) {
        urn.sample_vector(reactions, &mut self.counts).unwrap();
        self.curr_species = 0;
        if self.dimensions > 1 {
            for i in 0..self.q {
                let subresult = self.subresults.as_mut().unwrap();
                subresult[i].sample_batch_result(self.counts[i], urn);
            }
        }
    }
    // TODO docstring here. Also this should probably implement an iterable trait
    fn get_next(&mut self) -> (Vec<State>, usize, bool) {
        assert!(self.curr_species < self.q, "NDBatchResult iterated past final species");
        let mut done = false;
        if self.dimensions == 1 {
            let mut curr_reaction = vec![0;self.o];
            curr_reaction[self.o - self.dimensions] = self.curr_species;
            self.curr_species += 1;
            return (curr_reaction, self.counts[self.curr_species - 1], self.curr_species == self.q);
        } else {
            let curr_subresult = &mut self.subresults.as_mut().unwrap()[self.curr_species];
            let (mut curr_reaction, count, subresult_done) = curr_subresult.get_next();
            curr_reaction[self.o - self.dimensions] = self.curr_species;
            if subresult_done {
                self.curr_species += 1;
                if self.curr_species == self.q {
                    done = true;
                }
            }
            return (curr_reaction, count, done);
        }
    }
}

fn make_batch_result(dimensions: usize, length: usize) -> NDBatchResult {
    let mut result = NDBatchResult {
        dimensions: dimensions, 
        q: length, 
        o: dimensions,
        curr_species: 0,
        counts: vec![0;length], 
        subresults: Some(Vec::with_capacity(length))};
    result.populate_empty();
    result
}


// const CAP_BATCH_THRESHOLD: bool = true;

impl SimulatorCRNMultiBatch {
    fn batch_step(&mut self, t_max: usize) -> () {
        self.updated_counts.reset();
        let uniform = Uniform::standard();
        assert_eq!(self.n_including_extra_species, self.urn.size, "Self.n should match self.urn.size.");

        flame::start("sample batch");

        let u = self.rng.sample(uniform);

        let has_bounds = false;
        flame::start("sample_coll");
        let l = self.sample_coll(0, u, has_bounds);
        flame::end("sample_coll");

        let mut rxns_before_coll = l;
        let initial_t_including_nulls = self.t_including_nulls;


        assert!(l > 0, "sample_coll must return at least 1 for batching");

        // If the sampled collision happens after t_max, then include delayed agents up until t_max
        // and do not perform the collision.
        // TODO: There's some awkwardness here: if there *might* be enough collisions to go past
        // t_max, then we need to stop early. But this means that, for instance, if there are
        // 100 reactions left until t_max, and our typical batch size is 200, we'll just keep
        // running batches of size equal to t_max - t until getting t = t_max, since t only
        // actually increases in response to "real" reactions. So if lots of null reactions happen,
        // this will be super inefficient. Lots of possible options - the simplest might be,
        // just run Gillespie during the last little bit of each batch.
        let mut do_collision = true;
        if t_max > 0 && self.t + l >= t_max {
            assert!(t_max > self.t);
            rxns_before_coll = t_max - self.t;
            do_collision = false;
        }

        flame::end("sample batch");

        flame::start("process batch");

        // The idea here is to iterate through random_transitions and array_sums together; they should
        // both be indexed by q^o-tuples when iterated through this way, and the iteration order should
        // be lexicographic for both of them.
        self.array_sums.sample_batch_result(rxns_before_coll, &mut self.urn);
        let mut done = false;
        let reactions_iter = self.random_transitions.lanes(Axis(self.crn.o)).into_iter();
        for random_transition in reactions_iter {
            assert!(!done, "self.array_sums finished iterating before self.random_transitions");
            let next_array_sum = self.array_sums.get_next();
            // TODO maybe add an assert check that the two structures are iterated through
            // in the same order, i.e. reactants match
            let (reactants, quantity) = (next_array_sum.0, next_array_sum.1);
            done = next_array_sum.2;
            // println!("Handling {:?} reactions of type {:?}.", quantity, reactants);
            // println!("update config at this point is {:?}.", self.updated_counts.config);
            let initial_updated_counts_size = self.updated_counts.size;
            if quantity == 0 {continue}
            let (num_outputs, first_idx) = (random_transition[0], random_transition[1]);
            // TODO and WARNING: this code is more or less copy-paste with the collision sampling code.
            // They do the same thing. But it's apparently very annoying to reafactor this into a
            // helper method in rust because of the immutable borrow of self above.
            if num_outputs == 0 {
                // Null reaction. Move the reactants from self.urn to self.updated_counts (for collision sampling), and add W.
                for reactant in reactants {
                    self.updated_counts.add_to_entry(reactant, quantity as i64);
                }
                self.updated_counts.add_to_entry(self.crn.w, (quantity * self.crn.g) as i64);
            } else {
                // We'll add this for now, then subtract off the probabilistically null reactions later.
                self.t += quantity;
                let mut probabilities = self.transition_probabilities[first_idx..first_idx + num_outputs].to_vec();
                let probability_sum: f64 = probabilities.iter().sum();
                if probability_sum < 1.0 {
                    probabilities.push(1.0 - probability_sum);
                }
                flame::start("multinomial sample");
                multinomial_sample(quantity, &probabilities, &mut self.m[0..probabilities.len()], &mut self.rng);
                flame::end("multinomial sample");
                assert_eq!(
                    self.m[0..probabilities.len()].iter().sum::<usize>(),
                    quantity,
                    "sample sum mismatch"
                );
                for c in 0..num_outputs {
                    let idx = first_idx + c;
                    let outputs = &self.random_outputs[idx];
                    for output in outputs {
                        self.updated_counts.add_to_entry(output.clone(), self.m[c] as i64);
                    }
                }
                // Add any W produced by null reactions.
                if probability_sum < 1.0 {
                    let null_count = self.m[num_outputs];
                    self.updated_counts.add_to_entry(self.crn.w, null_count as i64);
                    for reactant in reactants {
                        self.updated_counts.add_to_entry(reactant, null_count as i64);
                    }
                    self.t -= null_count;
                }
            }
            // println!("update config afterward is {:?}.", self.updated_counts.config);
            // println!("Size changed by {:?}, compared to quantity being {:?}.", self.updated_counts.size - initial_updated_counts_size, quantity)
            assert_eq!(quantity * (self.crn.o + self.crn.g), self.updated_counts.size - initial_updated_counts_size, 
                "Mismatch between how many elements were added to updated_counts.")
        }
        assert!(done, "self.random_transitions finished iterating before self.array_sums");

        assert_eq!((self.crn.g + self.crn.o) * rxns_before_coll, self.updated_counts.size, "Total number of molecules added is not consistent");
        
        flame::end("process batch");
        flame::start("sample collision");
        // We need to sample a collision. It could involve as few as 1 already-used molecule,
        // or as many as o. So we need to decide how many are involved.
        // TODO: I'm going to use u128 here because I'm pretty worried about fitting things 
        // into anything smaller. In fact I'm a little worried about u128; on population size
        // 10^12 which is about 2^40, if we have 4 reactants, then the denominator for the
        // relevant probability distribution will be too large to store. 
        let mut num_resampled = 0;
        if do_collision {
            let updated_counts_before_collision = self.updated_counts.size;
            let mut collision_count_num_ways: Vec<u128> = Vec::with_capacity(self.crn.o);
            let num_new_molecules = (self.crn.o + self.crn.g) * rxns_before_coll;
            let num_old_molecules = self.n_including_extra_species - (self.crn.o * rxns_before_coll);
            for num_collisions in 1..self.crn.o + 1 {
                collision_count_num_ways.push(
                    (num_old_molecules as u128).pow((self.crn.o - num_collisions).try_into().unwrap()) 
                    * (num_new_molecules as u128).pow(num_collisions.try_into().unwrap()) 
                    * binomial(self.crn.o as u64, num_collisions as u64) as u128);
            }
            let total_ways_with_at_least_one_collision: u128 = collision_count_num_ways.iter().sum();
            let u2 = self.rng.sample(uniform);
            let mut num_colliding_molecules = 0;
            let mut total_ways_so_far = 0;
            for i in 0..self.crn.o {
                total_ways_so_far += collision_count_num_ways[i];
                if u2 < (total_ways_so_far as f64) / (total_ways_with_at_least_one_collision as f64) {
                    num_colliding_molecules = i + 1;
                    break;
                }
            }
            assert!(num_colliding_molecules > 0, "Failed to sample collision size");
            let mut collision: Vec<usize> = Vec::with_capacity(self.crn.o);
            for _ in 0..num_colliding_molecules {
                collision.push(self.updated_counts.sample_one().unwrap());
                num_resampled += 1;
            }
            for _ in num_colliding_molecules..self.crn.o {
                collision.push(self.urn.sample_one().unwrap());
            }
            // Index into random_probabilities to sample what the collision will do.
            let mut view = self.random_transitions.view();
            for dimension in 0..self.crn.o {
                view = view.index_axis_move(Axis(0), collision[dimension]);
            }
            // Verify that the view is now a 1-dimensional subarray of random_probabilities,
            // which should just have two elements in it (number of random outputs and starting index)
            assert_eq!(view.ndim(), 1, "Was not left with 1-dimensional vector after indexing collision");
            assert_eq!(view.len(), 2, "Indexing collision did not leave two-element subarray");

            let (num_outputs, first_idx) = (view[0], view[1]);
            // TODO: this code is heavily copy-pasted. See other TODO comment above.
            if num_outputs == 0 {
                // Null reaction. 
                for reactant in collision {
                    self.updated_counts.add_to_entry(reactant, 1);
                }
                self.updated_counts.add_to_entry(self.crn.w, (self.crn.g) as i64);
            } else {
                self.t += 1;
                let mut probabilities = self.transition_probabilities[first_idx..first_idx + num_outputs].to_vec();
                let probability_sum: f64 = probabilities.iter().sum();
                if probability_sum < 1.0 {
                    probabilities.push(1.0 - probability_sum);
                }
                flame::start("multinomial sample");
                multinomial_sample(1, &probabilities, &mut self.m[0..probabilities.len()], &mut self.rng);
                flame::end("multinomial sample");
                assert_eq!(
                    self.m[0..probabilities.len()].iter().sum::<usize>(),
                    1,
                    "sample sum mismatch"
                );
                for c in 0..num_outputs {
                    let idx = first_idx + c;
                    let outputs = &self.random_outputs[idx];
                    for output in outputs {
                        self.updated_counts.add_to_entry(output.clone(), self.m[c] as i64);
                    }
                }
                // Add W if the collision was a probabilistic null reaction.
                if probability_sum < 1.0 {
                    let null_count = self.m[num_outputs];
                    self.updated_counts.add_to_entry(self.crn.w, null_count as i64);
                    for reactant in collision {
                        self.updated_counts.add_to_entry(reactant, null_count as i64);
                    }
                    self.t -= null_count;
                }
            }
            assert_eq!(self.updated_counts.size - updated_counts_before_collision, self.crn.o + self.crn.g - num_resampled,
                "Collision failed to add exactly one thing to updated_counts");
            self.t_including_nulls += 1;
        }
        flame::end("sample collision");
        
        self.t_including_nulls += rxns_before_coll;
        
        self.urn.add_vector(&self.updated_counts.config);
        self.urn.sort();
        // Check that we added the right number of things to the urn.
        assert_eq!(self.urn.size - self.n_including_extra_species, (self.t_including_nulls - initial_t_including_nulls) * self.crn.g,
            "Inconsistency between number of reactions simulated and population size change.");
        self.n_including_extra_species = self.urn.size;
        self.n = self.n_including_extra_species - self.urn.config[self.crn.k] - self.urn.config[self.crn.w];

        //self.update_enabled_reactions();
    }

    /// Do multinomial sampling.
    /// TODO better docstring
    // fn do_multinomial_sampling(&mut self, quantity: usize, first_idx: usize, num_outputs: usize) {
    //     if num_outputs == 0 {
    //         // Null reaction. 
    //         self.updated_counts.add_to_entry(self.crn.w, (quantity * self.crn.g) as i64);
    //     } else {
    //         let mut probabilities = self.transition_probabilities[first_idx..first_idx + num_outputs].to_vec();
    //         let probability_sum: f64 = probabilities.iter().sum();
    //         if probability_sum < 1.0 {
    //             probabilities.push(1.0 - probability_sum);
    //         }
    //         flame::start("multinomial sample");
    //         multinomial_sample(quantity, &probabilities, &mut self.m[0..probabilities.len()], &mut self.rng);
    //         flame::end("multinomial sample");
    //         assert_eq!(
    //             self.m[0..probabilities.len()].iter().sum::<usize>(),
    //             quantity,
    //             "sample sum mismatch"
    //         );
    //         for c in 0..num_outputs {
    //             let idx = first_idx + c;
    //             let outputs = &self.random_outputs[idx];
    //             for output in outputs {
    //                 self.updated_counts.add_to_entry(output.clone(), self.m[c] as i64);
    //             }
    //         }
    //         // Add any W produced by null reactions.
    //         if probability_sum < 1.0 {
    //             self.updated_counts.add_to_entry(self.crn.w, self.m[num_outputs] as i64);
    //         }
    //     }
    // }

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

    /// Update the count of K in preparation for the next batch. 
    /// We will try to choose a value for the count of K that maximizes the expected amount
    /// of progress we make in simulating the original CRN.
    fn reset_k_count(&mut self) {
        // TODO: do something more complicated than this to actually be efficient.
        // This is just making sure that half of everything is always k.
        let delta_k = self.n as i64 - self.urn.config[self.crn.k] as i64;
        assert!(self.n_including_extra_species as i64 + delta_k >= 0);
        self.n_including_extra_species = (self.n_including_extra_species as i64 + delta_k) as usize;
        self.urn.add_to_entry(self.crn.k, delta_k);
        (self.random_transitions, self.random_outputs, self.transition_probabilities) = self.crn.construct_transition_arrays(self.n);
    }

    /// Get rid of W from self.urn. 
    /// It is recycled to a better place.
    fn recycle_waste(&mut self) {
        let delta_w = -1 * self.urn.config[self.crn.w] as i64;
        assert!(self.n_including_extra_species as i64 + delta_w >= 0);
        self.n_including_extra_species = (self.n_including_extra_species as i64 + delta_w) as usize;
        self.urn.add_to_entry(self.crn.w, delta_w);
    }

    /// Sample a collision event from the urn
    /// Returns a sample l ~ coll(n, r, o, g) from the collision length distribution.
    /// See TODO: add reference to paper once it's on arxiv.
    /// The distribution gives the number of reactions that will occur before a collision.
    /// Inversion sampling with binary search is used, based on the formula
    ///     P(l >= t) = (n-r)! / (n-r-t*o)! * prod_{j=0}^{o-1} [(n-g-j)!(g) / (n+g(t-1)-j)!(g)].
    /// !(g) denotes a multifactorial: n!(g) = n * (n - g) * (n - 2g) * ..., until these terms become nonpositive. 
    /// This is the formula when g > 0; when g = 0 or g < 0, the formulas are slightly different 
    /// (see the full formula for coll(n,r,o,g) in the paper), but the method is the same:
    /// We sample a uniform random variable u, and find the value t such that
    ///     P(l >= t) < U < P(l >= t - 1).
    /// Taking logarithms and using the ln_gamma function, this required formula becomes
    ///     P(l >= t) < U
    ///       <-->
    ///     ln_gamma(n-r+1) - ln_gamma(n-r-t*o+1) + sum_{j=0}^{o-1} [log((n-g-j)!(g)) - log((n+g(t-1)-j)!(g))] < log(U).
    /// which can be rewritten by using the fact that gamma(x) = (x - 1) * gamma(x-1) even for non-integer x,
    /// by factoring out a factor of g from every term in the multifactorial. 
    /// To this end, if we let a and b denote the number of terms in these multifactorial products,
    /// that is, let a = ceil((n-g-j)/g) and b = ceil((n+g(t-1)-j)/g),
    ///     ln_gamma(n-r+1) - ln_gamma(n-r-t*o+1) + sum_{j=0}^{o-1} [log(g^a * gamma((n-j)/g) / gamma((n-ag-j)/g)) - log(g^b * gamma((n+gt-j)/g) / gamma((n+g(t-b)-j)/g))] < log(U).
    ///     ln_gamma(n-r+1) - ln_gamma(n-r-t*o+1) + sum_{j=0}^{o-1} [a*log(g) + ln_gamma((n-j)/g) - ln_gamma((n-ag-j)/g) - b*log(g) - ln_gamma((n+gt-j)/g) + ln_gamma((n+g(t-b)-j)/g)] < log(U).
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
        // If every agent counts as a collision, the next reaction is a collision.
        assert!(r <= self.n_including_extra_species);
        if r == self.n_including_extra_species {
            return 0;
        }
        let mut t_lo: usize;
        let mut t_hi: usize;

        let logu = u.ln();
        let diff = self.n_including_extra_species + 1 - r;
        let ln_gamma_diff = ln_factorial(diff - 1);

        // lhs tracks all of the terms that don't include t, i.e., those that we don't need to
        // update each iteration of binary search.
        let mut lhs = ln_gamma_diff - logu;
        
        if self.crn.g > 0 {
            for j in 0..self.crn.o {
                // Calculates a = ceil((n-g-j)/g). This is the number of terms in the expansion of
                // a multifactorial. For example, 11!^(3) (the third multifactorial of 11) is
                // 11 * 8 * 5 * 2 so there are 4 terms in it. The way we calculate the log of a
                // multifactorial is to "factor out" the amount each term decreases by (in this example 3,
                // in general it will always equal g for the multifactorials we care about) from
                // every term (whether or not they're divisible by it), then rewrite it using gamma.
                // In this example, 11 * 8 * 5 * 2 = 3^4 * (11 / 3) * (8 / 3) * (5 / 3) * (2 / 3).
                // So, log(11!^(3)) = 4*log(3) + log((11 / 3) * (8 / 3) * (5 / 3) * (2 / 3))
                // = 4*log(3) * log(Gamma(14/3) / Gamma(2/3)) [because Gamma(x) = (x-1)*Gamma(x-1)]
                // = 4*log(3) + lgamma(14/3) - lgamma(2/3).
                // These three terms are the three terms that are added and subtracted from lhs, to
                // account for the term log((n-g-j)!(g)).
                let num_static_terms: f64 = (((self.n_including_extra_species - j) as f64 - self.crn.g as f64) / self.crn.g as f64).ceil();
                lhs += num_static_terms * (self.crn.g as f64).ln();
                lhs += ln_gamma((self.n_including_extra_species - j) as f64 / self.crn.g as f64);
                lhs -= ln_gamma((self.n_including_extra_species as f64 - (num_static_terms * self.crn.g as f64) - j as f64) / self.crn.g as f64);
            }
        } else {
            // Nothing to do here. There are no other static terms in the g = 0 case.
        }

        // TODO: it might be worth adding some code to jump-start the search with precomputed values,
        // as can be done in the population protocols case.
        // For now, we start with bounds that always hold.
        
        t_lo = 0;
        t_hi = 1 + ((self.n_including_extra_species - r) / self.crn.o);

        // We maintain the invariant that P(l >= t_lo) >= u and P(l >= t_hi) < u
        while t_lo < t_hi - 1 {
            let t_mid = (t_lo + t_hi) / 2;
            // rhs tracks all of the terms that include t, i.e., those that we need to
            // update each iteration of binary search.
            let mut rhs: f64 = ln_gamma((self.n_including_extra_species - r - (t_mid * self.crn.o)) as f64 + 1.0);
            if self.crn.g > 0 {
                for j in 0..self.crn.o {
                    // Calculates b = ceil((n+g(t-1)-j)/g).
                    // See the comment in the loop above where num_static_terms is defined for an explanation.
                    // This is the same thing, for the term log((n+g(t-1)-j)!(g)).
                    let num_dynamic_terms = (((self.n_including_extra_species + (self.crn.g as usize * (t_mid - 1)) - j) as f64) / self.crn.g as f64).ceil();
                    rhs += num_dynamic_terms * (self.crn.g as f64).ln();
                    rhs += ln_gamma((self.n_including_extra_species + (self.crn.g as usize * t_mid) - j) as f64 / self.crn.g as f64);
                    rhs -= ln_gamma((self.n_including_extra_species as f64 + (self.crn.g as f64 * (t_mid as f64 - num_dynamic_terms)) - j as f64) / self.crn.g as f64);
                } 
            } else {
                // g = 0 case is much simpler; there's no multifactorial, as it's analogous
                // to the population protocols case. 
                for j in 0..self.crn.o {
                    rhs += t_mid as f64 * ((self.n_including_extra_species - j) as f64).ln();
                } 
            }
            
            if lhs < rhs {
                t_hi = t_mid;
            } else {
                t_lo = t_mid;
            }
        }

        // Return t_lo instead of t_hi (which simulator_pp_multibatch returns) because the CDF here
        // is written in terms of p(l >= t) instead of p(l > t).
        t_lo
    }

}
