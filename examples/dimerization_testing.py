import math
import numpy as np
import random
from tqdm import tqdm
from matplotlib import pyplot as plt
import time
import rebop as rb
from scipy import stats
import json
from collections import defaultdict

import importlib.util
from pathlib import Path
import sys
if False:
    # Path to your renamed .pyd file
    # custom_pyd_path = Path("C:/Dropbox/git/ppsim-rust/python/ppsim/ppsim_rust/ppsim_rust.cp312-win_amd64_rebop.pyd")
    custom_pyd_path = Path("C:/Dropbox/git/ppsim-rust/python/ppsim/ppsim_rust/ppsim_rust.cp312-win_amd64_f128.pyd")

    # Define a custom finder and loader for .pyd files
    class CustomPydFinder:
        @classmethod
        def find_spec(cls, fullname, path=None, target=None):
            # Only handle the specific module we want to redirect
            if fullname == "ppsim.ppsim_rust.ppsim_rust":
                return importlib.util.spec_from_file_location(fullname, str(custom_pyd_path))
            return None

    # Register our custom finder at the beginning of the meta_path
    sys.meta_path.insert(0, CustomPydFinder)

import ppsim as pp

def measure_time(fn, trials=1) -> float:
    """
    Measure the time taken by a function over a number of trials.
    """
    start_time = time.perf_counter()
    for _ in range(trials):
        fn()
    end_time = time.perf_counter()
    return (end_time - start_time) / trials

def write_running_time_results(fn: str, times: list[float], ns: list[int]):
    results = list(zip(ns, times))
    with open(fn, 'w') as f:
        json.dump(results, f, indent=4)

def create_rebop_running_time_data(fn: str, min_pop_exponent: int, max_pop_exponent: int, end_time: float):
    num_trials = 1
    rebop_times = []
    ns_rebop = []
    seed = 1

    print('creating rebop running time data for dimerization')
    for pop_exponent in range(min_pop_exponent, max_pop_exponent + 1):
        print(f'n = 10^{pop_exponent}')
        
        crn = rb.Gillespie()
        crn.add_reaction(0.1 ** pop_exponent, ['M', 'M'], ['D'])
        crn.add_reaction(1, ['D'], ['M', 'M'])

        predator_fraction = 0.5
        n = int(10 ** pop_exponent)
        
        a_init = int(n * (1 - predator_fraction))
        b_init = n - a_init
        rebop_inits = {"A": a_init, "B": b_init}

        def run_rebop():
            crn.run(rebop_inits, end_time, 1, rng=seed)
        
        if pop_exponent == min_pop_exponent:
            # for some reason the first time it runs, rebop takes a long time
            run_rebop()
            run_rebop()
        print('rebop')
        rebop_times.append(measure_time(run_rebop, num_trials))
        ns_rebop.append(n)
        write_running_time_results(fn, rebop_times, ns_rebop)

def create_ppsim_running_time_data(fn: str, min_pop_exponent: int, max_pop_exponent: int, end_time: float):
    num_trials = 1
    ppsim_times = []
    ns_ppsim = []
    seed = 1
    m,d = pp.species('M D')
    rxns = [ (m+m | d).k(1) ]

    print('creating ppsim running time data for dimerization')
    # for pop_exponent_increment in tqdm(range(num_ns)):
    for pop_exponent in range(min_pop_exponent, max_pop_exponent + 1):
        print(f'n = 10^{pop_exponent}')
        
        predator_fraction = 0.5
        n = int(10 ** pop_exponent)
        a_init = int(n * (1 - predator_fraction))
        b_init = n - a_init
        ppsim_inits = {a: a_init, b: b_init}
        sim = pp.Simulation(ppsim_inits, rxns, simulator_method="crn", continuous_time=True, seed=seed)
        
        def run_ppsim():
            sim.run(end_time, 0.1)
        
        if pop_exponent == min_pop_exponent:
            run_ppsim()
        ppsim_times.append(measure_time(run_ppsim, num_trials))
        ns_ppsim.append(n)
        write_running_time_results(fn, ppsim_times, ns_ppsim)

def read_running_time_results(fn: str) -> tuple[list[int], list[float]]:
    with open(fn, 'r') as f:
        data = json.load(f)
    ns = [item[0] for item in data]
    times = [item[1] for item in data]
    return ns, times

def plot_results(fn_rebop_data: str, fn_ppsim_data_f64: str, fn_ppsim_data_f128: str, fn_out: str):
    # figsize = (6,4)
    figsize = (5,4)
    _, ax = plt.subplots(figsize = figsize)
    import matplotlib
    # matplotlib.rcParams.update({'font.size': 14}) # default font is too small for paper figures
    # matplotlib.rcParams['mathtext.fontset'] = 'cm' # use Computer Modern font for LaTeX
    rebop_ns, rebop_times = read_running_time_results(fn_rebop_data)
    ppsim_ns_f64, ppsim_times_f64 = read_running_time_results(fn_ppsim_data_f64)
    ppsim_ns_f128, ppsim_times_f128 = read_running_time_results(fn_ppsim_data_f128)
    ax.loglog(ppsim_ns_f64, ppsim_times_f64, label="batching f64 run time", marker="o")
    ax.loglog(ppsim_ns_f128, ppsim_times_f128, label="batching f128 run time", marker="o")
    ax.loglog(rebop_ns, rebop_times, label="rebop run time", marker="o")
    ax.set_xlabel(f'Initial molecular count')
    ax.set_ylabel(f'Run time (s)')
    ax.set_xticks([10**i for i in range(3, 15)])
    ax.set_ylim(bottom=None, top=10**5)
    ax.legend(loc='upper left')
    
    plt.savefig(fn_out, bbox_inches='tight')
    return

def fn_count_samples(alg: str, pop_exponent: int, trials_exponent: int, species: str, final_time: float) -> str:
    return f'data/dimer_{species}-counts_time{final_time}_n1e{pop_exponent}_trials1e{trials_exponent}_{alg}.json'

def write_rebop_count_samples(crn: rb.Gillespie, inits: dict[str, int], 
                              pop_exponent: int, trials_exponent: int, species_name: str, final_time: float) -> None:
    fn = fn_count_samples('rebop', pop_exponent, trials_exponent, species_name, final_time)
    print(f'collecting rebop data with n = 10^{pop_exponent} for {trials_exponent} trials')
    print(f'writing to {fn}')
    counts = defaultdict(int)
    for _ in tqdm(range(10**trials_exponent)):
        while True:
            try:
                results_rebop = crn.run(inits, final_time, 1)
                count = int(results_rebop[species_name][-1])
                counts[count] += 1
                break
            except IndexError:
                pass
                print("Index error caught and ignored. Rebop distribution may be slightly off.")
    counts = sort_dict_by_key(counts)
    with open(fn, 'w') as f:
        json.dump(counts, f, indent=4)

def sort_dict_by_key(d: dict) -> dict:
    return dict(sorted(d.items(), key=lambda item: item[0]))

def write_ppsim_count_samples(sim: pp.Simulation, 
                              pop_exponent: int, trials_exponent: int, species_name: str, final_time: float) -> None:
    fn = fn_count_samples('ppsim', pop_exponent, trials_exponent, species_name, final_time)
    print(f'collecting ppsim data with n = 10^{pop_exponent} for {trials_exponent} trials')
    print(f'writing to {fn}')
    trials = 10**trials_exponent
    results_batching = sim.sample_future_configuration(final_time, num_samples=trials)
    count_list: list[int] = results_batching[species_name].squeeze().tolist() # type: ignore
    counts = defaultdict(int)
    for count in count_list:
        counts[count] += 1
    
    counts = sort_dict_by_key(counts)
    with open(fn, 'w') as f:
        json.dump(counts, f, indent=4)

def read_count_samples(fn: str) -> list[int]:
    """
    Read the count samples from a JSON file.
    """
    with open(fn, 'r') as f:
        counts = json.load(f)
    count_list = []
    for count, num_samples_with_count in counts.items():
        count_list.extend([int(count)] * num_samples_with_count)
    return count_list

def plot_rebop_ppsim_histogram(pop_exponent: int, trials_exponent: int, species_name: str, final_time: float):
    rebop_fn = fn_count_samples('rebop', pop_exponent, trials_exponent, species_name, final_time)
    ppsim_fn = fn_count_samples('ppsim', pop_exponent, trials_exponent, species_name, final_time)
    
    rebop_counts = read_count_samples(rebop_fn)
    ppsim_counts = read_count_samples(ppsim_fn)
    
    fig, ax = plt.subplots(figsize = (10,4))
    # print((results_batching).shape)
    # print((results_batching[state].squeeze().tolist()))
    # print(results_rebop) 
    # print([results_batching[state].squeeze().tolist(), results_rebop])
    # ax.hist(results_rebop)
    ax.hist([ppsim_counts, rebop_counts], # type: ignore
            bins = 20,
            alpha = 1, label=['batching', 'rebop']) #, density=True, edgecolor = 'k', linewidth = 0.5)
    ax.legend()

    ax.set_xlabel(f'Count of species {species_name}')
    ax.set_ylabel(f'Number of samples')
    ax.set_title(f'Species {species_name} distribution sampled at simulated time {final_time} '
                 f'(n=$10^{pop_exponent}$; trials=$10^{trials_exponent}$)')
    
    # plt.ylim(0, 200_000)
    pdf_fn = fn_count_samples('ppsim-vs-rebop', pop_exponent, trials_exponent, species_name, final_time)
    pdf_fn = pdf_fn.replace('.json', '.pdf')
    plt.savefig(pdf_fn, bbox_inches='tight')
    plt.show()

# change what's in the next two functions to change the CRN tested
def rebop_dimerization_with_inits(pop_exponent: int) -> tuple[rb.Gillespie, dict[str, int]]:
    crn = rb.Gillespie()
    crn.add_reaction(0.1 ** pop_exponent, ['M', 'M'], ['D'])
    crn.add_reaction(1, ['D'], ['M', 'M'])
    
    n = int(10 ** pop_exponent)
    m_init = n
    inits = {'M': m_init}
    
    return crn, inits

def ppsim_dimerization_crn(pop_exponent: int, seed: int) -> pp.Simulation:
    m, d = pp.species('M D')
    rxns = [
        (m + m >> d).k(1),
        (d >> m + m).k(1),
    ]
    
    n = int(10 ** pop_exponent)
    m_init = n
    inits = {m: m_init}
    
    sim = pp.Simulation(inits, rxns, simulator_method="crn", continuous_time=True, seed=seed)
    
    return sim

def main():
    pop_exponent = 2
    trials_exponent = 5
    final_time = 0.5
    species_name = 'M'
    seed = 4
    rebop_crn, rebop_inits = rebop_dimerization_with_inits(pop_exponent)
    ppsim_sim = ppsim_dimerization_crn(pop_exponent, seed)
    # print(f'Running dimerization test with n = 10^{pop_exponent} for 10^{trials_exponent} trials')
    # ppsim_sim.run(final_time, 0.01) # type: ignore
    # print(f'done with ppsim')
    # ppsim_sim.history.plot(figsize=(10, 4)) # type: ignore
    # plt.show()
    # write_rebop_count_samples(rebop_crn, rebop_inits, pop_exponent, trials_exponent, species_name, final_time)
    # write_ppsim_count_samples(ppsim_sim, pop_exponent, trials_exponent, species_name, final_time)
    plot_rebop_ppsim_histogram(pop_exponent, trials_exponent, species_name, final_time)

if __name__ == "__main__":
    main()