import sys
import importlib.util
import os
from pathlib import Path

if False:
    # Path to your renamed .pyd file
    custom_pyd_path = Path("C:/Dropbox/git/ppsim-rust/python/ppsim/ppsim_rust/ppsim_rust.cp312-win_amd64_2.pyd")

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
import numpy as np
import gpac as gp
import polars as pl
from matplotlib import pyplot as plt
import rebop as rb
import timeit

def main():
    import matplotlib.pyplot as plt

    # Get the default color cycle
    # default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    # print(default_colors)
    # return
    # for pop_exponent in [4, 6, 8]:
    for pop_exponent in [3]:
        #XXX: pop_exponent 5 and 6 these show the slowdown bug in ppsim
        # going to time 20, for n=10^5 around time 6.966 (35% progress bar)
        # and for n=10^6, around time 13.718 (69% progress bar),
        # there is a massive slowdown in ppsim compared to the other time intervals.
        # Despite using the same random seed each time, the exact times when this
        # happens is stochastic, but it is always around the same time.
        make_and_save_plot(pop_exponent)
    
def make_and_save_plot(pop_exponent: int) -> None:
    seed = 4
    crn = rb.Gillespie()
    crn.add_reaction(.1 ** pop_exponent, ['R', 'F'], ['F', 'F'])
    crn.add_reaction(1, ['R'], ['R', 'R'])
    crn.add_reaction(1, ['F'], [])
    n = int(10 ** pop_exponent)
    p = 0.5
    r_init = int(n * p)
    f_init = n - r_init
    inits = {'R': r_init, 'F': f_init}
    end_time = 20.0
    num_samples = 10**3
    results_rebop = {}
    print(f'running rebop with n = 10^{pop_exponent}')
    results_rebop = crn.run(inits, end_time, num_samples, rng=seed)

    r,f = pp.species('R F')
    rxns = [
        (r+f >> 2*f).k(1),
        (r >> 2*r).k(1),
        (f >> None).k(1),
    ]
    
    inits = {r: r_init, f: f_init}
    sim = pp.Simulation(inits, rxns, simulator_method="crn", continuous_time=True, seed=seed)

    sim.run(end_time, end_time / num_samples)
    # sim.history.plot(figsize = (15,4))
    # plt.ylim(0, 2.1 * n)
    # plt.title('lotka volterra (with batching)')
    
    # print(f"Total reactions simulated: {sampling_increment * len(results_rebop['R'])}")

    # f, ax = plt.subplots()
    f, ax = plt.subplots(figsize=(8, 4))

    blue, orange, green, red  = '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'
    ax.plot(results_rebop['time'], results_rebop['R'], label='R (rebop)', color=red)
    ax.plot(results_rebop['time'], results_rebop['F'], label='F (rebop)', color=green)
    ax.plot(sim.history['R'], label = 'R (batching)', color=blue)
    ax.plot(sim.history['F'], label = 'F (batching)', color=orange)
    ax.legend(loc='upper left')
    plt.savefig(f'data/lotka_volterra_counts_time10_n1e{pop_exponent}.pdf', bbox_inches='tight')
    # plt.show()
    

if __name__ == "__main__":
    main()