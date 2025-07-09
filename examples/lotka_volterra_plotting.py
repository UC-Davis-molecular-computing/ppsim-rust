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
    # for pop_exponent in [4, 5, 6, 7]:
    for pop_exponent in [5, 6]:
        #XXX: pop_exponent 5 and 6 these show the slowdown bug in ppsim
        # going to time 20, for n=10^5 around time 6.966 (35% progress bar)
        # and for n=10^6, around time 13.718 (69% progress bar),
        # there is a massive slowdown in ppsim compared to the other time intervals.
        # Despite using the same random seed each time, the exact times when this
        # happens is stochastic, but it is always around the same time.
        make_and_save_plot(pop_exponent)
    
def make_and_save_plot(pop_exponent: int) -> None:
    seed = 5
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
    num_samples = 10**5
    # results_rebop = {}
    # results_rebop = crn.run(inits, end_time, num_samples, rng=seed)

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

    # ax.plot(results_rebop['time'], results_rebop['R'], label='R (rebop)')
    # ax.plot(results_rebop['time'], results_rebop['F'], label='F (rebop)')
    # print(sim.history)
    # print(results_rebop)
    # print(np.linspace(0, end_time, num_samples + 1))
    # print(sim.history['R'])
    ax.plot(sim.history['R'], label = 'R (batching)')
    ax.plot(sim.history['F'], label = 'F (batching)')
    # ax2.plot(np.linspace(0, end_time, num_samples + 1), sim.history['R'], label='A (ppsim)')
    # ax2.plot(np.linspace(0, end_time, num_samples + 1), sim.history['F'], label='B (ppsim)')
    # ax.hist([results_rebop['R'], results_rebop['F']], bins = np.linspace(0, n, 20), 
    #         alpha = 1, label=['R', 'F']) #, density=True, edgecolor = 'k', linewidth = 0.5)
    ax.legend(loc='upper left')
    # sim.simulator.write_profile() # type: ignore
    plt.savefig(f'data/lotka_volterra_counts_time10_n1e{pop_exponent}.pdf', bbox_inches='tight')
    plt.show()
    # We could just write gpac reactions directly, but this is ensuring the gpac_format function works.
    # gp_rxns, gp_inits = pp.gpac_format(lotka_volterra, inits)
    # print('Reactions:')
    # for rxn in gp_rxns:
    #     print(rxn)
    # print('Initial conditions:')
    # for sp, count in gp_inits.items():
    #     print(f'{sp}: {count}')
    
    # # for trials_exponent in range(3, 7):
    # # for trials_exponent in range(3, 8):
    # print(f'*************\nCollecting rebop data for pop size 10^{pop_exponent} with 10^{trials_exponent} trials\n')
    # results_rebop = gp.rebop_crn_counts(gp_rxns, gp_inits, end_time)
    # df = pl.DataFrame(results_rebop).to_pandas()
    # df.plot(figsize=(10,5)) # .plot(figsize = (6, 4))
    # plt.title('approximate majority (ppsim)')
    # plt.show()
    # print("Done!")

if __name__ == "__main__":
    main()