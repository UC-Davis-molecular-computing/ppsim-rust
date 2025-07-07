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
    crn = rb.Gillespie()
    pop_exponent = 3
    crn.add_reaction(.1 ** pop_exponent, ['A', 'B'], ['B', 'B'])
    crn.add_reaction(1, ['A'], ['A', 'A'])
    crn.add_reaction(1, ['B'], [])
    sampling_increment = 1000
    n = int(10 ** pop_exponent)
    p = 0.5
    a_init = int(n * p)
    b_init = n - a_init
    inits = {"A": a_init, "B": b_init}
    end_time = 10.0
    num_samples = 200
    results_rebop = {}
    results_rebop = crn.run(inits, end_time, sampling_increment)

    a,b = pp.species('A B')
    rxns = [
        (a+b >> 2*b).k(1),
        (a >> 2*a).k(1),
        (b >> None).k(1),
    ]
    
    inits = {a: a_init, b: b_init}
    sim = pp.Simulation(inits, rxns, simulator_method="crn", continuous_time=True)

    sim.run(end_time, end_time / num_samples)
    # sim.history.plot(figsize = (15,4))
    # plt.ylim(0, 2.1 * n)
    # plt.title('lotka volterra (with batching)')
    
    print(f"Total reactions simulated: {sampling_increment * len(results_rebop['A'])}")

    f, ax = plt.subplots()

    ax.plot(results_rebop['time'], results_rebop['B'], label='B (rebop)')
    ax.plot(results_rebop['time'], results_rebop['A'], label='A (rebop)')
    # print(sim.history)
    # print(results_rebop)
    # print(np.linspace(0, end_time, num_samples + 1))
    # print(sim.history['A'])
    ax.plot(sim.history['A'], label = 'A (ppsim)')
    ax.plot(sim.history['B'], label = 'B (ppsim)')
    # ax2.plot(np.linspace(0, end_time, num_samples + 1), sim.history['A'], label='A (ppsim)')
    # ax2.plot(np.linspace(0, end_time, num_samples + 1), sim.history['B'], label='B (ppsim)')
    # ax.hist([results_rebop['A'], results_rebop['B']], bins = np.linspace(0, n, 20), 
    #         alpha = 1, label=['A', 'B']) #, density=True, edgecolor = 'k', linewidth = 0.5)
    ax.legend()
    sim.simulator.write_profile() # type: ignore

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