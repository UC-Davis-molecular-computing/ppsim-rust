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

def main():
    crn = rb.Gillespie()
    crn.add_reaction(.01, ['A', 'B'], ['B', 'B'])
    crn.add_reaction(10, ['A'], ['A', 'A'])
    crn.add_reaction(10, ['B'], [])

    # a, b = pp.species('A B')
    # lotka_volterra = [
    #     (a+b >> 2*b).k(1),
    #     (a >> 2*a).k(100),
    #     (b >> None).k(100),
    # ]
    pop_exponent = 3
    n = 10 ** pop_exponent
    p = 0.51
    a_init = int(n * p)
    b_init = n - a_init
    inits = {"A": a_init, "B": b_init}
    end_time = 0
    results_rebop = crn.run(inits, 1.1, 0)
    print(results_rebop['A'])

    fig, ax = plt.subplots(figsize = (10,4))
    state = 'A'
    # state = 'B'
    # state = 'U'
    ax.plot(results_rebop['time'], results_rebop['B'], label='B')
    ax.plot(results_rebop['time'], results_rebop['A'], label='A')
    # ax.hist([results_rebop['A'], results_rebop['B']], bins = np.linspace(0, n, 20), 
    #         alpha = 1, label=['A', 'B']) #, density=True, edgecolor = 'k', linewidth = 0.5)
    ax.legend()

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