import math
import numpy as np
import ppsim as pp
import random
from tqdm import tqdm
from matplotlib import pyplot as plt
import timeit
import rebop as rb

def test_time_scaling():
    times = []
    ns = []
    for pop_exponent_increment in range(10):
        pop_exponent = 5 + 0.1 * pop_exponent_increment
        n = int(10 ** pop_exponent)
        a,b = pp.species('A B')
        
        predator_fraction = 0.5 
        # pop_exponent = 6

        rxns = [
            (a+b >> 2*b).k(0.1 ** pop_exponent),
            (a >> 2*a).k(1),
            (b >> None).k(1),
        ]

        # n = 10 ** pop_exponent
        a_init = int(n * (1 - predator_fraction))
        b_init = n - a_init
        inits = {a: a_init, b: b_init}
        def timefn(sim):
            sim.run(50, 0.1)
        sim = pp.Simulation(inits, rxns, simulator_method="crn")
        times.append(timeit.timeit(lambda: timefn(sim), number=1))
        ns.append(n)
        
    fig, ax = plt.subplots(figsize = (10,4))
    ax.plot(ns, times, label="Run time vs pop size")
    plt.show()
    return

def get_rebop_samples(pop_exponent, trials, predator_fraction, state, final_time):
    n = 10 ** pop_exponent
    output = []
    for _ in tqdm(range(trials)):
        crn = rb.Gillespie()
        crn.add_reaction(0.1 ** pop_exponent, ['A', 'B'], ['B', 'B'])
        crn.add_reaction(1, ['A'], ['A', 'A'])
        crn.add_reaction(1, ['B'], [])
        b_init = int(n * predator_fraction)
        a_init = n - b_init
        inits = {"A": a_init, "B": b_init}
        # It should be very roughly 1 step every 1/n real time, so to get a particular number
        # of steps, it should be safe to run for, say, 3 times that much time
        while True:
            try:
                results_rebop = crn.run(inits, final_time, 1)
                # print(f"There are {len(results_rebop[state])} total steps in rebop simulation.")
                # print(results_rebop[state])
                output.append(int(results_rebop[state][-1]))
                break
            except IndexError:
                pass
                #print("Index error caught and ignored. Rebop distribution may be slightly off.")
    return output

def test_distribution():
    pop_exponent = 6
    trials_exponent = 2
    a,b = pp.species('A B')
    
    predator_fraction = 0.5
    final_time = 1

    rxns = [
        (a+b >> 2*b).k(0.1 ** pop_exponent),
        (a >> 2*a).k(1),
        (b >> None).k(1),
    ]

    n = 10 ** pop_exponent
    a_init = int(n * (1 - predator_fraction))
    b_init = n - a_init
    inits = {a: a_init, b: b_init}
    sim = pp.Simulation(inits, rxns, simulator_method="crn", continuous_time=True)
    trials = 10 ** trials_exponent
    
    # The simulator multiplies by n currently so just gonna be lazy here.
    results_batching = sim.sample_future_configuration(final_time, num_samples=trials)
    # state = 'A'
    state = 'B'
    results_rebop = get_rebop_samples(pop_exponent, trials, predator_fraction, state, final_time)
    
    fig, ax = plt.subplots(figsize = (10,4))
    # print((results_batching).shape)
    # print((results_batching[state].squeeze().tolist()))
    # print(results_rebop) 
    # print([results_batching[state].squeeze().tolist(), results_rebop])
    ax.hist([results_batching[state].squeeze().tolist(), results_rebop], # type: ignore
            #bins = np.linspace(int(n*0.32), int(n*.43), 20), # type: ignore
            alpha = 1, label=['ppsim', 'rebop']) #, density=True, edgecolor = 'k', linewidth = 0.5)
    ax.legend()

    ax.set_xlabel(f'Count of state {state}')
    ax.set_ylabel(f'Number of samples')
    ax.set_title(f'State {state} distribution sampled at simulated time {final_time} ($10^{trials_exponent}$ samples)')
    
    # plt.ylim(0, 200_000)

    # plt.savefig(pdf_fn, bbox_inches='tight')
    plt.show()


def main():
    # test_time_scaling()
    test_distribution()
    

if __name__ == "__main__":
    main()