import math
import numpy as np
import ppsim as pp
import random
from tqdm import tqdm
from matplotlib import pyplot as plt
import timeit
import time
import rebop as rb

def test_time_scaling_vs_population():
    ppsim_times = []
    rebop_times = []
    ns = []
    num_ns = 20
    min_pop_exponent = 5
    max_pop_exponent = 12
    num_checkpoints = 1
    num_trials = 2
    end_time = .001
    for pop_exponent_increment in tqdm(range(num_ns)):
        pop_exponent = min_pop_exponent + (max_pop_exponent - min_pop_exponent) * (pop_exponent_increment / (float(num_ns) - 1))
        a,b = pp.species('A B')
        
        predator_fraction = 0.5 
        # pop_exponent = 6

        rxns = [
            (a+b >> 2*b).k(0.1 ** pop_exponent),
            (a >> 2*a).k(1),
            (b >> None).k(1),
        ]

        
        def timefn(n):
            a_init = int(n * (1 - predator_fraction))
            b_init = n - a_init
            inits = {a: a_init, b: b_init}
            seed=76567
            sim = pp.Simulation(inits, rxns, simulator_method="crn", continuous_time=True, seed=seed)
            sim.run(end_time, end_time / float(num_checkpoints))
        # sim = pp.Simulation(inits, rxns, simulator_method="crn", continuous_time=True)
        
        crn = rb.Gillespie()
        crn.add_reaction(0.1 ** pop_exponent, ['A', 'B'], ['B', 'B'])
        crn.add_reaction(1, ['A'], ['A', 'A'])
        crn.add_reaction(1, ['B'], [])

        def timefnrebop(n):
            a_init = int(n * (1 - predator_fraction))
            b_init = n - a_init
            inits = {"A": a_init, "B": b_init}
            results_rebop = crn.run(inits, end_time, 1)

        
        n = int(10 ** pop_exponent)
        ppsim_times.append(timeit.timeit(lambda: timefn(n), number=num_trials))
        rebop_times.append(timeit.timeit(lambda: timefnrebop(n), number=num_trials))
        ns.append(n)
        
    fig, ax = plt.subplots(figsize = (10,4))
    ax.loglog(ns, ppsim_times, label="ppsim run time")
    ax.loglog(ns, rebop_times, label="rebop run time")
    # sim.simulator.write_profile() # type: ignore
    ax.set_xlabel(f'Initial molecular count')
    ax.set_ylabel(f'Run time (s)')
    ax.legend()
    plt.show()
    return

def test_time_scaling_vs_end_time():
    ppsim_times = []
    rebop_times = []
    times = []
    num_times = 2
    pop_exponent = 9
    n = 10 ** pop_exponent
    min_time_exponent = -6
    max_time_exponent = -2
    num_checkpoints = 1
    num_trials = 1
    for time_exponent_increment in tqdm(range(num_times)):
        time_exponent = min_time_exponent + (max_time_exponent - min_time_exponent) * (time_exponent_increment / (float(num_times - 1.0)))
        a,b = pp.species('A B')
        
        predator_fraction = 0.5 

        rxns = [
            (a+b >> 2*b).k(0.1 ** pop_exponent),
            (a >> 2*a).k(1),
            (b >> None).k(1),
        ]

        
        def timefn(t):
            a_init = int(n * (1 - predator_fraction))
            b_init = n - a_init
            inits = {a: a_init, b: b_init}
            seed=random.randint(1, 10000)
            sim = pp.Simulation(inits, rxns, simulator_method="crn", continuous_time=True, seed=4)
            sim.sample_future_configuration(t, num_samples=1)
            # sim.run(t, t / float(num_checkpoints))
        # sim = pp.Simulation(inits, rxns, simulator_method="crn", continuous_time=True)
        
        

        def timefnrebop(t):
            crn = rb.Gillespie()
            crn.add_reaction(0.1 ** pop_exponent, ['A', 'B'], ['B', 'B'])
            crn.add_reaction(1, ['A'], ['A', 'A'])
            crn.add_reaction(1, ['B'], [])
            a_init = int(n * (1 - predator_fraction))
            b_init = n - a_init
            inits = {"A": a_init, "B": b_init}
            results_rebop = crn.run(inits, t, 1)
            # get_rebop_samples(pop_exponent, 1, b_init, "B", t)
        
        end_time = 10 ** time_exponent
        ppsim_times.append(timeit.timeit(lambda: timefn(end_time), number=num_trials))
        rebop_times.append(timeit.timeit(lambda: timefnrebop(end_time), number=num_trials))
        times.append(end_time)
        
    fig, ax = plt.subplots(figsize = (10,4))
    ax.loglog(times, ppsim_times, label="ppsim run time")
    ax.loglog(times, rebop_times, label="rebop run time")
    # sim.simulator.write_profile() # type: ignore
    ax.set_xlabel(f'Simulated continuous time units)')
    ax.set_ylabel(f'Run time (s)')
    ax.legend()
    plt.show()
    return

def get_rebop_samples(pop_exponent, trials, predator_count, state, final_time):
    n = 10 ** pop_exponent
    output = []
    total_time_inside_simulation = 0.0
    total_time_outside = 0.0
    for _ in tqdm(range(trials)):
        crn = rb.Gillespie()
        crn.add_reaction(0.1 ** pop_exponent, ['A', 'B'], ['B', 'B'])
        crn.add_reaction(1, ['A'], ['A', 'A'])
        crn.add_reaction(1, ['B'], [])
        b_init = predator_count
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
                print("Index error caught and ignored. Rebop distribution may be slightly off.")
    return output

def test_distribution():
    pop_exponent = 9
    trials_exponent = 6
    final_time_exponent = -6
    a,b = pp.species('A B')
    
    final_time = 10 ** final_time_exponent
    rxns = [
        (a+b >> 2*b).k(0.1 ** pop_exponent),
        (a >> 2*a).k(1),
        (b >> None).k(1),
    ]

    n = 10 ** pop_exponent
    a_init = n // 2
    b_init = n - a_init
    inits = {a: a_init, b: b_init}
    sim = pp.Simulation(inits, rxns, simulator_method="crn", continuous_time=True, seed=4)
    
    trials = 10 ** trials_exponent
    
    # The simulator multiplies by n currently so just gonna be lazy here.
    # state = 'A'
    state = 'B'
    results_batching = sim.sample_future_configuration(final_time, num_samples=trials)
    print(f"total reactions simulated by batching: {sim.simulator.discrete_steps_not_including_nulls}") #type: ignore
    sim.simulator.write_profile() # type: ignore
    results_rebop = get_rebop_samples(pop_exponent, trials, b_init, state, final_time)
    
    fig, ax = plt.subplots(figsize = (10,4))
    # print((results_batching).shape)
    # print((results_batching[state].squeeze().tolist()))
    # print(results_rebop) 
    # print([results_batching[state].squeeze().tolist(), results_rebop])
    # ax.hist(results_rebop)
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
    # test_time_scaling_vs_population()
    # test_time_scaling_vs_end_time()
    test_distribution()

if __name__ == "__main__":
    main()