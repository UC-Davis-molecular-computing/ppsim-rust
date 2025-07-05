import math
import numpy as np
import ppsim as pp
import random
from tqdm import tqdm
from matplotlib import pyplot as plt
import timeit
import time
import rebop as rb
from scipy import stats

def test_time_scaling_vs_population():
    ppsim_times = []
    rebop_times = []
    ns = []
    num_ns = 13
    min_pop_exponent = 6
    max_pop_exponent = 14
    num_checkpoints = 1
    num_trials = 1
    end_time = .0001
    ESTIMATE_BIG_REBOP_VALUES = True
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
        
        seed=765467
        a_init = 1
        b_init = 1
        fake_inits = {a: a_init, b: b_init}
        sim = pp.Simulation(fake_inits, rxns, simulator_method="crn", continuous_time=True, seed=seed)
        
        def timefn(n):
            a_init = int(n * (1 - predator_fraction))
            b_init = n - a_init
            inits = {a: a_init, b: b_init}
            sim.reset(inits) # type: ignore
            # s = pp.Simulation(inits, rxns, simulator_method="crn", continuous_time=True, seed=seed)
            sim.run(end_time, end_time / float(num_checkpoints))
            sim.simulator.write_profile() # type: ignore
        # sim = pp.Simulation(inits, rxns, simulator_method="crn", continuous_time=True)
        
        crn = rb.Gillespie()
        crn.add_reaction(0.1 ** pop_exponent, ['A', 'B'], ['B', 'B'])
        crn.add_reaction(1, ['A'], ['A', 'A'])
        crn.add_reaction(1, ['B'], [])

        def timefnrebop(n, t):
            a_init = int(n * (1 - predator_fraction))
            b_init = n - a_init
            inits = {"A": a_init, "B": b_init}
            results_rebop = crn.run(inits, t, 1)

        
        n = int(10 ** pop_exponent)
        print(f"Iteration {pop_exponent_increment}! Population {n}.")
        ppsim_times.append(timeit.timeit(lambda: timefn(n), number=num_trials))
        # If this is gonna take way too long, run it on a smaller number of interactions
        # and multiply to correct. It looks like for big population sizes, run time is
        # generally linear with respect to input time.
        # We'll do that regression to make sure we don't accidentally overestimate based
        # on the intercept of the line.
        print("Done with ppsim, now rebop!")
        if ESTIMATE_BIG_REBOP_VALUES:
            est_runtimes = []
            est_ts = []
            first_test_end_time_exponent = 4 - pop_exponent
            last_test_end_time_exponent = 6 - pop_exponent
            for i in range(100):
                # print(f"Testing time {i}")
                test_time_exponent = first_test_end_time_exponent + i * (last_test_end_time_exponent - first_test_end_time_exponent) / 100.0
                test_time = 10 ** test_time_exponent
                est_ts.append(test_time)
                est_runtimes.append(timeit.timeit(lambda: timefnrebop(n, test_time), number = num_trials))
            slope, intercept, r, _, _ = stats.linregress(est_ts, est_runtimes)
            if r.item() < 0.99: #type: ignore
                print(f"Careful: r is {r}")
                assert False
            new_end_time = 10 ** last_test_end_time_exponent
            correction_factor = end_time/new_end_time
            print(f"Correction factor is {correction_factor} and new time is {new_end_time}")
            time_taken = timeit.timeit(lambda: timefnrebop(n, new_end_time), number=num_trials)
            # assume time_taken = a = intercept + slope * new_end_time
            # we want to find intercept + slope * end_time = (a - intercept) * (correction_factor) + intercept
            rebop_times.append((time_taken - intercept.item()) * correction_factor + intercept.item()) # type: ignore
        else:
            rebop_times.append(timeit.timeit(lambda: timefnrebop(n, end_time), number=num_trials))
        ns.append(n)
        
    fig, ax = plt.subplots(figsize = (10,4))
    rebop_label = "rebop run time"
    if ESTIMATE_BIG_REBOP_VALUES:
        rebop_label += "(estimated based on smaller t input)"
    ax.loglog(ns, ppsim_times, label="ppsim run time")
    ax.loglog(ns, rebop_times, label=rebop_label)
    ax.set_xlabel(f'Initial molecular count')
    ax.set_ylabel(f'Run time (s)')
    ax.legend()
    plt.show()
    print(stats.linregress([math.log(x) for x in ns], [math.log(x) for x in ppsim_times]))
    print(stats.linregress([math.log(x) for x in ns], [math.log(x) for x in rebop_times]))
    print(ns)
    print(ppsim_times)
    print(rebop_times)
    return

def test_time_scaling_vs_end_time():
    ppsim_times = []
    rebop_times = []
    times = []
    num_times = 20
    pop_exponent = 10
    n = 10 ** pop_exponent
    min_time_exponent = -5
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
            get_rebop_samples(pop_exponent, 1, b_init, "B", t)
        
        end_time = 10 ** time_exponent
        # end_time = 0.1 + 0.1 * time_exponent_increment
        ppsim_times.append(timeit.timeit(lambda: timefn(end_time), number=num_trials))
        rebop_times.append(timeit.timeit(lambda: timefnrebop(end_time), number=num_trials))
        times.append(end_time)
        
    fig, ax = plt.subplots(figsize = (10,4))
    ax.plot(times, ppsim_times, label="ppsim run time")
    ax.plot(times, rebop_times, label="rebop run time")
    # sim.simulator.write_profile() # type: ignore
    ax.set_xlabel(f'Simulated continuous time units)')
    ax.set_ylabel(f'Run time (s)')
    # ax.set_xscale("log")
    # ax.set_yscale("log")
    ax.legend()
    print(stats.linregress(times, rebop_times))

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
    pop_exponent = 10.69
    trials_exponent = 0
    final_time_exponent = -2
    a,b = pp.species('A B')
    
    final_time = 10 ** final_time_exponent
    rxns = [
        (a+b >> 2*b).k(0.1 ** pop_exponent),
        (a >> 2*a).k(1),
        (b >> None).k(1),
    ]

    n = 10 ** pop_exponent
    print(f"n = {n}")
    a_init = n // 2
    b_init = n - a_init
    inits = {a: a_init, b: b_init}
    sim = pp.Simulation(inits, rxns, simulator_method="crn", continuous_time=True, seed=4) #type: ignore
    
    trials = 3
    
    # The simulator multiplies by n currently so just gonna be lazy here.
    # state = 'A'
    state = 'B'
    results_batching = sim.sample_future_configuration(final_time, num_samples=trials)
    print(f"total reactions simulated by batching: {sim.simulator.discrete_steps_not_including_nulls}") #type: ignore
    # sim.simulator.write_profile() # type: ignore
    # results_rebop = get_rebop_samples(pop_exponent, trials, b_init, state, final_time)
    # fig, ax = plt.subplots(figsize = (10,4))
    # # print((results_batching).shape)
    # # print((results_batching[state].squeeze().tolist()))
    # # print(results_rebop) 
    # # print([results_batching[state].squeeze().tolist(), results_rebop])
    # # ax.hist(results_rebop)
    # ax.hist([results_batching[state].squeeze().tolist(), results_rebop], # type: ignore
    #         #bins = np.linspace(int(n*0.32), int(n*.43), 20), # type: ignore
    #         alpha = 1, label=['ppsim', 'rebop']) #, density=True, edgecolor = 'k', linewidth = 0.5)
    # ax.legend()

    # ax.set_xlabel(f'Count of state {state}')
    # ax.set_ylabel(f'Number of samples')
    # ax.set_title(f'State {state} distribution sampled at simulated time {final_time} ($10^{trials_exponent}$ samples)')
    
    # # plt.ylim(0, 200_000)

    # # plt.savefig(pdf_fn, bbox_inches='tight')
    # plt.show()


def main():
    test_time_scaling_vs_population()
    # test_time_scaling_vs_end_time()
    # test_distribution()

if __name__ == "__main__":
    main()