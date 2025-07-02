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
    num_ns = 10
    min_pop_exponent = 8
    max_pop_exponent = 11
    end_time = 1
    num_checkpoints = 1
    num_trials = 1
    end_time = .01
    for pop_exponent_increment in tqdm(range(num_ns)):
        pop_exponent = min_pop_exponent + (max_pop_exponent - min_pop_exponent) * (pop_exponent_increment / float(num_ns))
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
            # results_rebop = crn.run(inits, end_time, 1)

        
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
    num_times = 50
    pop_exponent = 5
    n = 10 ** pop_exponent
    min_time_exponent = -1
    max_time_exponent = 3
    num_checkpoints = 1
    num_trials = 1
    for time_exponent_increment in tqdm(range(num_times)):
        time_exponent = min_time_exponent + (max_time_exponent - min_time_exponent) * (time_exponent_increment / float(num_times))
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
            sim = pp.Simulation(inits, rxns, simulator_method="crn", continuous_time=True, seed=seed)
            sim.run(t, t / float(num_checkpoints))
        # sim = pp.Simulation(inits, rxns, simulator_method="crn", continuous_time=True)
        
        crn = rb.Gillespie()
        crn.add_reaction(0.1 ** pop_exponent, ['A', 'B'], ['B', 'B'])
        crn.add_reaction(1, ['A'], ['A', 'A'])
        crn.add_reaction(1, ['B'], [])

        def timefnrebop(t):
            a_init = int(n * (1 - predator_fraction))
            b_init = n - a_init
            inits = {"A": a_init, "B": b_init}
            results_rebop = crn.run(inits, t, 1)
        
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
                #print("Index error caught and ignored. Rebop distribution may be slightly off.")
    return output

def test_distribution():
    pop_exponent = 10
    trials_exponent = 2
    a,b = pp.species('A B')
    
    final_time = .0001
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
    sim.simulator.write_profile() # type: ignore
    plt.show()


def main():
    # test_time_scaling_vs_population()
    # test_time_scaling_vs_end_time()
    test_distribution()
    # print(np.var([0.0004137004210096707, 0.00043045890645416904, 0.00042289507521219087, 0.00042930694152943413, 0.00042757780755112314, 0.0004319395534457655, 0.0004137262540749692, 0.0004195466537484041, 0.00041517585295358325, 0.00043222741198055783, 0.0004133793741877948, 0.00041756095932248607, 0.00041933378757451936, 0.0003979980033954224, 0.00043204236816201725, 0.00041731527480158077, 0.0004131465520193244, 0.0004211260171395059, 0.000441768510785614, 0.0004050573067975612, 0.00040492177307242225, 0.0004458242314179573, 0.00045264472856995977, 0.0004217743371984388, 0.000399110318162805, 0.00042152598745219175, 0.00043816832289057524, 0.00042691920215318416, 0.0004497979835066482, 0.0004420971326279384, 0.0004437145100382427, 0.0004409408414047351, 0.00043554277809294394, 0.00039613945272503676, 0.0004215552355627866, 0.0003981297012068525, 0.0004421315045357234, 0.00044971532753710285, 0.00041690601094114115, 0.0004348007740253359, 0.00040692602984519457, 0.000430317198830488, 0.0004110104244337196, 0.0004229624616312514, 0.00044663665822517335, 0.00041912799712501324, 0.0004366303605919236, 0.00040887998340656507, 0.00042489607766396355, 0.0004236016349475535, 0.00041312813831126394, 0.00042452089341532735, 0.000434130263918514, 0.0004305471190639567, 0.00044944117540135427, 0.00040271589861531426, 0.0004121127915412501, 0.0004428128869152366, 0.0004071904491549638, 0.00042444481844412574, 0.00042946980659656865, 0.0004190895022219966, 0.00042298518503897857, 0.00042328696204984407, 0.00041600526478329755, 0.0004265891058490687, 0.0004194111180807067, 0.0004323659207818684, 0.00042974991066277283, 0.00039912933057382723, 0.00040784573792046247, 0.0004443352641085213, 0.0004442558451533904, 0.0004354142630048045, 0.0004275331637374317, 0.0004472210805598265, 0.0004078963073400525, 0.00042729648781939395, 0.000417872962230421, 0.00038949335505490977, 0.0003983775191771069, 0.00040461103128694784, 0.00045085063594050857, 0.00038795895787137563, 0.00042270851317873946, 0.00040450989176128217, 0.0004089559202190627, 0.0004197570009233255, 0.00041576392176527694, 0.0004186836827460945, 0.0004286987002127719, 0.00043189414557410393, 0.00042398614528359066, 0.0004099010883518591, 0.00042507030714893904, 0.00041689382221718154, 0.0004374587334876468, 0.00041898768319061915, 0.0004181184922373353, 0.00042655668333320857, 0.0004430120327989079, 0.0004419480179229753, 0.0004251225670326515, 0.0004046588888913046, 0.00043173948730691595, 0.0004071892786230759, 0.00043388225287421014, 0.00041391083078273387, 0.00043146849917749666, 0.000426308270128256, 0.00042087155005172006, 0.00042755229602364244, 0.000431711233486056, 0.0004250147239310635, 0.0004178220724372195, 0.0004262138347699991, 0.0004453943510596014, 0.0004241254128868023, 0.0004356384597282755, 0.0004305020448315432, 0.00041542295542438656, 0.0004311464875087571, 0.0004194990961694067, 0.0004185759044205945, 0.0004071052295832242, 0.0004029333327509819, 0.00042427081733552145, 0.0004486871484144486, 0.00041858017178699695, 0.0004295129474848864, 0.0004269564158388808, 0.0004236272747167649, 0.0004071549125087583, 0.0004249104681506722, 0.00044531210365219117, 0.00042662675431426095, 0.0004416715282439108, 0.00043105163202483503, 0.0004187186817366287, 0.0004194250627898391, 0.0004265367824545604, 0.00042402948614173074, 0.00040633370758629174, 0.0004389439650778068, 0.0004254544027793518, 0.00040080754722790837, 0.00042498808136305535, 0.00043823361560018026, 0.0004126160622583608, 0.00041433346892296336, 0.0004151445075540107, 0.00044525054382104864, 0.00045402456912425546, 0.0004401238011185329, 0.00041532242416357496, 0.0004380129164375552, 0.0004034589871096936, 0.00040709676753272473, 0.00043340480077702105, 0.00046218589916988687, 0.00040766058168368913, 0.00045587188157990445, 0.00041205505754663655, 0.0004130582779061274, 0.00043351215776853125, 0.0004382008349518642, 0.0004246691409837343, 0.0004047203468398125, 0.00039152144676664005, 0.00041429065087927663, 0.0004306294070116435, 0.00043027609384189384, 0.00044774435741125963, 0.0004113231128523897, 0.0004429763954759183, 0.00041930415372136583, 0.0004532615214679796, 0.00043267645683425494, 0.00042759641449575573, 0.0004190725333076735, 0.0004398030249755155, 0.00040418885024693274, 0.00044488419411623767, 0.00043402499655466293, 0.0004291475202087274, 0.00042840626159247234, 0.00037523410712518807, 0.00045331395303708187, 0.0004098170883006862, 0.000428868279724004, 0.0004201659753704669, 0.00041894458289426864, 0.0004323032530713543, 0.00043533275272146255, 0.0004038894303576785, 0.000402332976635194, 0.0004238970898447312, 0.0004176828207778465, 0.00042894482852584575, 0.0004480685745891749]))

if __name__ == "__main__":
    main()