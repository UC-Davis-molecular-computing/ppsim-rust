
import ppsim as pp
from matplotlib import pyplot as plt

def main():
    a, b, u = pp.species('A B U')
    approx_majority = [
        a + b >> 2 * u,
        a + u >> 2 * a,
        b + u >> 2 * b,
    ]
    # init = {a: 50_001_000, b: 49_990_000}
    n = 10**9
    a_init = int(n * 0.51)
    b_init = n - a_init
    init = {a: a_init, b: b_init}
    # init = {a: 6, b: 4}
    sim = pp.Simulation(init, approx_majority, seed=1)
    # i, = species('I')
    # epidemic = [ i+u >> 2*i ]
    # init = { i:1, u:9 }
    # sim = Simulation(init, epidemic, seed=0)
    sim.run(10)
    # sim.history.plot()
    # plt.title('approximate majority protocol')
    # plt.xlim(0, sim.times[-1])
    # plt.ylim(0, sum(init.values()))
    # plt.savefig('examples/approx_majority_plot.png')
    # print("Plot saved to examples/approx_majority_plot.png")
    print(f"history = {sim.history}")


    num_multibatch_steps = sum(sim.simulator.collision_counts.values())
    num_collisions = 0
    digits_steps = 1
    digits_collisions = 1
    for collision_count, steps in sim.simulator.collision_counts.items():
        num_collisions += steps * collision_count
        digits_collisions = max(digits_collisions, len(str(num_collisions)))
        digits_steps = max(digits_steps, len(str(steps)))
    print(f'collision counts = (total: {num_collisions}, num_multibatch_steps: {num_multibatch_steps})')  
    for steps in sorted(sim.simulator.collision_counts.keys()):
        print(f'{sim.simulator.collision_counts[steps]:{digits_steps}} multibatch steps with {steps} collisions')
    
    # for count in sorted(sim.simulator.collision_counts.keys()):
    #     print(f'  {count}: {count*sim.simulator.collision_counts[count]:{digits_collisions}} collisions')
    # print('-'*20)
    
    
    sim.simulator.write_profile()
    

def main2():
    a,b,u = pp.species('A B U')
    approx_majority = [
    a+b >> 2*u,
    a+u >> 2*a,
    b+u >> 2*b,
    ]
    n = 10 ** 7
    p = 0.51
    a_init = int(n * p)
    b_init = n - a_init
    init = {a: a_init, b: b_init}
    # for seed in range(100):
    #     print(f'{seed=}')
    seed = 10
    sim = pp.Simulation(init, approx_majority, seed=seed, 
                        # simulator_method='sequential'
                        )
    sim.run(20, 1)
    # sim.run(100)
    print(sim.history)

def main3():
    # derived rate constants of the formal reaction simulated by DNA strand displacement (units of /M/s)
    k1,k2,k3 = 9028, 2945, 1815
    total_concentration = 80 * 1e-9 # 1x volume was 80 nM
    vol = 1e-14 # 1 uL
    n = pp.concentration_to_count(total_concentration, vol)
    a,b,u = pp.species('A B U')
    approx_majority_rates = [
        (a+b >> 2*u).k(k1, units=pp.RateConstantUnits.mass_action),
        (a+u >> 2*a).k(k2, units=pp.RateConstantUnits.mass_action),
        (b+u >> 2*b).k(k3, units=pp.RateConstantUnits.mass_action),
    ]
    # set the initial concentrations near where the the mass-action CRN would reach an unstable equilibrium
    p = 0.45
    inits = {a: int(p*n), b: int((1-p)*n)}
    print(f'{inits=}')
    sim = pp.Simulation(inits, approx_majority_rates, volume=vol, time_units='seconds')
    print('delta:')
    for row in sim.simulator.delta:
        print(row)
    print('random_transitions:')
    for row in sim.simulator.random_transitions: # type: ignore
        print(row)
    print(f'{sim.simulator.random_outputs=}') # type: ignore
    print(f'{sim.simulator.transition_probabilities=}') # type: ignore
    # sim.run()
    # print(f"history = {sim.history}")

if __name__ == '__main__':
    main2()
