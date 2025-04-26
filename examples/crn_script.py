
from ppsim import species, Simulation
from matplotlib import pyplot as plt

def main():
    a, b, u = species('A B U')
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
    sim = Simulation(init, approx_majority, seed=1)
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


    # num_multibatch_steps = sum(sim.simulator.collision_counts.values())
    # num_collisions = 0
    # digits_steps = 1
    # digits_collisions = 1
    # for collision_count, steps in sim.simulator.collision_counts.items():
    #     num_collisions += steps * collision_count
    #     digits_collisions = max(digits_collisions, len(str(num_collisions)))
    #     digits_steps = max(digits_steps, len(str(steps)))
    # print(f'collision counts = (total: {num_collisions}, num_multibatch_steps: {num_multibatch_steps})')  
    # for steps in sorted(sim.simulator.collision_counts.keys()):
    #     print(f'{sim.simulator.collision_counts[steps]:{digits_steps}} multibatch steps with {steps} collisions')
    
    # for count in sorted(sim.simulator.collision_counts.keys()):
    #     print(f'  {count}: {count*sim.simulator.collision_counts[count]:{digits_collisions}} collisions')
    # print('-'*20)
    
    
    sim.simulator.write_profile()
    

def main2():
    a,b,u = species('A B U')
    approx_majority = [
    a+b >> 2*u,
    a+u >> 2*a,
    b+u >> 2*b,
    ]
    n = 10 ** 2
    p = 0.51
    a_init = int(n * p)
    b_init = n - a_init
    init = {a: a_init, b: b_init}
    # for seed in range(100):
    #     print(f'{seed=}')
    seed = 10
    sim = Simulation(init, approx_majority, seed=seed)
    # sim.run(20, 0.1)
    sim.run(100)
    print(sim.history)


if __name__ == '__main__':
    main2()
