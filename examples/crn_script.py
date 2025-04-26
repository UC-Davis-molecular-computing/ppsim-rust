
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
    # for collision_count, count in sim.simulator.collision_counts.items():
    #     num_collisions += count * collision_count
    # print(f'collision counts = (total: {num_collisions}, num_multibatch_steps: {num_multibatch_steps})')
    # for count in sorted(sim.simulator.collision_counts.keys()):
    #     print(f'  {count}: {count*sim.simulator.collision_counts[count]}')
    # for count in sorted(sim.simulator.collision_counts.keys()):
    #     print(f'  {count}: {sim.simulator.collision_counts[count]}')
    sim.simulator.write_profile()
    sim.simulator.print_ln_fact_stats()


if __name__ == '__main__':
    main()
