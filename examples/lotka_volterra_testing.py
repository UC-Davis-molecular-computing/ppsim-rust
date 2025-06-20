import math
import numpy as np
import ppsim as pp
import random
from tqdm import tqdm
from matplotlib import pyplot as plt

def main():
    a,b = pp.species('A B')
    rxns = [
        a+b >> 2*b,
        a >> 2*a,
        b >> None,
    ]

    predator_fraction = 0.5 
    pop_exponent = 3
    n = 10 ** pop_exponent
    a_init = int(n * (1 - predator_fraction))
    b_init = n - a_init
    inits = {a: a_init, b: b_init}

    sim = pp.Simulation(inits, rxns, simulator_method="crn")
    sim.run(20, 0.1)
    
    sim.history.plot(figsize=(10,5)) # .plot(figsize = (6, 4))
    plt.title('approximate majority (ppsim)')
    plt.show()
    print("Done!")

if __name__ == "__main__":
    main()