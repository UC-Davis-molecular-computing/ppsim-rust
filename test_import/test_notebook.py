"""
Test script to verify that we can run one of the example notebooks with the virtual environment.
"""

import sys
import os
import subprocess

def main():
    print("Python executable:", sys.executable)
    print("Python version:", sys.version)
    print()
    
    try:
        import ppsim
        print("Successfully imported ppsim package!")
        
        # Try to import some specific modules
        from ppsim import species, Simulation
        print("Successfully imported species and Simulation from ppsim!")
        
        # Try to run a simple simulation
        a, b, u = species('A B U')
        approx_majority = [
            a + b >> 2 * u,
            a + u >> 2 * a,
            b + u >> 2 * b,
        ]
        n = 100  # Using a small value for quick testing
        a_init = int(n * 0.51)
        b_init = n - a_init
        init = {a: a_init, b: b_init}
        sim = Simulation(init, approx_majority, seed=1)
        sim.run(10)
        print("Successfully ran a simulation!")
        print(f"Final configuration: {sim.config_dict}")
        
        print("\nAll tests successful!")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
