"""
Test script to verify that the ppsim package can be imported from outside the project root.
"""

try:
    import ppsim
    print("Successfully imported ppsim package!")
    print(f"ppsim version: {ppsim.__version__ if hasattr(ppsim, '__version__') else 'unknown'}")
    
    # Try to import some specific modules
    from ppsim import species, Simulation
    print("Successfully imported species and Simulation from ppsim!")
    
    # Try to import the Rust extension
    from ppsim_rust import SimulatorSequentialArray, SimulatorMultiBatch
    print("Successfully imported SimulatorSequentialArray and SimulatorMultiBatch from ppsim_rust!")
    
    print("\nAll imports successful!")
except ImportError as e:
    print(f"Error importing ppsim: {e}")
