"""
ppsim: A Python package with Rust backend for simulation
"""

# Re-export everything from Python modules
from ppsim.simulation import *
from ppsim.snapshot import *
from ppsim.crn import *

import numpy as np
import numpy.typing as npt

from abc import ABC

RustState: TypeAlias = int
"""
Type alias for how states are represented in internally in the Rust simulators, 
as integers 0,1,...,num_states-1.
"""

class Simulator(ABC):
    config: list[RustState]
    """TODO"""
    
    n: int
    """TODO"""

    t: int
    """TODO"""

    delta: list[list[tuple[RustState, RustState]]]
    """TODO"""

    silent: bool
    """TODO"""

    def run_until_silent(self) -> None:
        """TODO"""
        ...

    def reset(
            self,
            config: npt.NDArray[np.uint],
            t: int = 0
    ) -> None: 
        """TODO"""
        ...
        
    def run(
            self,
            t_max: int,
            max_wallclock_time: float = 3600.0
    ) -> None:
        """
        Run the simulation for a specified number of steps or until max time is reached.
        
        Args:
            t_max: Maximum number of simulation steps to execute
            max_wallclock_time: Maximum wall clock time in seconds before stopping (default: 1 hour)
        """
        ...

    def __init__(
            self,
            init_config: npt.NDArray[np.uint],
            delta: npt.NDArray[np.uint],
            random_transitions: npt.NDArray[np.uint],
            random_outputs: npt.NDArray[np.uint],
            transition_probabilities: npt.NDArray[np.float64],
            transition_order: str,
            gillespie: bool = False,
            seed: int = 3,
            crn = None,
            k = None,
            w = None,
    ) -> None:
        """TODO"""
        ...


class SimulatorSequentialArray(Simulator):
    """
    Simulator for population protocols using an array to store individual agent states, 
    and sampling them two at a time to interact using the straightforward sequential
    simulation method.
    """

    population: list[RustState]
    """TODO"""




class SimulatorMultiBatch(Simulator):
    """
    Simulator for population protocols using the MultiBatch algorithm described in 
    this paper: 
    
    Petra Berenbrink, David Hammer, Dominik Kaaser, Ulrich Meyer, Manuel Penschuck, and Hung Tran. 
    Simulating Population Protocols in Sub-Constant Time per Interaction. 
    In 28th Annual European Symposium on Algorithms (ESA 2020). 
    Volume 173, pp. 16:1-16:22, 2020.
    
    - https://arxiv.org/abs/2005.03584 
    - https://doi.org/10.4230/LIPIcs.ESA.2020.16
    """

    do_gillespie: bool
    """TODO"""

    reactions: list[list[RustState]]
    """TODO"""

    enabled_reactions: list[int]
    """TODO"""
    
    num_enabled_reactions: int
    """TODO"""

    reaction_probabilities: list[float]
    """TODO"""

    def get_enabled_reactions(self) -> None:
        """TODO"""
        ...

    def get_total_propensity(self) -> float:
        """TODO"""
        ...


class SimulatorCRNMultiBatch(Simulator):
    """
    Simulator for CRNs using TODO cite paper once citeable.
    """

    do_gillespie: bool
    """TODO"""

    reactions: list[list[RustState]]
    """TODO"""

    enabled_reactions: list[int]
    """TODO"""
    
    num_enabled_reactions: int
    """TODO"""

    reaction_probabilities: list[float]
    """TODO"""

    def get_enabled_reactions(self) -> None:
        """TODO"""
        ...

    def get_total_propensity(self) -> float:
        """TODO"""
        ...