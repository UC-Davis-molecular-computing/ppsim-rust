"""
ppsim: A Python package with Rust backend for simulation
"""

# import ppsim.crn
# import ppsim.snapshot
# import ppsim.simulation

# Re-export everything from Python modules
from ppsim.simulation import *
from ppsim.snapshot import *
from ppsim.crn import *
# from ppsim.crn import Reaction, reactions_to_dict, Specie, Expression
# from ppsim.snapshot import Snapshot, TimeUpdate
# from ppsim.simulation import time_trials

import numpy as np
import numpy.typing as npt

RustState: TypeAlias = int


class SimulatorSequentialArray:
    config: list[RustState]
    n: int
    t: int
    delta: list[list[tuple[RustState, RustState]]]
    null_transitions: list[list[bool]]
    silent: bool
    population: list[RustState]

    def __init__(
            self,
            init_config: npt.NDArray[np.uint],
            delta: npt.NDArray[np.uint],
            null_transitions: npt.NDArray[np.bool_],
            random_transitions: npt.NDArray[np.uint],
            random_outputs: npt.NDArray[np.uint],
            transition_probabilities: npt.NDArray[np.float64],
            seed: int | None = None
    ) -> None: ...

    def make_population(self) -> None: ...

    def run(
            self,
            t_max: int,
            max_wallclock_time: float = 3600.0
    ) -> None: ...

    def run_until_silent(self) -> None: ...

    def reset(
            self,
            config: npt.NDArray[np.uint],
            t: int = 0
    ) -> None: ...

    def write_profile(self, filename: str | None = None) -> None: ...


class SimulatorMultiBatch:
    config: list[RustState]
    n: int
    t: int
    delta: list[list[tuple[RustState, RustState]]]
    null_transitions: list[list[bool]]
    do_gillespie: bool
    silent: bool
    reactions: list[list[RustState]]
    enabled_reactions: list[int]
    num_enabled_reactions: int
    reaction_probabilities: list[float]

    def __init__(
            self,
            init_config: npt.NDArray[np.uint],
            delta: npt.NDArray[np.uint],
            null_transitions: npt.NDArray[np.bool_],
            random_transitions: npt.NDArray[np.uint],
            random_outputs: npt.NDArray[np.uint],
            transition_probabilities: npt.NDArray[np.float64],
            seed: int | None = None
    ) -> None: ...

    def run(
            self,
            t_max: int,
            max_wallclock_time: float = 3600.0
    ) -> None: ...

    def run_until_silent(self) -> None: ...

    def reset(
            self,
            config: npt.NDArray[np.uint],
            t: int = 0
    ) -> None: ...

    def get_enabled_reactions(self) -> None: ...

    def get_total_propensity(self) -> float: ...

    def write_profile(self, filename: str | None = None) -> None: ...