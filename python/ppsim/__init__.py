"""
ppsim: A Python package with Rust backend for simulation
"""

try:
    from importlib.metadata import version
    __version__ = version("ppsim")
except (ImportError, ModuleNotFoundError):
    __version__ = "0.1.0"  # fallback version

from ppsim.snapshot import *
from ppsim.crn import *
from ppsim.simulation import *
