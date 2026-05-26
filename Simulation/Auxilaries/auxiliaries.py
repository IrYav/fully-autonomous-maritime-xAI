from dataclasses import dataclass

from meshio import Mesh

from Simulation.Environment.agent import Ship
from Simulation.Environment.environment import Environment
from Simulation.Environment.harmonic import HarmonicField

@dataclass
class SimulationState:
  env: Environment
  mesh: Mesh
  field: HarmonicField
  ship: Ship