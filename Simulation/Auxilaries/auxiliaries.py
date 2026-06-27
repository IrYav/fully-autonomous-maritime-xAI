from dataclasses import dataclass

from Simulation.Environment.agent import Ship
from Simulation.Environment.environment import Environment
from Simulation.Environment.harmonic import HarmonicField
from Simulation.Environment.mesh import BuiltMesh

@dataclass
class SimulationState:
  env: Environment
  mesh: BuiltMesh
  field: HarmonicField
  ship: Ship

