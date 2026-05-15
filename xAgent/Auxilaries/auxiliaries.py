from dataclasses import dataclass

from meshio import Mesh

from xAgent.Environment.agent import Ship
from xAgent.Environment.environment import Environment
from xAgent.Environment.harmonic import HarmonicField

@dataclass
class SimulationState:
  env: Environment
  mesh: Mesh
  field: HarmonicField
  ship: Ship