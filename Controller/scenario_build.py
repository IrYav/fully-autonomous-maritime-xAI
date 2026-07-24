import threading

from Agent.Curriculum.curriculum import Difficulty
from Simulation.Auxilaries.sim_config import SimulationConfig
from Simulation.Environment.harmonic import HarmonicField
from Simulation.Environment.environment import Environment, new_environment
from Simulation.Environment.mesh import build_mesh, BuiltMesh
from Simulation.Environment.mesh_prep import prepare_domain

# gmsh doesn't support multiple meshing instances within a single process.
_GMSH_LOCK = threading.Lock()

def build_scenario_from_env(config: SimulationConfig, env: Environment):
  '''
    Add onto a pre-existing env, creating a full scenario 
  '''
  mesh, field = _mesh_field(config, env)
  return env, mesh, field

def build_scenario(
    config: SimulationConfig, 
    difficulty: Difficulty
    ) -> tuple[Environment, BuiltMesh, HarmonicField]:
  '''
  Build a complete scenario:
    env, mesh, field
  '''

  ship_width = config.ship_params.width
  ship_len = config.ship_params.length
  ship_kappa = config.ship_params.max_curvature
  turning_radius = 1 / config.ship_params.max_curvature

  env = new_environment(
    config.boundary,
    num_islands=difficulty.num_islands,
    end_radius=max(5 * config.h, 1.5 * ship_len),
    corridor_width=2 * ship_width,
    min_border_dist=1.5 * turning_radius,
    max_curvature=ship_kappa,
    scenario=difficulty.scenario,
    start_radius=ship_len * 2
  )

  mesh, field = _mesh_field(config, env)

  return env, mesh, field

def _build_mesh_threadsafe(mesh_raw):
  with _GMSH_LOCK:
    return build_mesh(mesh_raw)

def _mesh_field(config: SimulationConfig, env: Environment):
  mesh_raw = prepare_domain(
    env.islands,
    env.end_poly,
    config.boundary,
    h=config.h
  )

  mesh = _build_mesh_threadsafe(mesh_raw)
  field = HarmonicField(mesh)
  return mesh, field