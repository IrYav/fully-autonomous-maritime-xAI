from Simulation.Environment.environment import Environment
from Simulation.Auxilaries.shapely_aux import flatten_polygons

import numpy as np
from shapely.geometry import Polygon, LineString
from dataclasses import dataclass

@dataclass
class ShipState:
  position: np.ndarray  # (2,)
  heading: float # Radian
  velocity: float # m/s

@dataclass
class ShipParams:
  max_curvature: float         # 1 / m, positive
  max_velocity: float          # m / s, positive
  max_reverse_velocity: float  # m / s, positive
  length: float                # m, positive
  width: float                 # m, positive
  tau: float                   # s, positive

@dataclass
class Ship:
  state: ShipState 
  params: ShipParams 

@dataclass
class Action:
  rudder: float # \in -1 to 1
  acceleration: float # \in -1 to 1

@dataclass
class Trajectory:
  s: float        # Distance travelled
  kappa: float    # Curvature
  dtheta: float   # Angle change
  velocity_new: float    # New velocity

@dataclass
class StepResult:
  state: ShipState
  collisions: list[str]

def simulate_step(ship: Ship, action: Action, env: Environment, dt: float) -> StepResult:
  traj = compute_trajectory(ship.state, action, ship.params, dt)
  collisions = check_for_collision(ship, env, traj)
  state_new = apply_trajectory(ship.state, traj)
  return StepResult(state_new, collisions)

def get_ship_polygon(ship: Ship) -> Polygon:
  L = ship.params.length
  W = ship.params.width
  pos = ship.state.position
  h = rot_vect(ship.state.heading)

  p1 = pos - 0.5 * L * h
  p2 = pos + 0.5 * L * h

  return LineString([p1, p2]).buffer(W / 2)
  
def check_for_collision(ship_path: Polygon, env: Environment) -> list[str]:
  '''
    Approximates the steps path, by using a circle arc, with length/2 straight ends.
    Order of results (if existing): `boundary`, then `island_i` (e.g, `island_5`) (as ordered in `env`) and then `end`.
  '''
  result = []

  #Checks for Outside the boundary 
  if not ship_path.within(Polygon(env.boundary.exterior)):
    result.append("boundary")

  #Inside an island (and/or the end)
  obstacles = flatten_polygons(env.islands)
  for i, obstacle in enumerate(obstacles):
    if ship_path.intersects(obstacle):
      result.append(f"island_{i}")
  
  if ship_path.intersects(env.end_poly):
    result.append(f"end")

  return result

def compute_trajectory(state: ShipState, action: Action, params: ShipParams, dt: float) -> Trajectory:
  u = state.velocity

  # --- Clamping actions ---
  r_cmd = np.clip(action.rudder, -1.0, 1.0)
  a_cmd = np.clip(action.acceleration, -1.0, 1.0)

  # --- Target Velocity ---
  if a_cmd >= 0: 
    u_target = a_cmd * params.max_velocity 
  else:
    u_target = a_cmd * params.max_reverse_velocity
    
  # --- Relaxationsmodell ---
  tau = params.tau
  k = 1.0 / tau

  # neue Geschwindigkeit (exakt)
  u_new = u_target + (u - u_target) * np.exp(-k * dt)

  # Integrated distance travelled
  s = (
    u_target * dt +
    (u - u_target) * (1.0 - np.exp(-k * dt)) / k
  )

  kappa = r_cmd * params.max_curvature

  # Winkeländerung
  dtheta = kappa * s
  return Trajectory(s=s, kappa=kappa, dtheta=dtheta, velocity_new=u_new)

def apply_trajectory(state: ShipState, traj: Trajectory) -> ShipState:
  pos = state.position
  theta = state.heading

  if abs(traj.kappa) < 1e-8:
    # Gerade
    dx = traj.s * np.cos(theta)
    dy = traj.s * np.sin(theta)

    pos_new = pos + np.array([dx, dy])
    theta_new = theta

  else:
    # Kreis
    kappa = traj.kappa
    dtheta = traj.dtheta

    h = rot_vect(theta)
    n = np.array([-h[1], h[0]])

    center = pos + (1.0 / kappa) * n

    rel = pos - center

    c, s = np.cos(dtheta), np.sin(dtheta)
    Rm = np.array([[c, -s],
                   [s,  c]])

    pos_new = center + Rm @ rel
    theta_new = theta + dtheta

  return ShipState(position=pos_new, heading=theta_new, velocity=traj.velocity_new)

def get_start_state(env: Environment, start_vel = 0) -> ShipState:
  '''
    Get the starting position, heading from env and create the according `ShipState` instance
  '''
  pos = env.start
  end = np.array(env.end_poly.centroid.coords[0])
  d = end - pos
  heading = np.arctan2(d[1], d[0])

  return ShipState(pos, heading, start_vel)

def get_path(state: ShipState, traj: Trajectory, params: ShipParams, n_samples: int = 20) -> Polygon:
  '''
    Returns a polygon approximating the union over all ship positions along a given trajectory.
    Rather than a proper union of all areas over each (positions, heading), 
    this approach uses a buffered skeleton (circle arch + straight line segments for start and end) 
  '''
  pos0 = state.position
  th0 = state.heading
  L, W = params.length, params.width

  # Edge Case: Ship doesn't move.
  if abs(traj.s) < 1e-8:
    return get_ship_polygon(Ship(state, params))
  # --- Middle Part ---
  if abs(traj.kappa) < 1e-8:
    # Straight Line
    pts = [pos0, pos0 + traj.s * rot_vect(th0)]
  else:
    # Circle Arch
    kappa = traj.kappa
    R = 1.0 / abs(kappa)

    h0 = rot_vect(th0)
    n0 = np.array([-h0[1], h0[0]])
    center = pos0 + (1.0 / kappa) * n0

    rel0 = pos0 - center
    a0 = np.arctan2(rel0[1], rel0[0])
    angles = np.linspace(a0, a0 + traj.dtheta, n_samples)

    pts = [center + R * np.array([np.cos(a), np.sin(a)]) for a in angles]

  arc = LineString(pts)

  # --- Start and End Segments ---
  h0 = rot_vect(th0)
  start_seg = LineString([pos0 - 0.5 * L * h0, pos0 + 0.5 * L * h0])

  end_state = apply_trajectory(state, traj)
  pos1, th1 = end_state.position, end_state.heading
  h1 = rot_vect(th1)
  end_seg = LineString([pos1 - 0.5 * L * h1, pos1 + 0.5 * L * h1])

  skeleton = arc.union(start_seg).union(end_seg)
  return skeleton.buffer(W / 2, quad_segs = 1)
  

def rot_vect(theta:float):
  return np.array([np.cos(theta), np.sin(theta)])