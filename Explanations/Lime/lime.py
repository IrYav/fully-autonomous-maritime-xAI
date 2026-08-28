from enum import Enum
from pathlib import Path
import numpy as np

from Controller.agentcontroller import AgentController
from Simulation.Environment.agent import Ship
from Simulation.Visualization.RightHalf.ExplTab.lime_view_model import Source, StepValueClass
from ThirdParty.lime.lime.lime_tabular import LimeTabularExplainer

from Agent.Observation.observation import feature_names
from Simulation.Auxilaries.io import load_runs
from Simulation.Auxilaries.sim_config import sim_config


class LimeExplainer():
  ac : AgentController
  explainer : LimeTabularExplainer

  def __init__(self, ac: AgentController):
    self.ac = ac
    #The explainer itself does not depend on the (future) instance.
    self.explainer = self.calculate_explainer()

  def calculate_explainer(self) -> LimeTabularExplainer:
    '''
      Learns and returns a LIME explainer.
    '''
    samples_path = Path("Explanations/Samples")
    steps = np.concatenate(
      [
        np.asarray(run.steps, dtype=object)
        for run in load_runs(samples_path, sim_config)
      ],
      axis=0,
    )
    print(steps.shape)
    train = np.stack(steps[:, 0])      
    print(train.shape)

    explainer = LimeTabularExplainer(
      train, 
      feature_names= feature_names(pretty=True),
      mode="regression",
    )
    return explainer

  def get_samples(self,
                 ship: Ship,
                 size_x: float, size_y: float,
                 num_x: int, num_y: int, num_thetas: int):
    '''
      For x (and analogously y):
      Setup grid calculates the positions, 
      in a equidistant grid of num_x points,
      from pos_x - size_x to pos_x + size_x.
      The ship will always receive a point.
      Ideally use odd `num_x` and `num_y`. 
      For an even number of samples s, it will use first s + 1 samples, 
      before discarding the least sample, hencing leaving a slightly asymetrical
      sample intervall. 

      For the angle, it samples equidistant points around [0; 2 * pi].
      The angle samples are offset by the ships angle (with wrap-around), 
      such that the exact ship state is always sampled.
      `num_thetas` should be ideally even, 
      in order to get the ship heading + pi.


      As we produce `num_x * num_y * num_thetas` many points, 
      runtime scales with `num_x * num_y * num_thetas`.

      num_x and num_y need to be at least 3, num_thetas at least 1.
      Returns:
      positions : ndarray, shape (N, 2)
          Spatial sampling locations.

      theta_values : ndarray, shape (T,)
          Discrete heading samples.

      The sampled state space is the Cartesian product
      positions × theta_values, containing N*T states.
    '''
    if num_x < 3 or num_y < 3:
      raise ValueError("num_x and num_y must be at least 3")
    if num_thetas < 1:
      raise ValueError("num_thetas must be at least 1")

    def axis_values(position: float, size: float, num: int) -> np.ndarray:
      # An odd-sized grid is symmetric around the ship.  For an even-sized grid, 
      # the negative boundary is discarded to preserve the ship position and 
      # uniform spacing. The resulting grid is intentionally asymmetric by one sample.
      # This keeps the spacing uniform while retaining the ship's position.
      values = np.linspace(position - size, position + size, num + (num % 2 == 0))
      if num % 2 == 0:
        values = values[1:]
      return values

    x_values = axis_values(ship.state.position[0], size_x, num_x)
    y_values = axis_values(ship.state.position[1], size_y, num_y)

    x_grid, y_grid = np.meshgrid(x_values, y_values, indexing="xy")
    positions = np.column_stack((x_grid.ravel(), y_grid.ravel()))

    theta_values = (
      ship.state.heading
      + np.arange(num_thetas, dtype=float) * (2 * np.pi / num_thetas)
    ) % (2 * np.pi)

    return positions, theta_values

  def evaluate_step_samples(self, 
    positions: np.ndarray, #(N, 2)
    thetas: np.ndarray, #(N,)

  ):
    return

  def n_ghost_steps(self, 
    positions: np.ndarray, #(N, 2)
    thetas: np.ndarray, #(N,)
  ):
    N = len(positions)
    T = len(thetas)
    S = len(Source)
    C = len(StepValueClass)

    if N != T:
      raise ValueError(
        "Positions, and theta need the same length: "
        f"{N} != {T}")
    
    values = np.empty((N, T, S, C), dtype=np.float32)

    for pos, theta in zip(positions,thetas):
      #values[pos,theta,source,value_enum] = v
      break
    return

  def get_lime_step_fns(self):
    '''
      Returns the three predict_fns that map a 12d state 
      (plus the `self.ac` image vectory) to the three instances
      of `StepValueClass`. Or in otherwords:
      get_lime_step_fns(self)[0](state) = heading
      get_lime_step_fns(self)[1](state) = throttle
      get_lime_step_fns(self)[2](state) = reward
    '''

  def set_manipulated_obs():
    '''
      Soll den 12d vektor korrekt aus 3d ableiten.
      Konsistenzchecks sind importante, e.g. pos -> rel pos
    '''

  def get_all_explainers(self, state):
    '''
    get_all_explainers
    Ideally returns 3 funcs 12d -> 1d

    Probably needs some way to access surrogate 
    or at least its 13 weights (flat bias + 12 linear weights)

    Instanz erhalten, braucht vektor eval. 
    Predict easy, aber wie reward?
    Reward vermutlich stumpf über ghost steps.
    '''
    NUM_FEATURES = 5
    NUM_SAMPLES = 5000
    exp_rudder = self.explainer.explain_instance(
      state,
      predict_fn=lambda X: self.ac.states_predict(X)[:, 0],
      num_features=NUM_FEATURES, num_samples=NUM_SAMPLES,
    )

    exp_throttle = self.explainer.explain_instance(
      state,
      predict_fn=lambda X: self.ac.states_predict(X)[:, 1],
      num_features=NUM_FEATURES, num_samples=NUM_SAMPLES,
    )

    exp_lreward = self.explainer.explain_instance(
      state,
      predict_fn=lambda X: self.ac.states_predict(X)[:, 1],
      num_features=NUM_FEATURES, num_samples=NUM_SAMPLES,
    )

  def ghost_steps(self, obs):
    '''
      Maps a (n, ) 12D observations (c.f.  build_observation), using the self.ac image
      stacks to n local rewards.
    '''
    for ob in obs:
      self.ac.model_step(True)

  def setup_ground_truth():
    
    #Probably vectors (x,y,[step res], [run res])

    #Sample around the ship (some kind of vicinity)

    #Directly collect all relevant sizes.
      #local reward <- ac.model_step(True)["reward"]
      #rudder <- ac.model_step(True)["action"][0]
      #velocity <- ac.model_step(True)["action"][1]
      #Reward Delta <- Small auxilary, ac.finish_episode(True)[0] - base_reward
      #Step Count to goal <- ac.finish_episode(True)[1]
      #Collisions (Categorical) <- ac.finish_episode(True)[2] but modified, 
        # Four! categories. normal 3, plus time-out ()

    return
