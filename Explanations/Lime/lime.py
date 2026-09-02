from enum import Enum
from pathlib import Path
import numpy as np

from Controller.agentcontroller import AgentController
from Simulation.Environment.agent import Ship
from Explanations.Lime.lime_model import (
  RunValueClass, Source, StepValueClass, ValueField,
)
from ThirdParty.lime.lime.lime_tabular import LimeTabularExplainer

from Agent.Observation.observation import feature_names
from Simulation.Auxilaries.io import load_runs
from Simulation.Auxilaries.sim_config import sim_config


class LimeExplainer():
  ac : AgentController
  explainer : LimeTabularExplainer

  def __init__(self, ac: AgentController):
    self.ac = ac
    self._explainer_cache = {}
    # Keep the default regression explainer available for compatibility.
    self.explainer = self.get_explainer()

  def get_explainer(
      self, mode: str = "regression", class_names: list[str] | None = None,
      discretize_continuous: bool = True,
  ) -> LimeTabularExplainer:
    '''
      Return a cached LIME explainer, creating it on first request.
    '''
    cache_key = (
      mode, tuple(class_names or ()), bool(discretize_continuous),
    )
    if cache_key in self._explainer_cache:
      return self._explainer_cache[cache_key]

    samples_path = Path("Explanations/Samples")
    steps = np.concatenate(
      [
        np.asarray(run.steps, dtype=object)
        for run in load_runs(samples_path, sim_config)
      ],
      axis=0,
    )
    train = np.stack(steps[:, 0])[:, [0, 1, 2, 3, 4, 5, 10, 11]]

    explainer = LimeTabularExplainer(
      train, 
      feature_names=feature_names(pretty=True, reduced=True),
      mode=mode,
      class_names=class_names,
      discretize_continuous=discretize_continuous,
    )
    self._explainer_cache[cache_key] = explainer
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

  def get_sample_states(
      self,
      ship: Ship,
      size_x: float, size_y: float,
      num_x: int, num_y: int, num_thetas: int,
  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''Return the sample grid and its corresponding normalized states.'''
    positions, theta_values = self.get_samples(
      ship, size_x, size_y, num_x, num_y, num_thetas,
    )
    base_state = self.ac.store_base_state()
    map_min = np.asarray(self.ac.config.boundary[0], dtype=float)
    map_size = np.asarray(self.ac.config.boundary[1], dtype=float) - map_min

    states = np.repeat(
      np.asarray([
        ((position - map_min) / map_size) * 2 - 1
        for position in positions
      ], dtype=float),
      len(theta_values),
      axis=0,
    )
    states = np.column_stack([
      states,
      np.tile(
        np.array([
          base_state.heading_cos,
          base_state.heading_sin,
          base_state.velocity,
          base_state.time_elapsed,
          base_state.end_x,
          base_state.end_y,
        ], dtype=float),
        (len(positions) * len(theta_values), 1),
      ),
    ])
    states[:, 2] = np.cos(np.tile(theta_values, len(positions)))
    states[:, 3] = np.sin(np.tile(theta_values, len(positions)))
    return positions, theta_values, states.reshape(
      len(positions), len(theta_values), 8,
    )

  def evaluate_grid(
      self,
      explanation,
      target: StepValueClass | RunValueClass,
      ship: Ship,
      size_x: float, size_y: float,
      num_x: int, num_y: int, num_thetas: int,
      label: int | None = None,
      discretize_continuous: bool | None = None,
  ) -> ValueField:
    '''Evaluate truth, surrogate, and signed error over a sample grid.'''
    positions, theta_values, states = self.get_sample_states(
      ship, size_x, size_y, num_x, num_y, num_thetas,
    )
    flat_states = states.reshape(-1, 8)

    if isinstance(target, StepValueClass):
      column = {
        StepValueClass.HEADING: 0,
        StepValueClass.THROTTLE: 1,
        StepValueClass.LREWARD: 2,
      }[target]
      truth = self.ac.eval_step_perturbation_array(flat_states)[:, column]
      surrogate = self.evaluate_surrogate(
        explanation, flat_states, discretize_continuous=discretize_continuous,
      )
      value_enum = StepValueClass
    elif isinstance(target, RunValueClass):
      value_enum = RunValueClass
      if target is RunValueClass.COLLISIONS:
        if label is None:
          labels = explanation.available_labels()
          if not labels:
            raise ValueError("The classification explanation has no labels.")
          label = labels[0]
        truth = self.ac.eval_run_collision_probability_array(flat_states)[:, label]
        surrogate = self.evaluate_surrogate(
          explanation, flat_states, label,
          discretize_continuous=discretize_continuous,
        )
      else:
        column = {
          RunValueClass.REWARD_DELTA: 0,
          RunValueClass.STEPCOUNT: 1,
        }[target]
        truth = self.ac.eval_run_perturbation_array(flat_states)[:, column]
        surrogate = self.evaluate_surrogate(
          explanation, flat_states, discretize_continuous=discretize_continuous,
        )
    else:
      raise TypeError(f"Unsupported target type: {type(target).__name__}")

    shape = (len(positions), len(theta_values))
    truth = truth.reshape(shape)
    surrogate = surrogate.reshape(shape)
    values = np.full((
      *shape, len(Source), len(value_enum),
    ), np.nan, dtype=np.float32)
    source_index = {source: i for i, source in enumerate(Source)}
    target_index = list(value_enum).index(target)
    values[:, :, source_index[Source.GROUND_TRUTH], target_index] = truth
    values[:, :, source_index[Source.LIME], target_index] = surrogate
    values[:, :, source_index[Source.ERROR], target_index] = truth - surrogate
    return ValueField(positions, theta_values, values)

  def plot_grid(
      self,
      field: ValueField,
      target: StepValueClass | RunValueClass,
      source: Source = Source.LIME,
      theta_index: int = 0,
      ax=None,
  ):
    '''Draw one source/target slice of a grid as a heatmap.'''
    import matplotlib.pyplot as plt

    if theta_index < 0 or theta_index >= len(field.theta_values):
      raise IndexError(f"Theta index out of range: {theta_index}")

    value_enum = StepValueClass if isinstance(target, StepValueClass) else RunValueClass
    target_index = list(value_enum).index(target)
    source_index = list(Source).index(source)
    x_values = np.unique(field.positions[:, 0])
    y_values = np.unique(field.positions[:, 1])
    if len(x_values) * len(y_values) != len(field.positions):
      raise ValueError("Grid positions do not form a rectangular grid.")

    heatmap = field.values[:, :, source_index, target_index]
    heatmap = heatmap[:, theta_index].reshape(len(y_values), len(x_values))
    if ax is None:
      _, ax = plt.subplots()
    image = ax.pcolormesh(x_values, y_values, heatmap, shading="auto")
    ax.set_xlabel("ship x")
    ax.set_ylabel("ship y")
    ax.set_title(f"{source.name}: {target.name}, theta={field.theta_values[theta_index]:.3f}")
    ax.figure.colorbar(image, ax=ax)
    return ax

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

  def get_explainer_instance(
        self,
        base_state: np.ndarray, #(8,) 
        target: StepValueClass | RunValueClass, 
        num_samples = 500,
        discretize_continuous: bool = True,
      ):
    '''
      Get an explainer for the selected `target`, given the
      `base_state`
    '''
    if isinstance(target, StepValueClass):
      predict_column = {
        StepValueClass.HEADING: 0,
        StepValueClass.THROTTLE: 1,
        StepValueClass.LREWARD: 2,
      }[target]
      predict_fn = lambda X: self.ac.eval_step_perturbation_array(X)[:, predict_column]
      explainer = self.get_explainer(
        discretize_continuous=discretize_continuous,
      )
    elif isinstance(target, RunValueClass):
      if target is RunValueClass.REWARD_DELTA:
        predict_fn = lambda X: self.ac.eval_run_perturbation_array(X)[:, 0]
        explainer = self.get_explainer(
          discretize_continuous=discretize_continuous,
        )
      elif target is RunValueClass.STEPCOUNT:
        predict_fn = lambda X: self.ac.eval_run_perturbation_array(X)[:, 1]
        explainer = self.get_explainer(
          discretize_continuous=discretize_continuous,
        )
      elif target is RunValueClass.COLLISIONS:
        predict_fn = self.ac.eval_run_collision_probability_array
        explainer = self.get_explainer(
          mode="classification",
          class_names=["end", "island", "boundary", "time_out"],
          discretize_continuous=discretize_continuous,
        )
      else:
        raise ValueError(f"Unsupported run target: {target}")
    else:
      raise TypeError(f"Unsupported target type: {type(target).__name__}")

    explanation = explainer.explain_instance(
      np.asarray(base_state, dtype=float),
      predict_fn=predict_fn,
      num_features=8,
      num_samples=num_samples, sampling_method="lhs"
    )
    # Explanation does not retain a reference to its LimeTabularExplainer.
    # Keep the setting here so downstream grid evaluation can use the exact
    # feature representation used by this explanation.
    explanation._discretize_continuous = discretize_continuous
    return explanation

  def evaluate_surrogate(
      self, explanation, data: np.ndarray, label: int | None = None,
      discretize_continuous: bool | None = None,
  ) -> np.ndarray:
    '''Evaluate a calculated LIME surrogate on one or more states.

    LIME fits its linear model on standardized feature values. This method
    applies the matching cached explainer scaler before evaluating
    ``intercept + sum(weight * feature)``.
    '''
    data = np.asarray(data, dtype=float)
    if data.ndim == 1:
      data = data.reshape(1, -1)
    if data.ndim != 2 or data.shape[1] != 8:
      raise ValueError(f"Expected data with shape (n, 8), got {data.shape}.")

    if discretize_continuous is None:
      discretize_continuous = getattr(
        explanation, "_discretize_continuous",
        explanation.domain_mapper.discretized_feature_names is not None,
      )

    if explanation.mode == "regression":
      lime_explainer = self.get_explainer(
        discretize_continuous=discretize_continuous,
      )
      label = 1 if label is None else label
    elif explanation.mode == "classification":
      lime_explainer = self.get_explainer(
        mode="classification",
        class_names=["end", "island", "boundary", "time_out"],
        discretize_continuous=discretize_continuous,
      )
      if label is None:
        labels = explanation.available_labels()
        if not labels:
          raise ValueError("The classification explanation has no labels.")
        label = labels[0]
    else:
      raise ValueError(f"Unsupported explanation mode: {explanation.mode}")

    if label not in explanation.local_exp:
      raise ValueError(f"The explanation has no surrogate for label {label}.")

    if discretize_continuous:
      if lime_explainer.discretizer is None:
        raise ValueError(
          "The explanation is marked as discretized, but its explainer "
          "has no discretizer."
        )

      # LIME changes every continuous feature into a categorical feature
      # after binning. Its local data is therefore 1 iff the value belongs to
      # the same bin as the explained instance, rather than the bin number or
      # a standardized raw value.
      discretized_data = lime_explainer.discretizer.discretize(data)
      names = explanation.domain_mapper.discretized_feature_names
      if names is None:
        raise ValueError(
          "Cannot recover the reference bins from the explanation."
        )
      base_bins = np.empty(data.shape[1], dtype=int)
      for feature in range(data.shape[1]):
        bin_names = lime_explainer.discretizer.names.get(feature)
        if bin_names is None:
          base_bins[feature] = data[0, feature]
          continue
        try:
          base_bins[feature] = bin_names.index(names[feature])
        except (KeyError, ValueError) as error:
          raise ValueError(
            f"Could not recover the reference bin for feature {feature}."
          ) from error
      scaled_data = (discretized_data == base_bins).astype(float)
    else:
      scaled_data = (
        data - lime_explainer.scaler.mean_
      ) / lime_explainer.scaler.scale_
    prediction = np.full(
      len(data), float(explanation.intercept[label]), dtype=float
    )
    for feature, weight in explanation.local_exp[label]:
      prediction += float(weight) * scaled_data[:, feature]
    return prediction
  
  def ghost_steps(self, obs):
    '''
      Maps a (n, ) 12D observations (c.f.  build_observation), using the self.ac image
      stacks to n local rewards.
    '''
    for ob in obs:
      self.ac.model_step(True)
