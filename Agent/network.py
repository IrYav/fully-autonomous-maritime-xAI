# network.py

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class NetworkConfig:
  """
  image_size:
    Width/height of the square local map input.

  vector_size:
    Number of scalar auxiliary inputs.

  num_actions:
    Number of continuous action outputs.
  """

  image_size: int
  vector_size: int
  num_actions: int

  cnn_channels: tuple[int, ...] = (16, 32, 64)
  mlp_hidden: tuple[int, ...] = (128, 128)


class MaritimeCNN(nn.Module):
  """
  Very early/simple navigation network.

  Inputs:
    image:
      Local ego-centric map around ship.
      Shape:
        (B, C, H, W)

      Expected channels for now:
        channel 0: islands / obstacles
        channel 1: goal marker (optional)

    vector:
      Auxiliary state vector.
      Shape:
        (B, vector_size)

      Suggested contents:
        [
          target_cos,
          target_sin,

          target_dx,
          target_dy,

          velocity,
        ]

  Outputs:
    action tensor:
      Shape:
        (B, num_actions)

      Interpretation intentionally left to
      training/environment layer.
  """

  def __init__(self, config: NetworkConfig):
    super().__init__()

    self.config = config

    self.cnn = self._build_cnn()
    self.cnn_out_features = self._infer_cnn_features()

    mlp_input = self.cnn_out_features + config.vector_size

    self.mlp = self._build_mlp(
      mlp_input,
      config.mlp_hidden,
      config.num_actions,
    )

  def forward(
    self,
    image: torch.Tensor,
    vector: torch.Tensor,
  ) -> torch.Tensor:

    x_img = self.cnn(image)
    x_img = torch.flatten(x_img, start_dim=1)

    x = torch.cat([x_img, vector], dim=1)

    return self.mlp(x)

  def _build_cnn(self) -> nn.Sequential:
    layers: list[nn.Module] = []

    in_channels = 1

    for out_channels in self.config.cnn_channels:
      layers.extend([
        nn.Conv2d(
          in_channels,
          out_channels,
          kernel_size=3,
          stride=2,
          padding=1,
        ),
        nn.ReLU(),
      ])

      in_channels = out_channels

    return nn.Sequential(*layers)

  def _build_mlp(
    self,
    input_size: int,
    hidden_sizes: tuple[int, ...],
    output_size: int,
  ) -> nn.Sequential:

    layers: list[nn.Module] = []

    current_size = input_size

    for hidden_size in hidden_sizes:
      layers.extend([
        nn.Linear(current_size, hidden_size),
        nn.ReLU(),
      ])

      current_size = hidden_size

    layers.append(nn.Linear(current_size, output_size))

    return nn.Sequential(*layers)

  def _infer_cnn_features(self) -> int:
    """
    Runs a dummy tensor through the CNN to infer
    flattened output size automatically.
    """

    with torch.no_grad():
      dummy = torch.zeros(
        1,
        1,
        self.config.image_size,
        self.config.image_size,
      )

      out = self.cnn(dummy)

    return int(torch.numel(out))

#Some network
# Probably CNN + Crude MLP for now.

#Input local image (centered around agent, rotated so agent faces up):
#  (2 * x)², with x probably something like k * max(turning_radius, dt * max_velocity) 

#Secondary Input: Vector of data points:
# Including (but not limited to), 
# Target Related:
#   \delta angle ((cosθ,sinθ)) between heading and end, 
#   distance (component wise, not sqrt(x² + y²))
# Ship Related:
# Velocity, single number, we only drive up

#Expansion later on, given strategy map:
# -position (x,y), 
# -velocity (x,y)
# -Heading (cosθ,sinθ) (verify gentle wrap around)