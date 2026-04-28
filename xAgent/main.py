#External
import numpy as np
import sys
from PySide6.QtWidgets import QApplication

#Internal
from xAgent.Environment.harmonic import HarmonicField
import xAgent.Environment.environment as envi
import xAgent.Environment.mesh_prep as mesher_prep
import xAgent.Environment.mesh as mesher
import xAgent.Visualization.visEnvironment as vis

#Rectangle for agent.
boundary = np.array([[-800, -800], [-600, -600]], dtype=float)
h = 5#np.max(boundary[1] - boundary[0]) / 40 # Mesh Resolution
eps = 0.5 # Affects island simplification before mesher input. Higher means simpler.

def main():
  # Initialize
  app = QApplication(sys.argv)
  viewer = vis.EnvironmentViewer(boundary)

  # Buttons, Toggleboxes, etc.
  viewer.add_button("New Instance", newTrainingInstance)
  viewer.show()

  newTrainingInstance(viewer)

  sys.exit(app.exec())
  return 

def newTrainingInstance(viewer):
  env = envi.new_environment(boundary, num_islands=15, end_radius= 5 * h)
  prepped_mesh = mesher_prep.prepare_domain(env.islands, env.end_poly, boundary, h=h, eps = eps)
  mesh = mesher.build_mesh(prepped_mesh)
  bc = HarmonicField.get_boundary_conditions(mesh)
  field = HarmonicField(mesh, bc)

  viewer.set_state(env, mesh, field)
  
main()