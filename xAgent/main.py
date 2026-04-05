#External
import numpy as np
import sys
from PySide6.QtWidgets import QApplication

#Internal
import xAgent.Environment.environment as envi
import xAgent.Environment.mesh as mesher
import xAgent.Visualization.visEnvironment as vis


# 
boundary = np.array([[5, -50], [150, 100]], dtype=float)
n_points = 5


#Generate agent.
#Train agent.
#Visualize
#Modify
#\Delta-Lime
#SP-\Delta-Lime
#Probe
#Verify Probe
def main():
  app = QApplication(sys.argv)
  viewer = vis.EnvironmentViewer(boundary)

  viewer.add_button("New Instance", newTrainingInstance)

  viewer.show()

  newTrainingInstance(viewer)

  sys.exit(app.exec())
  return 

def newTrainingInstance(viewer):
  env = envi.new_environment(boundary)
  mesh = mesher.build_mesh(env.islands, boundary)
  viewer.set_state(env, mesh)

def renderNew():
  return

main()