from xAgent.Environment.environment import generate_points, projection_sort, interpolate, get_corridor, generate_islands

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, CheckButtons
from shapely.plotting import plot_polygon

# --- Configuration ---
boundary = np.array([[5, -50], [150, 100]], dtype=float)
n_points = 5
u_eval = np.linspace(0.0, 1.0, 300)


fig, ax = plt.subplots()  


def main():

  # --- Button ---
  button_ax = plt.axes([0.4, 0.05, 0.2, 0.075])
  button = Button(button_ax, "New Spline")
  button.on_clicked(update)

  newInstance()
  plt.show()
# --- Pipeline (single source of truth) ---
def newInstance():
  #Calculate new graphs
  points = generate_points(n_points, boundary) 
  points = projection_sort(points)
  spline = interpolate(points)
  curve = spline(u_eval)
  corridor = get_corridor(spline, 500, 7.5)
  islands = generate_islands(6,corridor, boundary)

  #Plot new graphs
  ax.clear()

  pointsRef = ax.plot(points[:, 0], points[:, 1])
  splineRef = ax.plot(curve[:, 0], curve[:, 1])

  plot_polygon(corridor, ax=ax, color= "lightgray", add_points=False)
  plot_polygon(islands,  ax=ax, color= "lightgreen", add_points=False,)
  
  ax.set_aspect('equal', adjustable='box')
  ax.legend(["Control polygon", "Spline"])

  ax.relim()
  ax.autoscale_view()
  fig.canvas.draw_idle()
  return
  
# --- Update callback ---
def update(event):
  newInstance()

if __name__ == "__main__":
  main()