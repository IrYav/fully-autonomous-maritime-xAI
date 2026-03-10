from xAgent.Environment.environment import generate_points, projection_sort, interpolate, get_corridor, generate_islands

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, CheckButtons
from shapely.plotting import plot_polygon
import shapely.validation as val
# --- Configuration ---
boundary = np.array([[5, -50], [150, 100]], dtype=float)
n_points = 5
u_eval = np.linspace(0.0, 1.0, 300)


fig, ax = plt.subplots()

def main():
  newInstance()
  # --- Button ---
  button_ax = plt.axes([0.4, 0.05, 0.2, 0.075])
  button = Button(button_ax, "New Instance")
  button.on_clicked(update)
  plt.sca(ax)
  print("Showing in:", ax)
  plt.show()
# --- Pipeline (single source of truth) ---
def newInstance():
  #View and generation box
  scale = boundary[1] - boundary[0]
  center = (boundary[0] + boundary[1]) / 2
  halfsize = 0.75 * scale

  xmin, ymin = center - halfsize
  xmax, ymax = center + halfsize

  #Calculate new graphs
  points = generate_points(n_points, boundary) 
  points = projection_sort(points)
  spline = interpolate(points)
  curve = spline(u_eval)
  corridor = get_corridor(spline, 500, 3.5)
  islands, curve = generate_islands(6,corridor, boundary * 1.5)
  #Plot new graphs
  ax.clear()
  #pointsRef = ax.plot(points[:, 0], points[:, 1])
  #splineRef = ax.plot(curve[:, 0], curve[:, 1])

  plot_polygon(corridor, ax=ax, color= "lightgray", add_points=False)
  plot_polygon(islands,  ax=ax, color= "lightgreen", add_points=False,)
  ax.set_aspect('equal', adjustable='box')
  ax.set_xlabel("Distance (100 m)")
  ax.set_ylabel("Distance (100 m)")
  #ax.plot(curve[:,0], curve[:,1], color="red", linewidth=2)
  #ax.legend(["Control polygon", "Spline"])
  ax.set_xlim(xmin, xmax)
  ax.set_ylim(ymin, ymax)
  fig.canvas.draw_idle()
  return
  
# --- Update callback ---
def update(event):
  newInstance()

if __name__ == "__main__":
  main()