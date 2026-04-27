import numpy as np
import matplotlib.tri as mtri
from matplotlib.figure import Figure

from PySide6.QtWidgets import QApplication, QHBoxLayout, QWidget, QVBoxLayout, QPushButton
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as Canvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from shapely.plotting import plot_polygon, plot_line


class EnvironmentViewer(QWidget):
  def __init__(self, boundary):
    super().__init__()

    # State
    self.env = None
    self.mesh = None
    self.harfield = None

    # Artists
    self.artists = {
      "corridor": None,
      "islands": None,
      "mesh": None,
      "potential_contour": None,
      "potential_fill": None,
      "gradient": None
    }

    # Matplotlib Figure
    self.fig = Figure()
    self.ax = self.fig.add_subplot(111)
    self.canvas = Canvas(self.fig)

    self.toolbar = NavigationToolbar2QT(self.canvas, self)

    layout = QVBoxLayout()
    layout.addWidget(self.toolbar)
    layout.addWidget(self.canvas)

    # eigener Bereich für Buttons
    self.button_layout = QHBoxLayout()  # oder QVBoxLayout
    layout.addLayout(self.button_layout)

    self.fig.canvas.mpl_connect('button_press_event', self.on_click)
    self.setLayout(layout)
  # ---------- Public API ----------
  def set_state(self, env, mesh, harmonic_field):
    self.env = env
    self.mesh = mesh
    self.harfield = harmonic_field
    self._render()

  def add_button(self, label, callback):
    btn = QPushButton(label)
    btn.clicked.connect(lambda *_: callback(self))
    self.button_layout.addWidget(btn)
    return btn

  # ---------- Rendering ----------
  def _render(self):
    if self.env is None or self.mesh is None:
      return

    self._draw_environment()
    self._draw_mesh()
    self._draw_potential()
    
    self.canvas.draw()

  def _draw_environment(self):
    self._remove_artist("corridor")
    self._remove_artist("islands")

    self.artists["corridor"] = plot_polygon(
      self.env.corridor,
      ax=self.ax,
      color="lightgray",
      add_points=False
    )

    self.artists["islands"] = plot_polygon(
      self.env.islands,
      ax=self.ax,
      color="lightgreen",
      add_points=False
    )

    self.ax.set_aspect('equal', adjustable='box')
    self.ax.set_xlabel("Distance (100 m)")
    self.ax.set_ylabel("Distance (100 m)")

  def _draw_mesh(self):
    self._remove_artist("mesh")

    points = self.mesh.points[:, :2]
    triangles = self.mesh.cells_dict["triangle"]

    triang = mtri.Triangulation(
      points[:, 0],
      points[:, 1],
      triangles
    )

    self.artists["mesh"] = self.ax.triplot(triang, linewidth=0.5, color="gray")
   
  def _draw_potential(self, stride=3):

    tri = self.harfield.tri
    phi = self.harfield.u
    # print(np.finfo(phi.dtype).bits)
    # eps = 1e-12
    # delta = np.clip(1 - phi, eps, 1.0)
    # phi = -np.log10(delta)
    grad = self.harfield.grad

    self._remove_artist("potential_contour")
    self._remove_artist("potential_fill")
    self._remove_artist("gradient")

    #Contour
    contour = self.ax.tricontour(tri, phi, levels=30)
    self.artists["potential_contour"] = contour

    #Filled contour (optional)
    fill = self.ax.tricontourf(tri, phi, levels=30, alpha=0.3)
    self.artists["potential_fill"] = fill

    #Zentroiden
    triangles = tri.triangles
    cx = np.mean(tri.x[triangles], axis=1)
    cy = np.mean(tri.y[triangles], axis=1)

    #Gradient (downsample)
    cx = cx[::stride]
    cy = cy[::stride]
    g = grad[:, ::stride]

    dx = g[0]
    dy = g[1]

    #Normalize ()
    g = np.vstack((dx, dy))           # shape (2, N)
    norm = np.hypot(dx, dy)           # shape (N,)

    g_norm = np.divide(
      g,
      norm,
      out=np.zeros_like(g),
      where=norm > 0
    )

    #Now normalized
    dx = g_norm[0]
    dy = g_norm[1]

    #Quiver
    quiv = self.ax.quiver(cx, cy, dx, dy, scale=30)
    self.artists["gradient"] = quiv

  # ---------- Utilities ----------
  def _remove_artist(self, name):
    artist = self.artists.get(name)

    if artist is None:
      return

    # ContourSet (Multiple Collections)
    if hasattr(artist, "collections"):
      for coll in artist.collections:
        coll.remove()
    elif isinstance(artist, list):
      for a in artist:
        a.remove()
    else:
      artist.remove()

    self.artists[name] = None

  def inspect_triangle(self, pos):
    tri_id = self.harfield.finder(*pos)
    if tri_id == -1:
      print("Outside mesh")
      return

    tri = self.harfield.mesh.t[:, tri_id]     # 3 node indices
    coords = self.harfield.mesh.p[:, tri]     # (2,3)
    values = self.harfield.u[tri]

    print(f"Triangle ID: {tri_id}")
    print("Nodes:", tri)
    print("Coordinates:\n", coords.T)
    print("Values u:", values)

    # Differences
    diffs = [
      abs(values[i] - values[j])
      for i in range(3)
      for j in range(i+1, 3)
    ]
    print("Value differences:", diffs)

    # Gradient
    grad = self.harfield.grad[:, tri_id]
    norm = np.linalg.norm(grad)

    print("Gradient:", grad)
    print("Gradient norm:", norm)
    if norm != 0:
      print("Gradient (normalized):", grad/norm)
    else:
      print("Gradient (normalized): n/a (norm == 0)")
    # Log Scale
    u_safe = np.clip(values, 1e-300, None)
    print("u in power of 10: 10^", np.log10(u_safe))

    # --- Edge labels ---
    print("Edge labels:")

    edges = [
      (tri[0], tri[1]),
      (tri[1], tri[2]),
      (tri[2], tri[0]),
    ]

    for i, j in edges:
      key = frozenset((int(i), int(j)))
      label = self.harfield.edge_labels.get(key, None)

      #Doesn't belong to border part of the domain.
      if label is None:
        label = "interior"

      print(f"  Edge ({i}, {j}): {label}")
  
  def on_click(self, event):
    if event.inaxes != self.ax:
      return

    if event.xdata is None or event.ydata is None:
      return

    pos = np.array([event.xdata, event.ydata])

    tri_id = self.harfield.finder(*pos)
    if tri_id == -1:
      print("Outside mesh")
      return

    print("\n--- CLICK ---")
    self.inspect_triangle(pos)

    # optional: visuelles Highlight
    tri_nodes = self.harfield.mesh.t[:, tri_id]
    coords = self.harfield.mesh.p[:, tri_nodes]

    self.ax.scatter(coords[0], coords[1], color='red')
    self.fig.canvas.draw()