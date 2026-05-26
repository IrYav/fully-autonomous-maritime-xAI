import numpy as np
import matplotlib.tri as mtri
from matplotlib.figure import Figure

from PySide6.QtWidgets import QApplication, QCheckBox, QDoubleSpinBox, QFormLayout, QHBoxLayout, QWidget, QVBoxLayout, QPushButton
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as Canvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from shapely.plotting import plot_polygon, plot_line



from Simulation.Auxilaries.auxiliaries import SimulationState
from Simulation.Environment.agent import Action, get_ship_polygon


class EnvironmentViewer(QWidget):
  def __init__(self):
    super().__init__()

    # State
    self.sim_state : SimulationState = None

    # Artists
    self.artists = {
      "corridor": None,
      "islands": None,
      "mesh": None,
      "potential_contour": None,
      "potential_fill": None,
      "gradient": None,
      "ship" : None,
      "trajectory" : None
    }

    # Matplotlib Figure
    self.fig = Figure()
    self.ax = self.fig.add_subplot(111)
    self.canvas = Canvas(self.fig)

    self.toolbar = NavigationToolbar2QT(self.canvas, self)

    # Layout Areas
    self.button_layout = QHBoxLayout()

    self.sidebar_layout = QVBoxLayout()

    viewer_layout = QHBoxLayout()
    viewer_layout.addLayout(self.sidebar_layout,0)
    viewer_layout.addWidget(self.canvas,1)

    main_layout = QVBoxLayout()
    main_layout.addWidget(self.toolbar)
    main_layout.addLayout(viewer_layout)
    main_layout.addLayout(self.button_layout)


    self._setup_visibility_controls(self.sidebar_layout)
    self._setup_inputs(self.sidebar_layout)
    # Push checkboxes up.
    self.sidebar_layout.addStretch()

    self.setLayout(main_layout)
  
  # ---------- Init Helpers --------
  # General Structure, create widget, add it to layout.
  def _setup_inputs(self, layout):
    form = QFormLayout()

    self.rudder_input = QDoubleSpinBox()
    self.rudder_input.setRange(-1.0, 1.0)
    self.rudder_input.setSingleStep(0.1)
    self.rudder_input.setValue(0.0)

    self.accel_input = QDoubleSpinBox()
    self.accel_input.setRange(-1.0, 1.0)
    self.accel_input.setSingleStep(0.1)
    self.accel_input.setValue(0.0)

    self.dt_input = QDoubleSpinBox()
    self.dt_input.setRange(0.01, 100.0)
    self.dt_input.setSingleStep(0.1)
    self.dt_input.setValue(1.0)

    form.addRow("Rudder", self.rudder_input)
    form.addRow("Acceleration", self.accel_input)
    form.addRow("dt", self.dt_input)

    container = QWidget()
    container.setLayout(form)

    layout.addWidget(container)
  
  def _setup_visibility_controls(self, layout):
    self.visibility_boxes = {}

    for name in self.artists.keys():
      box = QCheckBox(name)
      box.setChecked(True)

      box.stateChanged.connect(
        lambda _, n=name: self.toggle_artist(n)
      )

      layout.addWidget(box)
      self.visibility_boxes[name] = box
  # ---------- Public API ----------
  def set_sim_state(self, sim_state : SimulationState):
    self.sim_state = sim_state
    self._render()

  def add_main_button(self, label, callback):
    btn = QPushButton(label)
    btn.clicked.connect(lambda *_: callback())
    self.button_layout.addWidget(btn)
    return btn

  def add_sidebar_button(self, label, callback):
    btn = QPushButton(label)
    btn.clicked.connect(lambda *_: callback())
    self.sidebar_layout.addWidget(btn)
    return btn
  
  def get_sim_input(self):
    return (
      Action(
        self.rudder_input.value(), 
        self.accel_input.value()), 
      self.dt_input.value())
  
  def update_step(self, ship_path):
    self._draw_ship()
    self._draw_trajectory(ship_path)
    self.canvas.draw()

  # ---------- Rendering ----------
  def _render(self):
    if self.sim_state.env is None or self.sim_state.mesh is None:
      return

    self._draw_environment()
    self._draw_mesh()
    self._draw_potential()
    self._draw_ship()
    self.canvas.draw()

  def _draw_environment(self):
    env = self.sim_state.env

    self._remove_artist("corridor")
    self._remove_artist("islands")

    self.artists["corridor"] = plot_polygon(
      env.corridor,
      ax=self.ax,
      color="lightgray",
      add_points=False
    )

    self.artists["islands"] = plot_polygon(
      env.islands,
      ax=self.ax,
      color="lightgreen",
      add_points=False
    )

    self.ax.set_aspect('equal', adjustable='box')
    self.ax.set_xlabel("Distance (100 m)")
    self.ax.set_ylabel("Distance (100 m)")

  def _draw_mesh(self):
    mesh = self.sim_state.mesh
    self._remove_artist("mesh")

    points = mesh.points[:, :2]
    triangles = mesh.cells_dict["triangle"]

    triang = mtri.Triangulation(
      points[:, 0],
      points[:, 1],
      triangles
    )

    self.artists["mesh"] = self.ax.triplot(triang, linewidth=0.5, color="gray")
   
  def _draw_potential(self, stride=5):

    field = self.sim_state.field

    tri = field.tri
    phi = field.u
    # print(np.finfo(phi.dtype).bits)
    # eps = 1e-12
    # delta = np.clip(1 - phi, eps, 1.0)
    # phi = -np.log10(delta)
    grad = field.grad

    self._remove_artist("potential_contour")
    self._remove_artist("potential_fill")
    self._remove_artist("gradient")

    #eps = 1e-6
    #phi = np.log10(phi + eps)

    levels = np.linspace(0, 1, 10)
    levels = levels**10 # More levels for lower potential
    #Contour
    contour = self.ax.tricontour(tri, phi, levels=levels, alpha = 1)
    self.artists["potential_contour"] = contour

    #Filled contour (optional)
    fill = self.ax.tricontourf(tri, phi, levels=levels, alpha=0.2)
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

  def _draw_ship(self):
    ship_poly = get_ship_polygon(self.sim_state.ship)

    self._remove_artist("ship")

    self.artists["ship"] = plot_polygon(
      ship_poly,
      ax=self.ax,
      color="steelblue",
      add_points=False
    )

  def _draw_trajectory(self, path_poly):
    self._remove_artist("trajectory")

    self.artists["trajectory"] = plot_polygon(
      path_poly,
      ax=self.ax,
      color="steelblue",
      alpha=0.3,
      add_points=False
    )

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

  def toggle_artist(self, name):
    artist = self.artists.get(name)

    if artist is None:
      return

    visible = self.visibility_boxes[name].isChecked()

    # ContourSet
    if hasattr(artist, "collections"):
      for coll in artist.collections:
        coll.set_visible(visible)

    # Listen von Artists
    elif isinstance(artist, list):
      for a in artist:
        a.set_visible(visible)

    # Einzelartist
    else:
      artist.set_visible(visible)

    self.canvas.draw_idle()

  def inspect_triangle(self, pos):
    field = self.sim_state.field
    tri_id = field.finder(*pos)
    if tri_id == -1:
      print("Outside mesh")
      return

    tri = field.mesh.t[:, tri_id]     # 3 node indices
    coords = field.mesh.p[:, tri]     # (2,3)
    values = field.u[tri]

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
    grad = field.grad[:, tri_id]
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
      label = field.edge_labels.get(key, None)

      #Doesn't belong to border part of the domain.
      if label is None:
        label = "interior"

      print(f"  Edge ({i}, {j}): {label}")
  
  def on_click(self, event):
    if event.inaxes != self.ax:
      return

    if event.xdata is None or event.ydata is None:
      return

    field = self.sim_state.field
    pos = np.array([event.xdata, event.ydata])

    tri_id = field.finder(*pos)
    if tri_id == -1:
      print("Outside mesh")
      return

    print("\n--- CLICK ---")
    self.inspect_triangle(pos)

    # optional: visuelles Highlight
    tri_nodes = field.mesh.t[:, tri_id]
    coords = field.mesh.p[:, tri_nodes]

    self.ax.scatter(coords[0], coords[1], color='red')
    self.fig.canvas.draw()