from matplotlib.patches import PathPatch
import numpy as np
import matplotlib.tri as mtri
from matplotlib.figure import Figure

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
  QButtonGroup, QCheckBox, QDoubleSpinBox, QFormLayout, QHBoxLayout, 
  QLabel, QRadioButton, QTabWidget, QWidget, QVBoxLayout, QPushButton, QSlider,
  )

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as Canvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from shapely.plotting import plot_line, plot_polygon
from shapely import affinity


from Simulation.Auxilaries.auxiliaries import SimulationState
from Simulation.Environment.agent import Action, get_ship_polygon
from Simulation.Visualization.Mouse.mouse_tool import MouseTool
from Simulation.Visualization.Mouse.mouse_tool_actions import MouseToolActions
from Simulation.Visualization.RightHalf.AdminTab.editor_controller import EditorController
from Simulation.Visualization.RightHalf.AdminTab.env_viewer import EditorTab
from Simulation.Visualization.RightHalf.AdminTab.draft_model import Material
from Simulation.Visualization.RightHalf.views import ImageView, StateVector


class EnvironmentViewer(QWidget):
  def __init__(self, ec: EditorController):
    super().__init__()

    # State
    self.sim_state : SimulationState = None
    
    #Render details for polygons
    self._set_alphas()
    self._set_colors()

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

    #Central handling of mouse events within the canvas.
    self.mouse: MouseTool
    self.connect_mouse_tool(MouseTool())
    self.tools = MouseToolActions(self.mouse)
    self.set_editor_cont(ec)

    # Layout Areas
    self.button_layout = QHBoxLayout()
  
    self.sidebar_layout = QVBoxLayout()
    self.local_view_layout = QVBoxLayout()

    viewer_layout = QHBoxLayout()
    viewer_layout.addLayout(self.sidebar_layout,0)
    viewer_layout.addWidget(self.canvas,3)
    viewer_layout.addLayout(self.local_view_layout,1)

    main_layout = QVBoxLayout()
    main_layout.addWidget(self.toolbar)
    main_layout.addLayout(viewer_layout)
    main_layout.addLayout(self.button_layout)


    self._setup_visibility_controls(self.sidebar_layout)
    self._setup_inputs(self.sidebar_layout)
    self._setup_right_half(self.local_view_layout)
    # Push checkboxes up.
    self.sidebar_layout.addStretch()

    self.setLayout(main_layout)
  
  # ---------- Init Helpers --------
  # General Structure, create widget, add it to layout.
  def _setup_inputs(self, layout):
    form = QFormLayout()

    self.rudder_input = QSlider(Qt.Orientation.Horizontal)
    self.rudder_input.setRange(-100, 100)
    self.rudder_input.setValue(0)
    self.rudder_input.setTickPosition(QSlider.TickPosition.TicksBelow)
    self.rudder_input.setTickInterval(25)
    #Slider right means front is pushed to the right.
    self.rudder_input.setInvertedAppearance(True)

    self.accel_input = QSlider(Qt.Orientation.Horizontal)
    self.accel_input.setRange(-100, 100)
    self.accel_input.setValue(0)
    self.accel_input.setTickPosition(QSlider.TickPosition.TicksBelow)
    self.accel_input.setTickInterval(25)

    self.rudder_value = QLabel("0.00")
    self.rudder_input.valueChanged.connect(
        lambda v: self.rudder_value.setText(f"{v/100:.2f}")
    )    
    self.accel_value = QLabel("0.00")
    self.accel_input.valueChanged.connect(
        lambda v: self.accel_value.setText(f"{v/100:.2f}")
    )

    self.dt_input = QDoubleSpinBox()
    self.dt_input.setRange(0.01, 100.0)
    self.dt_input.setSingleStep(0.1)
    self.dt_input.setValue(1.0)

    form.addRow("Rudder", self.rudder_input)
    form.addRow("Rudder Value", self.rudder_value)
    form.addRow("Acceleration", self.accel_input)
    form.addRow("Acceleration Value", self.accel_value)
    form.addRow("dt", self.dt_input)

    container = QWidget()
    container.setLayout(form)

    layout.addWidget(container)
  
  def _setup_visibility_controls(self, layout):
    #Disable cluttering artists per default.
    default_visible = {
      "mesh": False,
      "gradient": False,          # Quiver
      "potential_fill": False,
      "potential_contour": False,
      "corridor": False,
    }
        

    self.visibility_boxes = {}

    for name in self.artists.keys():
      box = QCheckBox(name)
      box.setChecked(default_visible.get(name, True))

      box.stateChanged.connect(
          lambda _, n=name: self.toggle_artist(n)
      )

      layout.addWidget(box)
      self.visibility_boxes[name] = box
  
  def _setup_right_half(self, layout):

    self.tabs = QTabWidget()
    #Input tabs
    self.input_tab = QTabWidget()

    self.tabs.addTab(self.input_tab, "Input")

    self.local_view = ImageView(["obstacles", "goal", "ship"])
    self.global_view = ImageView(["islands", "goal", "lview"])
    self.state_vector = StateVector()

    self.input_tab.addTab(self.local_view, "Local View")
    self.input_tab.addTab(self.global_view, "Global View")
    self.input_tab.addTab(self.state_vector, "State")

    #Admin tabs
    self.editor_tab = EditorTab(self.ec,self.tools, self)
    self.tabs.addTab(self.editor_tab, "Env-Tools")

    layout.addWidget(self.tabs)

  # ---------- Public API ----------
  def set_sim_state(self, sim_state : SimulationState):
    self.sim_state = sim_state
    self.ec.set_env(sim_state.env)
    self._render()
    self._set_ax_lims()

  def set_lviews(self, obs: np.ndarray):
    self.local_view.set_views(obs)

  def set_gviews(self, obs: np.ndarray):
    self.global_view.set_views(obs)

  def set_state_vector(self, data: dict):
    self.state_vector.set_state(data)

  def set_editor_cont(self, ec: EditorController):
    self.ec = ec
    self.ec.changed.connect(self._draw_draft)

  def add_main_button(self, label, callback):
    btn = QPushButton(label)
    btn.clicked.connect(lambda *_: callback())
    self.button_layout.addWidget(btn)
    return btn

  def add_sidebar_button(self, label, callback, toggleable = False):
    btn = QPushButton(label)
    if toggleable:
      btn.setCheckable(True)
      btn.toggled.connect(callback)
    else:
      btn.clicked.connect(lambda *_: callback())

    self.sidebar_layout.addWidget(btn)
    return btn
  
  def add_radio_group(self, title: str, options: dict[str, callable]):
    container = QWidget()
    layout = QVBoxLayout(container)

    layout.addWidget(QLabel(title))

    group = QButtonGroup(container)

    for i, (label, callback) in enumerate(options.items()):
      btn = QRadioButton(label)

      if i == 0:
        btn.setChecked(True)

      btn.toggled.connect(
        lambda checked, cb=callback: checked and cb()
      )

      group.addButton(btn)
      layout.addWidget(btn)

    self.sidebar_layout.addWidget(container)

    return group

  def get_sim_input(self):
    return (
      Action(
        self.rudder_input.value() / 100, 
        self.accel_input.value() / 100), 
      self.dt_input.value())
  
  def update_step(self, ship_path):
    self._draw_ship()
    self._draw_trajectory(ship_path)
    self.canvas.draw()

  def update_draft(self):
    '''
      Redraw draft.
    '''
    self._draw_draft()
    self.canvas.draw()
  
  def set_preview(self, patches: list[PathPatch]):
    '''
      Adds the patches to the current ax,
      after deleting the old ones.

      Can be called with `patches = []` to wipe the preview.
    '''
    self._remove_artist("Preview")
    artists = []
    for patch in patches:
        self.ax.add_patch(patch)
        artists.append(patch)

    self.artists["Preview"] = artists
    self.canvas.draw()

  # ---------- Rendering ----------
  def _render(self):
    if self.sim_state.env is None or self.sim_state.mesh is None:
      return
    
    self._draw_boundary()
    self._draw_corridor()
    self._draw_draft()
    self._set_axes()
    self._draw_mesh()
    self._draw_potential()
    self._draw_ship()

    self._apply_visibility_settings()

    self.canvas.draw()

  def _draw_draft(self):
    #Delete all draft
    for mat in Material:
      self._remove_artist(mat)
      self.artists[mat] = []
    for ao in self.ec.draft:
      self.artists[ao.material].append(
        plot_polygon(
          ao.polygon,
          ax=self.ax,
          alpha=self.alphas[ao.material],
          color=self.colors[ao.material],
          add_points=False
        )
      )

  def _draw_corridor(self):
    self._remove_artist("corridor")
    env = self.sim_state.env
    self.artists["corridor"] = plot_polygon(
      env.corridor,
      ax=self.ax,
      color="lightgray",
      add_points=False
    )

  def _draw_boundary(self):
    self._remove_artist("boundary")
    env = self.sim_state.env
    self.artists["boundary"] = plot_line(
      env.boundary.exterior,
      ax=self.ax,
      color="darkgray",
      alpha=0.5,
      add_points=False
    )

  def _draw_mesh(self):
    mesh = self.sim_state.mesh.mesh_tri
    self._remove_artist("mesh")

    triang = mtri.Triangulation(
        *mesh.p,
        mesh.t.T,
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

    #Quiver (little arrows)
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

  def _set_axes(self):
    self.ax.set_aspect('equal', adjustable='box')
    self.ax.set_xlabel("Distance in m")
    self.ax.set_ylabel("Distance in m")

  def _set_colors(self):
    self.colors = {
      Material.LAND: "#009E73",   # Bluish Green
      Material.VOID: "#56B4E9",  # Sky Blue
      Material.END: "#E69F00",    # Orange
    }

  def _set_alphas(self):
    self.alphas = {
      Material.LAND: 0.35,
      Material.VOID: 0.7,
      Material.END: 0.35,
    }

  def _set_ax_lims(self):
    '''
      Set the default view to slightly bigger than the boundary.
    '''
    boundary = self.sim_state.env.boundary
    boundary = affinity.scale(boundary, 1.05, 1.05)
    xmin, ymin, xmax, ymax = boundary.bounds
    self.ax.set_xlim(xmin, xmax)
    self.ax.set_ylim(ymin, ymax)

  # ---------- Utilities ----------
  def _remove_artist(self, key):
    artist = self.artists.get(key)

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

    self.artists[key] = None

  def _apply_visibility_settings(self):
    for name in self.visibility_boxes:
      self.toggle_artist(name)

  def toggle_artist(self, name):
    artist = self.artists.get(name)

    if artist is None:
      return

    visible = self.visibility_boxes[name].isChecked()

    # ContourSet
    if hasattr(artist, "collections"):
      for coll in artist.collections:
        coll.set_visible(visible)

    # Lists of artists
    elif isinstance(artist, list):
      for a in artist:
        a.set_visible(visible)

    # Singleartist
    else:
      artist.set_visible(visible)

    self.canvas.draw_idle()

  def inspect_triangle(self, pos):
    field = self.sim_state.field
    tri_id = field.tri.get_trifinder()(*pos)
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
  
  def connect_mouse_tool(self, mouse):
    """
    Connect a MouseController to the matplotlib canvas.
    """
    self.mouse = mouse

    def to_pos(event):
      if (
        event.inaxes != self.ax or
        event.xdata is None or
        event.ydata is None
      ):
        return None

      return np.array([event.xdata, event.ydata])

    def on_press(event):
      if self.mouse is None:
        return

      pos = to_pos(event)
      if pos is not None:
        self.mouse.on_press(pos)

    def on_move(event):
      if self.mouse is None:
        return

      pos = to_pos(event)
      if pos is not None:
        self.mouse.on_move(pos, bool(event.buttons))

    def on_release(event):
      if self.mouse is None:
        return

      pos = to_pos(event)
      if pos is not None:
        self.mouse.on_release(pos)

    self.canvas.mpl_connect("button_press_event", on_press)
    self.canvas.mpl_connect("motion_notify_event", on_move)
    self.canvas.mpl_connect("button_release_event", on_release)