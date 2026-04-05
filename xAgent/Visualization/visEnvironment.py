import numpy as np
import matplotlib.tri as mtri
import matplotlib.pyplot as plt

from PySide6.QtWidgets import QApplication, QHBoxLayout, QWidget, QVBoxLayout, QPushButton
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as Canvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from shapely.plotting import plot_polygon


class EnvironmentViewer(QWidget):
  def __init__(self, boundary):
    super().__init__()

    self.boundary = boundary

    # State
    self.env = None
    self.mesh = None

    # Artists
    self.artists = {
      "corridor": None,
      "islands": None,
      "mesh": None
    }

    # Matplotlib Figure
    self.fig, self.ax = plt.subplots()
    self.canvas = Canvas(self.fig)
    
    self.toolbar = NavigationToolbar2QT(self.canvas, self)

    layout = QVBoxLayout()
    layout.addWidget(self.toolbar)
    layout.addWidget(self.canvas)

    # eigener Bereich für Buttons
    self.button_layout = QHBoxLayout()  # oder QVBoxLayout
    layout.addLayout(self.button_layout)

    self.setLayout(layout)

  # ---------- Events ----------
  def _on_new_instance(self):
    if self.on_new_instance is not None:
      self.on_new_instance(self)

  # ---------- Public API ----------
  def set_state(self, env, mesh):
    self.env = env
    self.mesh = mesh
    self._render()

  def add_button(self, label, callback):
    btn = QPushButton(label)
    btn.clicked.connect(lambda: callback(self))
    self.button_layout.addWidget(btn)
    return btn

  # ---------- Rendering ----------
  def _render(self):
    if self.env is None or self.mesh is None:
      return

    self._draw_environment()
    self._draw_mesh()
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

    self.artists["mesh"] = self.ax.triplot(triang, linewidth=0.5)

  # ---------- Utilities ----------
  def _remove_artist(self, name):
    artist = self.artists.get(name)

    if artist is None:
      return

    if isinstance(artist, list):
      for a in artist:
        a.remove()
    else:
      artist.remove()

    self.artists[name] = None