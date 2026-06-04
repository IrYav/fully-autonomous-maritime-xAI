import numpy as np
import numpy.typing as npt
import matplotlib.tri as mtri
from collections import defaultdict
from meshio import Mesh
from skfem import Basis, MeshTri, asm, solve, condense
from skfem.element import ElementTriP1
from skfem.models.poisson import laplace

class HarmonicField:
  mesh: MeshTri
  tri: mtri.Triangulation
  finder: mtri.TriFinder
  grad: np.ndarray
  u: np.ndarray
  edge_labels: dict[frozenset[int], str]
  
  def __init__(self, gmsh_mesh: Mesh, bc: list[tuple[npt.NDArray[np.int_] | None, float]]):
    '''
      Creates the harmonic field on the mesh, given the boundary conditions.
      Currently only accepts dirichlet boundary conditions.
      The harmonic potential is approximated using first-order (P1)
      Lagrange finite elements on a triangular mesh.
      Args:
        gmsh_mesh: The mesh to solve with.
        bc: List of (node_indices, bc_value)
          Currently only Dirichlet boundary conditions. 
          Extracts via `get_boundary_conditions` if `None`
      Returns:
        A HarmonicField, satisfying the boundary conditions.
    '''

    if bc is None:
      bc = self.get_boundary_conditions(mesh)
    # --- Mesh conversion ---
    points = np.ascontiguousarray(
        gmsh_mesh.points[:, :2].T
    )

    triangles = np.ascontiguousarray(
        gmsh_mesh.cells_dict["triangle"].T
    )

    self.mesh = MeshTri(points,triangles)

    # --- Boundary topology ---
    self.edge_labels = self._extract_edge_labels(gmsh_mesh)

    # --- FEM system ---
    A, b_vec, basis = self._build_system(self.mesh)

    # --- Solve Dirichlet problem ---
    self.u = self._solve_dirichlet(A, b_vec, bc, A.shape[0])

    # --- Triangle locator ---
    self.tri = mtri.Triangulation(
      self.mesh.p[0],
      self.mesh.p[1],
      self.mesh.t.T
    )
    self.finder = self.tri.get_trifinder()

    # --- Gradient (P1 → constant per triangle) ---
    interp = basis.interpolate(self.u)
    self.grad = interp.grad[:, :, 0]
  @staticmethod
  def get_boundary_conditions(mesh: Mesh):
    bc = []

    id_to_name = {
      data[0]: name
      for name, data in mesh.field_data.items()
    }

    for cells, phys in zip(mesh.cells, mesh.cell_data["gmsh:physical"]):
      if cells.type != "line":
        continue

      lines = cells.data

      for group_id in np.unique(phys):
        name = id_to_name.get(group_id, None)
        if name is None:
          continue

        nodes = np.unique(lines[phys == group_id].ravel())

        if name.startswith("island_") or name == "outer":
          bc.append((nodes, 0.0))
        elif name == "end":
          bc.append((nodes, 1.0))

    return bc
  
  @staticmethod
  def _extract_edge_labels(gmsh_mesh: Mesh) -> dict[frozenset[int], str]:
    edge_labels = {}

    # Mapping: physical id → name
    id_to_name = {
      data[0]: name
      for name, data in gmsh_mesh.field_data.items()
    }

    for cells, phys in zip(gmsh_mesh.cells, gmsh_mesh.cell_data["gmsh:physical"]):
      if cells.type != "line":
        continue

      lines = cells.data  # shape (n_lines, 2)

      for line, group_id in zip(lines, phys):
        name = id_to_name.get(group_id, None)
        if name is None:
          continue

        i, j = int(line[0]), int(line[1])
        key = frozenset((i, j))

        edge_labels[key] = name

    return edge_labels
  
  @staticmethod
  def _build_system(mesh):
    basis = Basis(mesh, ElementTriP1())
    A = asm(laplace, basis)
    b = np.zeros(A.shape[0])
    return A, b, basis
  
  @staticmethod
  def _solve_dirichlet(A, b, bc, n_dofs):
    x = np.zeros(n_dofs)
    dirichlet_nodes = []

    for nodes, value in bc:
      x[nodes] = value
      dirichlet_nodes.append(nodes)

    D = np.unique(np.concatenate(dirichlet_nodes))

    A2, b2, x_full, I = condense(A, b, D=D, x=x)
    u_reduced = solve(A2, b2)
    x_full[I] = u_reduced

    return x_full

  def gradient_at_(self, position: np.ndarray, normalized=True) -> np.ndarray:
    tri_id = self.finder(*position)
    gradient = self.grad[:, tri_id]

    if not normalized:
      return gradient

    norm = np.linalg.norm(gradient)
    
    if norm == 0:
      return np.zeros_like(gradient)
    return gradient / norm

  def descend(self, position: np.ndarray, step: float = 0.01, max_steps: int = 1000):
    #TODO: Try Edge-Walker Alg
    path = []
    pos = np.array(position, dtype=float)

    for _ in range(max_steps):

      tri_id = self.finder(*pos)

      if tri_id == -1:
        break

      path.append(tri_id)

      g = self.grad[:, tri_id]
      direction = -g

      n = np.linalg.norm(direction)
      if n == 0:
        break

      direction /= n
      pos = pos + step * direction

    return path

  def analyze_path(path: list[int], n_tri: int):
    path = [int(p) for p in path]
    visit_counts = [0] * n_tri
    oscillations = []
    oscillation_counts = defaultdict(int)
    simplified = []

    i = 0
    while i < len(path):
      tri = path[i]
      visit_counts[tri] += 1

      # 2-step cycle: A B A
      if i >= 2 and tri == path[i - 2]:
        t1 = path[i - 1]
        t2 = tri

        oscillations.append((t1, t2))
        oscillation_counts[(t1, t2)] += 1

        # skip both steps of the oscillation
        i += 1
        continue

      simplified.append(tri)
      i += 1

    return {
      "path": path,
      "path_simplified": simplified,
      "visit_counts": visit_counts,
      "oscillations": oscillations,
      "oscillation_counts": dict(oscillation_counts),
    }

  def diagnose_field(field, n_samples=50, steps=[0.05, 0.02, 0.01, 0.005]):
    results = []

    # zufällige Startpunkte im Mesh
    xmin, xmax = field.mesh.p[0].min(), field.mesh.p[0].max()
    ymin, ymax = field.mesh.p[1].min(), field.mesh.p[1].max()

    def random_point():
      for _ in range(100):
        p = np.array([
          np.random.uniform(xmin, xmax),
          np.random.uniform(ymin, ymax)
        ])
        if field.finder(*p) != -1:
          return p
      return None

    for step in steps:
      oscillations_total = 0
      stuck = 0
      lengths = []

      for _ in range(n_samples):
        p0 = random_point()
        if p0 is None:
          continue

        path = field.descend(p0, step=step, max_steps=500)
        stats = HarmonicField.analyze_path(path, field.tri.triangles.shape[0])

        oscillations_total += len(stats["oscillations"])
        lengths.append(len(path))

        if len(path) < 5:
          stuck += 1

      results.append({
        "step": step,
        "avg_len": np.mean(lengths),
        "osc": oscillations_total / n_samples,
        "stuck_ratio": stuck / n_samples
      })

    return results