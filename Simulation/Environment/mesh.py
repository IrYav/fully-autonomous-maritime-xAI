from dataclasses import dataclass

import numpy as np

from Simulation.Environment.mesh_prep import PreparedDomain, LabeledRing

import gmsh
from collections import defaultdict
from skfem import MeshTri

@dataclass
class BuiltMesh:
  mesh_tri: MeshTri

  boundary_conditions: list[tuple[np.ndarray, float]]

  edge_labels: dict[frozenset[int], str]

def build_mesh(pd: PreparedDomain, verbose = False):

  gmsh.initialize()

  if not verbose:
    gmsh.option.setNumber("General.Verbosity", 1)

  gmsh.model.add("mesh")

  point_cache = {}
  label_to_lines = defaultdict(list)

  def point(xy):
    key = (round(xy[0], 12), round(xy[1], 12))
    if key not in point_cache:
      point_cache[key] = gmsh.model.geo.addPoint(xy[0], xy[1], 0, pd.h)
    return point_cache[key]

  def build_ring_from_labeled(ring: LabeledRing):
    coords = ring.coords
    labels = ring.labels

    lines = []

    for i in range(len(labels)):
      p = point(coords[i])
      q = point(coords[i+1])

      line = gmsh.model.geo.addLine(p, q)
      lines.append(line)

      label_to_lines[labels[i]].append(line)

    loop = gmsh.model.geo.addCurveLoop(lines)
    return loop

  # --- Build all rings ---
  loops = []
  for ring in pd.rings:
    loop = build_ring_from_labeled(ring)
    loops.append(loop)

  # --- First ring = exterior, rest = holes ---
  surface = gmsh.model.geo.addPlaneSurface(loops)

  gmsh.model.geo.synchronize()

  # --- Boundary labels ---
  for label, lines in label_to_lines.items():
    tag = gmsh.model.addPhysicalGroup(1, lines)
    gmsh.model.setPhysicalName(1, tag, label)

  # --- Domain label ---
  tag = gmsh.model.addPhysicalGroup(2, [surface])
  gmsh.model.setPhysicalName(2, tag, "domain")

  #Finer mesh near islands.
  _apply_distance_field(label_to_lines,pd.h)

  # --- Mesh ---
  gmsh.model.mesh.generate(2)

  
  node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
  points = np.asarray(node_coords).reshape(-1, 3)

  types, element_tags, element_nodes = gmsh.model.mesh.getElements(dim=2)

  triangles = np.asarray(element_nodes[0]).reshape(-1, 3)
  
  # Maps Gmsh node tags to contiguous MeshTri vertex indices.
  lookup = np.empty(node_tags.max() + 1, dtype=np.int32)
  lookup[node_tags] = np.arange(len(node_tags))

  triangles = lookup[
    np.asarray(element_nodes[0]).reshape(-1, 3)
  ]
  
  #--- Create result ---
  mesh_tri = MeshTri(
    np.ascontiguousarray(points[:, :2].T),
    np.ascontiguousarray(triangles.T),
  )

  boundary_conds = _get_dirichlet_bcs(lookup)
  edge_labels = _extract_edge_labels(lookup)

  gmsh.finalize()
  return BuiltMesh(mesh_tri, boundary_conds, edge_labels)

def _apply_distance_field(label_to_lines, h, h_max = None, h_min = None, d_max = None):
  d_max = 20 * h if d_max is None else d_max
  h_min = 1/3 * h if h_min is None else h_min
  h_max = 2 * h if h_max is None else h_max
  # --- Distance field ---
  dist = gmsh.model.mesh.field.add("Distance")

  curves = []
  for label, lines in label_to_lines.items():
    if label.startswith("island") or label == "end":
      curves.extend(lines)

  gmsh.model.mesh.field.setNumbers(dist, "CurvesList", curves)

  gmsh.option.setNumber("Mesh.CharacteristicLengthFromPoints", 0)
  gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 0)
  gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 0)

  # --- Threshold ---
  th = gmsh.model.mesh.field.add("Threshold")

  gmsh.model.mesh.field.setNumber(th, "InField", dist)
  gmsh.model.mesh.field.setNumber(th, "SizeMin", h_min)
  gmsh.model.mesh.field.setNumber(th, "SizeMax", h_max)
  gmsh.model.mesh.field.setNumber(th, "DistMin", 0.0)
  gmsh.model.mesh.field.setNumber(th, "DistMax", d_max)

  gmsh.model.mesh.field.setAsBackgroundMesh(th)

def _extract_edge_labels(lookup) -> dict[frozenset[int], str]:
  """
  Maps each boundary edge of the MeshTri to its physical group label.
  """

  edge_labels = {}

  for dim, phys_tag in gmsh.model.getPhysicalGroups(dim=1):

    label = gmsh.model.getPhysicalName(dim, phys_tag)

    # All geometric entities of this physical group
    entities = gmsh.model.getEntitiesForPhysicalGroup(dim, phys_tag)

    for entity in entities:
      _, _, element_nodes = gmsh.model.mesh.getElements(
        dim=1,
        tag=entity,
      )

      if len(element_nodes) == 0:
        continue

      edges = lookup[
        np.asarray(element_nodes[0]).reshape(-1, 2)
      ]

      for edge in edges:
        edge_labels[frozenset(edge)] = label

  return edge_labels

def _get_dirichlet_bcs(lookup) -> list[tuple[np.ndarray, float]]:
  '''
    Groups mesh nodes by their physical group e.g., `outer` or `island_5` and 
    assigns a value for the dirichlet boundary conditions.
  '''
  boundary_conditions = []

  for dim, phys_tag in gmsh.model.getPhysicalGroups(dim=1):

    label = gmsh.model.getPhysicalName(dim, phys_tag)

    if label == "end":
      value = 1.0
    elif label == "outer" or label.startswith("island_"):
      value = 0.0
    else:
      continue

    node_arrays = []

    # All geometric entities of this physical group
    entities = gmsh.model.getEntitiesForPhysicalGroup(dim, phys_tag)

    for entity in entities:
      _, _, nodes = gmsh.model.mesh.getElements(
        dim=1,
        tag=entity,
      )

      if len(nodes) > 0:
        node_arrays.append(nodes[0])

    if not node_arrays:
      continue

    nodes = np.unique(
      lookup[np.concatenate(node_arrays)]
    )

    boundary_conditions.append((nodes, value))

  return boundary_conditions