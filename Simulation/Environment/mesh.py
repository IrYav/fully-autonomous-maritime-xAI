from Simulation.Environment.mesh_prep import PreparedDomain, LabeledRing

import gmsh
import meshio
from collections import defaultdict

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

  
  gmsh.write("mesh.msh")
  mesh = meshio.read("mesh.msh")

  gmsh.finalize()
  return mesh

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