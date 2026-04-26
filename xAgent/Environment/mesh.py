from xAgent.Environment.mesh_prep import PreparedDomain, LabeledRing

import gmsh
import meshio
from collections import defaultdict

def build_mesh(pd):

  gmsh.initialize()
  gmsh.model.add("mesh")

  point_cache = {}
  label_to_lines = defaultdict(list)

  def point(xy):
    key = (round(xy[0], 12), round(xy[1], 12))
    if key not in point_cache:
      point_cache[key] = gmsh.model.geo.addPoint(xy[0], xy[1], 0, pd.h)
    return point_cache[key]

  def build_ring_from_labeled(ring):
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

  # --- Mesh ---
  gmsh.model.mesh.generate(2)
  
  gmsh.write("mesh.msh")
  mesh = meshio.read("mesh.msh")

  gmsh.finalize()
  return mesh