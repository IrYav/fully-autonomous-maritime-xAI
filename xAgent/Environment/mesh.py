from xAgent.Environment.mesh_prep import PreparedDomain, BoundarySegment, Hole

import numpy as np
import gmsh
import meshio
from collections import defaultdict
from shapely import snap, unary_union, length, clip_by_rect
from shapely.geometry import Point, LineString, GeometryCollection, box, Polygon, MultiPolygon
from shapely.geometry.base import BaseGeometry

def build_mesh(pd: PreparedDomain):

  gmsh.initialize()
  gmsh.model.add("mesh")

  ctx = GeomBuilder(h=pd.h)

  #Outer Border
  loop = ctx.build_boundary(pd.boundary)

  #Holes
  hole_loops = ctx.build_holes(pd.holes)

  #Mesh the area
  surface_id = gmsh.model.geo.addPlaneSurface([loop] + hole_loops)


  gmsh.model.geo.synchronize()

  #Label everything (For boundary conditions)
  ctx.add_boundary_labels()
  tag = gmsh.model.addPhysicalGroup(2, [surface_id])
  gmsh.model.setPhysicalName(2, tag, "domain")


  gmsh.model.mesh.generate(2)

  #gmsh.option.setNumber("Mesh.SaveAll", 1)
  gmsh.write("mesh.msh")

  return meshio.read("mesh.msh")
  
  

class GeomBuilder:
  def __init__(self, h):
    self.h = h
    self.point_cache = {}
    self.labels = defaultdict(list) # (Labels -> Edge ID's)
  # --- Punkt-Deduplizierung ---
  def point(self, xy):
    key = (xy[0], xy[1])
    if key not in self.point_cache:
      self.point_cache[key] = gmsh.model.geo.addPoint(
        xy[0], xy[1], 0, self.h
      )
    return self.point_cache[key]

  # --- LineString → Gmsh Lines ---
  def lines_from_curve(self, curve: LineString):
    '''
      Shapely LineString to gmsh lines
    '''
    coords = curve.coords
    return [
      gmsh.model.geo.addLine(
        self.point(coords[i]),
        self.point(coords[i+1])
      )
      for i in range(len(coords) - 1)
    ]

  # --- Boundary ---
  def build_boundary(self, segments: list[BoundarySegment]):
    lines = []

    for seg in segments:
      seg_lines = self.lines_from_curve(seg.curve)
      lines.extend(seg_lines)
      self.labels[seg.label].extend(seg_lines)

    loop = gmsh.model.geo.addCurveLoop(lines)
    return loop

  # --- Holes ---
  def build_holes(self, holes: list[Hole]):
    hole_loops = []

    for hole in holes:
      lines = self.lines_from_curve(hole.poly.exterior)

      loop = gmsh.model.geo.addCurveLoop(lines)
      hole_loops.append(loop)

      self.labels[hole.label].extend(lines)

    return hole_loops

  def add_boundary_labels(self):
    for label, group in self.labels.items():
      tag = gmsh.model.addPhysicalGroup(1, group)
      gmsh.model.setPhysicalName(1, tag, label)