import math

import numpy as np
import pygmsh
from shapely import snap, unary_union, length, clip_by_rect
from shapely.geometry import Point, LineString, GeometryCollection, box, Polygon, MultiPolygon
from shapely.geometry.base import BaseGeometry
# Function taking environment -> mesh
# Environment = Boundary, Islands



def build_mesh(islands: MultiPolygon, boundary, h=5.0, eps= 0.5):
  """
    Builds a 2D mesh for a rectangular domain with polygonal obstacles ("islands").

    Returns a geometrically cleaned and topologically well-defined mesh that is
    robust for meshing and FEM computations. In particular:

    Obstacles are clipped to the domain.
    Degenerate point contacts with the boundary are resolved into proper barriers.
    The resulting geometry has a clear, unambiguous topology.
    The mesh is suitable for stable numerical methods (e.g. FEM, potential fields).
    Args:
      islands :shapely.geometry.MultiPolygon
        Collection of obstacle polygons.

    boundary: np.ndarray of shape (2, 2)
      Axis-aligned bounding box:
      [[xmin, ymin],
      [xmax, ymax]]
    h: float, optional
      Target mesh size.
    eps: float, optional
      Tolerance used to simplify the geometry. The resulting geometry may
      deviate from the original by up to `eps` in Hausdorff distance.

      Suggested default: 0.1 * h
    Returns:
      Mesh object
      A valid 2D mesh of the domain with obstacles, ready for numerical simulation.
  """

  xmin, ymin = boundary[0]
  xmax, ymax = boundary[1]
    
  outer = np.array([
    [xmin, ymin],
    [xmax, ymin],
    [xmax, ymax],
    [xmin, ymax],
  ])
  outer_poly_shapely = Polygon(outer)

  # Step 1: Simplify mesh, meshing works badly with too short edges.
  polys = islands.simplify(eps, preserve_topology=True)


  # Step 2: Remove everything strictly outside the boundary.
  polys = polys.intersection(outer_poly_shapely)

  polys = getattr(polys, "geoms", [polys])

  # Step 3: Too short bordersegments (or borderpoints) cause unstable behavior,
  #         make them to borderlines of sufficient length.
  polys = [fix_boundary_contacts(p, outer_poly_shapely, 1.2 * h) for p in polys]

  # Step 4: Fill up lakes.
  polys = [fill_lakes(p, outer_poly_shapely) for p in polys]

  # Step 5: Connect close boundary, too small passages shouldn't allow the field to pass-through.
  # Step 5.1: Can activate 'easy-mode' by making removing gaps smaller than ship-width x k. 
  #   Probably more robust. Maybe required, sounds relatively doable.
  # TODO : Implement. Should add stability to mesh and learning feedback.

  #Meshing everything
  with pygmsh.occ.Geometry() as geom:

    
    outer_poly_geom = geom.add_polygon(outer, mesh_size=h)

    hole_geoms = [
      geom.add_polygon(
        np.asarray(poly.exterior.coords[:-1]),
        mesh_size=h
      )
      for poly in polys
    ]

    domain = geom.boolean_difference([outer_poly_geom], hole_geoms)

    assert outer_poly_shapely.is_valid
    for p in polys:
      assert p.is_valid

    geom.synchronize()
    mesh = geom.generate_mesh()
  return mesh

#TODO: Redo the code. Should still work however.
def fix_boundary_contacts(polygon: Polygon, boundary: Polygon, l: float) -> Polygon:
  """
    Ensures that any contact with a border has a minimal length of l.
    If not, expands contact by a square of l x l size, shifted, so it is contained within the boundary.
    Squares near corners get snapped into it. A square has at either 0 or > l distance to an edge.    
    Args:
      polygon: The polygon that might have border contacts of length < l
      boundary: Rectangle, fully has to contain `polygon`.
      l: Minimum length of accepted contact segments.
    Returns:
      Polygon, with all contact points expanded.
  """
  contacts = polygon.intersection(boundary.exterior)
  fixes = MultiPolygon()

  def make_patch(point: Point, boundary: Polygon, l: float) -> Polygon:
    xmin, ymin, xmax, ymax = boundary.bounds
    x, y = point.x, point.y

    #Lower left corner of square
    x0 = x - l/2
    y0 = y - l/2

    corners = [(xmin, ymin), (xmin, ymax), 
               (xmax, ymin), (xmax, ymax)]

    # Corner Cases
    for cx, cy in corners:
      if math.hypot(x - cx, y - cy) < 1.5 * l:
        x0 = min(cx, xmax - l)
        y0 = min(cy, ymax - l)
        return box(x0, y0, x0 + l, y0 + l)
      
    # Edge cases.
    x0 = min(max(x0, xmin), xmax - l)
    y0 = min(max(y0, ymin), ymax - l)

    return box(x0, y0, x0 + l, y0 + l)

  #Returns a list of contact points/contact lines
  def extract_contacts(inter: BaseGeometry) -> list[BaseGeometry]:
    if inter.is_empty:
      return []

    result = []
    stack = [inter]

    while stack:
      g = stack.pop()
      t = g.geom_type

      if t in ("Point", "LineString"):
        result.append(g)

      elif t in ("MultiPoint", "MultiLineString", "GeometryCollection"):
        stack.extend(g.geoms)

      else:
        raise RuntimeError(f"Unexpected type: {t}")

    return result

  #Get contacts
  contacts = extract_contacts(contacts)

  #Retain too short contacts. Includes all points.
  contacts = [i for i in contacts if i.length < l] 
  
  #Replace LineStrings with their half-way point. 
  contacts = [
    g.interpolate(0.5, normalized=True) if g.geom_type == "LineString" else g
    for g in contacts
  ]

  patches = [make_patch(g, boundary, l) for g in contacts]

  if patches:
    polygon = polygon.union(unary_union(patches))

  #Verification and cleaning.
  assert not polygon.is_empty, "Empty geometry after boundary fix"

  # Union should give either one of those.
  assert polygon.geom_type in ("Polygon", "MultiPolygon"), \
    f"Unexpected geometry type: {polygon.geom_type}"

  # If a MultiPolygon occurs, we expect it to contain exactly one component.
  if polygon.geom_type == "MultiPolygon":
    assert len(polygon.geoms) == 1, (
      f"Unexpected MultiPolygon ({len(polygon.geoms)} parts): "
      f"{[{'type': g.geom_type, 'area': g.area, 'bounds': g.bounds} for g in polygon.geoms]}"
    )
    polygon = polygon.geoms[0]

  return polygon

def fill_lakes(polygon: Polygon, boundary: Polygon):
  """
    Removes all except the single largest connected component in boundary.difference(polygon) i.e., the holes.

    Args:
      polygon: Contained by boundary, may share border points
      boundary: Rectangle.
    Returns:
      `polygon` if there was at most one hole,
      Polygon with the n - 1 smallest holes filled otherwise.
  """
  #Get all holes.
  holes = boundary.difference(polygon)

  #polygon is (probably) equal to boundary, no holes.
  if holes.is_empty:
    return polygon

  #Get all holes
  holes = holes.geoms if hasattr(holes, "geoms") else [holes]

  #Sort by area
  holes = sorted(holes, key=lambda g: g.area)

  #Only one hole, we keep the biggest. Nothing to do.
  if len(holes) <= 1:
    return polygon

  # All but the biggest hole.
  holes_to_fill = holes[:-1]

  #Fill holes.
  polygon = polygon.union(unary_union(holes_to_fill))
  return polygon

def close_gaps(islands, l):
  """
    If there are gaps < l, fill them up. 
    Mesher will likely misbehave in there, if gap is smaller than the resolution.
  """
  #TODO
  raise NotImplementedError()
  return