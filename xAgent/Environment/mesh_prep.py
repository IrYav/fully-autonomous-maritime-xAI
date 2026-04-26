import math
import numpy as np

from dataclasses import dataclass
from shapely.geometry import Point, LineString, GeometryCollection, box, Polygon, MultiPolygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union
from shapely import difference

# Contains a part of the Boundary, 
@dataclass
class BoundarySegment:
    curve: LineString
    label: str

@dataclass
class Hole:
  poly: Polygon
  label: str

class PreparedDomain():
  boundary: list[BoundarySegment]
  holes: list[Hole]
  h: float
  eps: float

  def __init__(self, boundary, holes, end_polygon, h, eps) -> None:
    self.boundary = boundary
    self.holes = holes
    self.end_polygon = end_polygon
    self.h = h
    self.eps = eps

def prepare_domain(islands: MultiPolygon, end: Point, boundary, h=5.0, eps=0.5) -> PreparedDomain:
  """
    Prep mesh TODO
    Args:
      islands :shapely.geometry.MultiPolygon
        Collection of obstacle polygons.

    boundary: np.ndarray of shape (2, 2)
      Axis-aligned bounding box:
      [[xmin, ymin],
      [xmax, ymax]]
    end: Target Point, will become a small polygon.
    h: float, optional
      Target mesh size.
    eps: float, optional
      Tolerance used to simplify the geometry. The resulting geometry may
      deviate from the original by up to `eps` in Hausdorff distance.

      Suggested default: 0.1 * h
    Returns:
  """
  xmin, ymin = boundary[0]
  xmax, ymax = boundary[1]
    
  outer = np.array([
    [xmin, ymin],
    [xmax, ymin],
    [xmax, ymax],
    [xmin, ymax],
  ])
  domain = Polygon(outer)
  end_poly = get_end_polygon(end,5*h)

  # Simplify mesh, meshing works badly with too short edges.
  islands = islands.simplify(eps, preserve_topology=True)

  # Remove everything strictly outside the boundary.
  islands = islands.intersection(domain)

  #List of Polygons, each one an island
  islands = getattr(islands, "geoms", [islands])

  # Too short bordersegments (incl. borderpoints) cause unstable behavior,
  # make them to borderlines of sufficient length.
  islands = [fix_boundary_contacts(p, domain, 1.2 * h) for p in islands]

  # Fill up lakes.
  islands = [fill_lakes(p, domain) for p in islands]

  # Connect close boundary, too small passages shouldn't allow the field to pass-through.
  
  # Can activate 'easy-mode' by making removing gaps smaller than ship-width x k. 
  #   Probably more robust. Maybe required, sounds relatively doable.
  # TODO: Implement. Should add stability to mesh and learning feedback.


  # Cut out islands and the end.
  domain = domain.difference(unary_union(islands))
  domain = domain.difference(end_poly)

  domain = _flatten_domain(domain)

  domain.normalize()

  # Get all proper holes and label them.
  holes = [
    Hole(Polygon(ring), f"island_{i}")
    for i, ring in enumerate(domain.interiors)
  ]

  #Segment boundary properly, so that the labelling works.
  boundary_segments = extract_boundary_segments(domain, islands, end_poly, isle_offset = len(holes))
  boundary_segments = order_segments_into_loop(boundary_segments)

  return PreparedDomain(boundary_segments, holes, end_poly, h, eps)
   

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

def get_end_polygon(end: Point, radius: float):
  '''
    Returns a regular 16-gon around end, with the given radius.
  '''
  end = Point(*end)
  end = end.buffer(radius, quad_segs=4)
  return end

def close_gaps(islands, l):
  """
    If there are gaps < l, fill them up. 
    Mesher will likely misbehave in there, if gap is smaller than the resolution.
  """
  #TODO
  raise NotImplementedError()
  return

def extract_boundary_segments(domain, islands, end_poly, tol=1e-6, isle_offset = 0) -> list[BoundarySegment]:
  #TODO Documentation
  B = domain.exterior

  # Get overlaps.
  end_part = end_poly.boundary.intersection(B)
  
  island_parts = []
  for isle in islands:
    part = isle.boundary.intersection(B)
    if not part.is_empty:
      island_parts.append(part)

  end_lines = _flatten_lines(end_part)
  island_lines = [
    l for g in island_parts for l in _flatten_lines(g)
  ]

  # Every non-boundary part
  all_parts = unary_union(end_lines + island_lines)

  # Unassigned boundary segments
  outer_geom = B.difference(all_parts)
  outer_lines = _flatten_lines(outer_geom)

  segments = []

  # Label all categories
  for l in end_lines:
    segments.append(BoundarySegment(l, "end"))

  for i, l in enumerate(island_lines):
    segments.append(BoundarySegment(l, f"island_{i+isle_offset}"))

  for l in outer_lines:
    segments.append(BoundarySegment(l, "outer"))

  # Error handling:
  _assert_on_boundary(segments, domain, tol)
  _assert_cover(segments, domain, tol)
  _assert_no_overlap(segments, tol)

  return segments

def order_segments_into_loop(segments: list[BoundarySegment], tol=1e-9):
  """
  Sorts segments, so that the segments form a closed loop.
  Args:
    segments: list[BoundarySegment]
  Returns:
    ordered list of BoundarySegment forming a closed loop
  Raises:
    RuntimeError: If segments can't form a loop.
  """

  unused = segments.copy()
  ordered = []

  # 
  current = unused.pop(0)
  ordered.append(current)

  while unused:
    end_pt = current.curve.coords[-1]

    found = False

    for i, seg in enumerate(unused):
      coords = list(seg.curve.coords)

      coords = seg.curve.coords

      #End matches this start.
      if _points_equal(coords[0], end_pt, tol):
        current = seg

      #End matches this end, needs a flip.
      elif _points_equal(coords[-1], end_pt, tol):
        current = type(seg)(seg.curve.reverse(), seg.label)

      else:
        continue

      ordered.append(current)
      unused.pop(i)
      found = True
      break

    if not found:
      for i, seg in enumerate(segments):
        coords = list(seg.curve.coords)
        print(f"{i} | {seg.label} | start={coords[0]} | end={coords[-1]} | n={len(coords)}")
      print(end_pt)
      raise RuntimeError("Could not connect segments into a loop")

  # Check closure.
  start_pt = ordered[0].curve.coords[0]
  end_pt = ordered[-1].curve.coords[-1]

  if not _points_equal(start_pt, end_pt, tol):
    raise RuntimeError("Segments do not form a closed loop")

  return ordered

def _assert_on_boundary(segments, domain, tol=1e-6):
  '''
    Raises an error, iff segments are too far away from domain boundary.
  '''
  B = domain.exterior
  for i, seg in enumerate(segments):
    d = seg.curve.distance(B)
    if d > tol:
      raise ValueError(
        f"Segment {i} not on boundary (distance={d:.3e})"
      )

def _assert_cover(segments, domain, tol=1e-6):
  '''
    Raises an error, iff boundary has sections not covered by the segments.
  '''
  union = unary_union([s.curve for s in segments])
  diff = union.symmetric_difference(domain.exterior).length

  if diff >= tol:
    raise ValueError(
      f"Boundary not fully covered (gap length={diff:.3e})"
    )

def _assert_no_overlap(segments, tol=1e-9):
  '''
    Raises an error, iff intersections of two segments are longer than `tol`.
  '''
  for i, a in enumerate(segments):
    for j, b in enumerate(segments):
      if i >= j:
        continue

      inter = a.curve.intersection(b.curve)

      if inter.length > tol:
        raise ValueError(
          f"Segments {i} and {j} overlap (length={inter.length:.3e})"
        )

def _points_equal(p, q, tol):
  return np.linalg.norm(np.array(p) - np.array(q)) < tol

def _flatten_domain(domain):
  '''
    Flattens domain, iff domain is a (Multi-)Polygon. Raises an error if domain otherwise.
  '''
  if isinstance(domain, Polygon):
    return domain

  elif isinstance(domain, MultiPolygon):
    # Multiple seperated Polygons, we pick the largest area.
    return max(domain.geoms, key=lambda p: p.area)

  else:
    raise RuntimeError(
      f"Domain construction failed.\n"
      f"Type: {type(domain)}\n"
      f"Valid: {getattr(domain, 'is_valid', 'n/a')}\n"
      f"Area: {getattr(domain, 'area', 'n/a')}"
    )

def _flatten_lines(geom):
  '''
    Given a (Multi)LineString, turns them into a list of LineStrings.
  '''
  if geom.is_empty:
    return []
  if geom.geom_type == "LineString":
    return [geom]
  if geom.geom_type == "MultiLineString":
    return list(geom.geoms)
  return []

def _debug_geometry(g: BaseGeometry):
  print(
    f"Debugged Geometry:\n"
    f"Type: {type(g)}\n"
    f"Valid: {getattr(g, 'is_valid', 'n/a')}\n"
    f"Area: {getattr(g, 'area', 'n/a')}"
  )

