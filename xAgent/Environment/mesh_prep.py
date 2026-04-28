import math
import numpy as np

from dataclasses import dataclass
from shapely import difference, orient_polygons
from shapely.geometry import Point, LineString, GeometryCollection, box, Polygon, MultiPolygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

# Closed 
@dataclass
class LabeledRing:
  coords: list[tuple[float, float]]
  labels: list[str]
  tag: str

class PreparedDomain():
  rings: list[LabeledRing]
  h: float
  eps: float

  def __init__(self, rings, h, eps) -> None:
    self.rings = rings
    self.h = h
    self.eps = eps

def prepare_domain(islands: MultiPolygon, end_poly: Polygon, boundary, h=5.0, eps=0.5) -> PreparedDomain:
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
  cutout = end_poly.union(unary_union(islands))
  domain = domain.difference(cutout)

  domain = _flatten_domain(domain)

  domain = orient_polygons(domain)

  #Label and prepare for gmsh.
  labeled_rings = label_domain(domain, islands, end_poly)
  return PreparedDomain(labeled_rings, h, eps)
   

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

def close_gaps(notImplementedYet):
  """
    If there are gaps < l, fill them up. 
    Mesher will likely misbehave in there, if gap is smaller than the resolution.
  """
  #TODO
  raise NotImplementedError()
  return

def label_domain(domain: Polygon, islands: list[Polygon], end_poly: Polygon, tol=1e-6) -> list[LabeledRing]:
  """
  Labels all boundary edges of domain (exterior + interiors).

  Priority:
    island > end > outer (exterior) / unknown (interior)

  Returns:
    list[LabeledRing]
  """

  rings = []

  # Precompute boundaries
  end_boundary = end_poly.exterior.buffer(1e-10)
  island_boundaries = [isle.exterior.buffer(1e-10) for isle in islands]

  # Edge classification, with priority.
  def classify_edge(edge: LineString, tol=1e-9):
    best_label = None
    best_overlap = 0.0

    edge_len = edge.length
    if edge_len == 0:
      return None

    # islands
    for i, isle in enumerate(island_boundaries):
      inter_len = edge.intersection(isle).length
      if inter_len > best_overlap:
        best_overlap = inter_len
        best_label = f"island_{i}"

    # end
    inter_len = edge.intersection(end_boundary).length
    if inter_len > best_overlap:
      best_overlap = inter_len
      best_label = "end"

    # threshold
    if best_overlap > tol:
      return best_label

    return None

  # --- process a single ring ---
  def process_ring(ring, tag, is_exterior):
    coords = list(ring.coords)
    n = len(coords) - 1

    labels = []

    for i in range(n):
      p = coords[i]
      q = coords[i+1]
      edge = LineString([p, q])

      lab = classify_edge(edge)

      if lab is None:
        if is_exterior:
          lab = "outer"
        else:
          lab = "unknown"

      labels.append(lab)

    return LabeledRing(coords=coords, labels=labels, tag=tag)

  # --- exterior ---
  rings.append(process_ring(domain.exterior, "exterior", True))

  # --- interiors ---
  for k, ring in enumerate(domain.interiors):
    rings.append(process_ring(ring, f"interior_{k}", False))

  return rings

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

