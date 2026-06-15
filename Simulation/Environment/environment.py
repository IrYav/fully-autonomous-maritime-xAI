import random
import numpy as np
import shapely.geometry as sgeom

from scipy.interpolate import make_interp_spline, BSpline
from shapely.geometry import LineString, Polygon, MultiPolygon, box
from shapely.affinity import rotate

Point  = np.ndarray  # shape (2,)
Points = np.ndarray  # shape (n,2)
Boundary = np.ndarray # shape (2,2)

class Environment:
  islands: MultiPolygon
  corridor: Polygon
  start: Point
  end_poly: Polygon
  boundary: Polygon

  def __init__(self, islands, corridor, start, end_poly, boundary) -> None:
    self.islands = islands
    self.corridor = corridor
    self.start = start
    self.end_poly = end_poly
    self.boundary = boundary

# Cubic B-splines
def interpolate(points: Points) -> BSpline:
  """
  Get a route through all points.
  Args:
    points: Sequence of at least two points to traverse, in that order.

  Returns:
    A BSpline object.
  """

  if len(points) < 2:
    raise ValueError("Can't interpolate with a single point, points = " + str(points))

  
  # chord-length parameterization
  distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
  t = np.concatenate(([0], np.cumsum(distances)))
  t = t / t[-1]

  #Dynamically allocate k, 3 points require quadratic spline.
  k = min(3, len(points) - 1)
  spline = make_interp_spline(t, points, k=k)
  return spline

def generate_points(n: int, boundary: Boundary) -> Points:
  """
  Generate random 2D points inside an axis-aligned bounding box.
  Guarantees a minimum distance between result[0] and result[-1]
  Args:
    n: Number of points to generate, including start and end.
    boundary: Bottom-Left and Top-Right corner of rectangle, shape (2,2)

  Returns:
    List of (x, y) coordinates.
  """
  width, height = np.abs(boundary[1] - boundary[0])
  perimeter = 2 * (width + height)
  result = []

  if n < 2:
    raise ValueError("Need at least 2 points for the checkpoints of a tour.")

  #Random number over perimeter for p0 and p1. Equally distributed.
  r0 = random.random()
  gap = 0.1 # Well defined for 0 < gap < 0.5
  #r1 should have at least gap distance to r0, equal distribution on the rest.
  r1 = random.random() * (1 - 2 * gap)
  r1 = (r0 + gap + r1) % 1
  r0 *= perimeter
  r1 *= perimeter

  p0 = boundary[0] + walk_around_perimeter(r0, width, height)
  p1 = boundary[0] + walk_around_perimeter(r1, width, height)

  result.append(p0)

  #Inner control points.
  for n in range (2,n):
    rpoint = np.array(
      (random.random() * width , 
       random.random() * height), dtype=float)
    result.append(rpoint + boundary[0])
  
  result.append(p1)

  return np.array(result, dtype=float)

def projection_sort(points: Points) -> Points:
  """
  Sort points based on the projection to the axis from points[0] to points[-1]

  Args:
    points: Points to project, at least 2. shape (n,)

  Returns:
    Sorted points, also shape (n,)
  """

  if len(points) < 2:
    raise ValueError("Less than two points.")

  start, end = points[0], points[-1]
  d = end - start
  denom = np.dot(d, d)

  if(denom == 0):
    raise ValueError("Start and end are the equal.")

  alphas = np.dot(points - start, d) / denom
  order = np.argsort(alphas)

  ordered = points[order]

  return np.array(ordered)

def walk_around_perimeter(dist:float, width:float, height:float) -> Point:
  """
  Walk dist around the perimeter of a rectangle,
  counter-clockwise, starting from the bottom-left.
  """
  dist = dist % (2 * (width + height)) # = dist % perimeter
  if dist < width:
    p0 = dist, 0
  elif dist < (width + height):
    p0 = width, dist - width
  elif dist < (2 * width + height):
    p0 = width - (dist - width - height), height
  else:
    p0 = 0, height - (dist - 2 * width - height)
  return np.array(p0, dtype=float)

def get_corridor(bsp: BSpline, samples: int, bufferSize: float) -> Polygon:
  """
  Get a polygon-corridor, consisting of `splines` many piecewise linear elements,
  with a buffer wide distance to both sides of the sampled spline.

  Args:
    bsp: BSpline object to build a corridor with.
    samples: BSpline is discretized into samples many linear units.
    bufferSize: Polygon contains all (x,y), that are less than `bufferSize` distant to bsp.
  Returns:
    A polygon around the spline.
  """
  u = np.linspace(0.0, 1.0, samples)
  curve = bsp(u)  # shape (samples, 2)
  line = LineString(curve) # Polyline approximating BSpline
  return line.buffer(bufferSize)

def generate_islands(count: int, 
                    corridor: Polygon, 
                    boundary: Boundary) -> MultiPolygon:
  """
  Generate at least `count` islands, with the corridor region staying island free.

  Args:
    count: Minimum number of islands to create
    corridor: A polygon that remains free of islands.
    boundary: Polygon contains all (x,y), that are less than `bufferSize` distant to bsp.
  Returns:
    A reasonably plausible island and the backbone spline.
  """
  curve = None

  islands = []
  scale = np.abs(boundary[1] - boundary[0])

  #Generate and Validate
  while len(islands) < count:
    islePoints = np.random.random((2,2)) * scale + boundary[0] #TODO: Enforce min distance later
    radius = np.clip(
      np.random.uniform(0.2, 0.35) * np.linalg.norm(islePoints[1] - islePoints[0]),
      0.05 * np.mean(scale),
      0.25 * np.mean(scale)
    )
    isle, curve = island(islePoints,radius=radius)

    if isle.is_valid and not any(isle.intersects(i) for i in islands) and not isle.intersects(corridor):
      islands.append(isle)

  return MultiPolygon(islands), curve

def new_environment(boundary, min_border_dist, max_curvature,
                    num_path_points=5, num_islands=6, 
                    corridor_width=3.5, end_radius = 5,
                    scenario = None) -> Environment:
  '''
    Generate a new environment, including a free corridor and non-overlapping islands.
    The free corridor guarantees a navigable path, though simpler paths may exist.
    Args:
      boundary: A numpy (2,2) array. Countains Path. Islands are not necessarily contained by the boundary.
      num_path_points: Number of path points, more points create a more complex path.
      num_islands: Number of islands. Uses generate and validate, thus too big numbers might cause issues.
      corridor_width: Width of the corridor.
      end_radius: Used to create an approximated circle, may have overlaps with islands, 
        but should retain enough free space for a ship to park, 
        iff `corridor_width` is appropriate and `end_radius` > ship's length
      min_border_dist: Avoids generating start, end and mid-points within 
        this distance to the boundary
      max_curvature: Free corridor has a curvature of at most `max_curvature`.
      scenario: Allows for specific island setups, guaranteeing a certain property. 
        May ignore other parameters.
        - None: Default island generation, randomly placed, no overlaps, free corridor.
        - `blocking`: Places a single island on a straight line between start and end.
    Returns:
      A pre-filled `Environment` object.
  '''
    
  #Inner box for points
  point_boundary = np.array([
    boundary[0] + min_border_dist,
    boundary[1] - min_border_dist
  ])

  #Innerbox must still be well defined
  if np.any(point_boundary[1] < point_boundary[0]):
    raise ValueError("Inner box is ill-defined:\n" \
      f"Lower left point:{boundary[0]} + {min_border_dist} = {point_boundary[0]}\n" \
      f"Upper right point:{boundary[1]} - {min_border_dist} = {point_boundary[1]}\n" \
      "First point has to be less than the second one, in each component."
      )

  #Repack boundary into a polygon.
  boxboundary = box(*boundary[0], *boundary[1])

    
  #Bigger boundary so some islands can be outside.
  islandBoundary = (boundary - boundary.mean(axis=0)) * 1.5 + boundary.mean(axis=0)

  #One island blocking the path.
  if scenario == "blocking":
    points = generate_points(2, point_boundary)
    # Pick them in an useful order 
    start, end = points[0], points[1]
    direction = end - start
    path_len = np.linalg.norm(end - start)

    #Island points.
    u1 = np.random.uniform(0.25, 0.4)
    u2 = np.random.uniform(0.6, 0.75)
    pair = np.array([
      start + u1 * direction,
      start + u2 * direction
    ])
    radius = np.random.uniform(0.1, 0.2) * path_len
    while True:
      islands, _ = island(pair, radius)
      if islands.is_valid:
        break
    corridor = Polygon()
  
  #Normal island generation. Creates random non-overlapping islands and
  #a free corridor with a constrained curvature.
  if scenario is None:
    valid_corridor, tries = False, 0
    while not valid_corridor:
      # Get n points for a path
      points = generate_points(num_path_points, point_boundary)
      # Pick them in an useful order 
      points = projection_sort(points)
      start, end = points[0], points[-1]
      #Interpolate them, this is the path center
      spline = interpolate(points)
      #Make the path into a Polygon with sufficient width
      corridor = get_corridor(spline, 500, corridor_width * 0.5)

      valid_corridor = (
        corridor.within(boxboundary) and
        _get_max_curvature(spline) < max_curvature
      )
      tries += 1
      if tries % 1000 == 0:
        print(f"Corridor generation took {tries} attempts!")
      if tries > 10000:
        raise RuntimeError("Corridor took too long to generate, aborting.")
    #Create non-overlapping islands outside of the corridor.
    islands, _ = generate_islands(num_islands, corridor, islandBoundary)


  #Target Area
  end_poly = _get_end_polygon(end, end_radius)
  
  return Environment(islands, corridor, start, end_poly, boxboundary)



def island(pair, radius=10, n=60):
  '''
    Creates a random (often curved) island around pair, with the given radius.
    Islands have a chance to contain self cuts, use shapely `is_valid` to filter out bad islands.
  '''
  #--Island Key Parameters--
  alpha = np.random.uniform(0.3, 0.6) #Symmetry of island sides, 0 is indepedent, 1 is symmetrical
  roughness = np.random.uniform(0.5, 0.8) # Amplitude of Noise
  fpower = np.random.uniform(1.5, 1.9) # Smoothing out higher frequency noise
  #--Create Island Backbone--
  # First step is a curved spline from start to end, 
  # for that we shift a midpoint away from the line defined by pair.
  start, end = pair

  direction = end - start
  L = np.linalg.norm(direction)

  normal = np.array([-direction[1], direction[0]]) / L

  #u lets midpoint travel from A to B
  u = np.random.uniform(0.2, 0.8)
  #v shifts midpoint orthogonal AB
  vmax = min(0.2, L/(2*radius)) # Prevent bad curvature causing self cuts.
  v = np.random.uniform(-vmax, vmax)

  midPoint = start + direction * u + normal * (radius * v)

  #--Curvature Correction--
  #Too steep curvature causes self cuts later on.
  A = start
  C = midPoint
  B = end

  tau = np.linalg.norm(C-A) / (np.linalg.norm(C-A) + np.linalg.norm(B-C))

  a = (C - A - tau*(B-A)) / (tau*(tau-1))
  b = (B-A) - a

  t_star = -(a @ b) / (2 * (a @ a))
  vel = 2*a*t_star + b

  kappa_max = abs(b[0]*a[1] - b[1]*a[0]) * 2 / np.linalg.norm(vel)**3

  if radius * kappa_max > 1:
      scale = 0.8 / (radius * kappa_max)
      v *= scale
      midPoint = start + direction * u + normal * (radius * v)

  #--Fourier-Noise island geography--
  # The island shape is generated by surrounding a spline backbone 
  # with a variable-width mantle. 
  # The mantle thickness is controlled by a Fourier-based noise model, 
  # producing irregular coastlines while preserving an overall 
  # smooth island structure.
  bspline = interpolate(np.array([start,midPoint,end]))
  #Creating a mantle around the backbone
  theta = np.linspace(0, 1, n)

  #Island Noise Parameters
  freqs = np.array([2,3,5,7,11,13])
  amps = roughness / freqs**fpower

  #Offset for each sine curve
  phases1 = np.random.random(len(freqs)) * 2*np.pi
  phases2 = np.random.random(len(freqs)) * 2*np.pi

  angles = freqs[:,None]*theta*2*np.pi

  noise1 = np.sin(angles + phases1[:,None])
  noise2 = np.sin(angles + phases2[:,None])
  # Taper for smooth wrap around
  noise_taper = np.sqrt(np.sin(np.pi * theta))
  #"Reverse" Fourier-Transform, overlap several sine's
  r1    = radius * noise_taper * (1 + (amps[:,None] * noise1).sum(axis=0))

  r_new = radius * noise_taper * (1 + (amps[:,None] * noise2).sum(axis=0))
  
  # Partial correlation of r1 and r2, to weakly couple both island sides.
  r2 = alpha * r1 + (1-alpha) * r_new

  #Cap R to alleviate Self-Cut island slightly.
  r_cap = 1.5 * radius
  r1 = np.minimum(r1, r_cap)
  r2 = np.minimum(r2, r_cap)

  curve = bspline(theta)
  tangent = bspline.derivative()(theta)
  normals = np.column_stack((-tangent[:,1], tangent[:,0]))
  normals /= np.linalg.norm(normals, axis=1)[:,None]

  #Island halves left and right of the spline, connected via taper.
  left  = curve + r1[:,None] * normals
  right = curve - r2[:,None] * normals

  points = np.vstack((left, right[::-1]))

  polygon = Polygon(points)

  #Random rotation of the entire angle
  angle = np.random.uniform(0, 360)
  polygon = rotate(polygon, angle, origin='centroid')

  return polygon, curve

def _get_end_polygon(end: Point, radius: float):
  '''
    Returns a regular 16-gon around end, with the given radius.
  '''
  end = sgeom.Point(*end)
  end = end.buffer(radius, quad_segs=4)
  return end

def _get_max_curvature(spline: BSpline, samples: int = 10_000) -> float:
    """
    Estimate the maximum absolute curvature of a
    2D parametric BSpline by dense sampling.
    """

    t_min = spline.t[spline.k]
    t_max = spline.t[-spline.k - 1]

    ts = np.linspace(t_min, t_max, samples)

    d1 = spline.derivative(1)(ts)
    d2 = spline.derivative(2)(ts)

    x1 = d1[:, 0]
    y1 = d1[:, 1]

    x2 = d2[:, 0]
    y2 = d2[:, 1]

    numerator = np.abs(
        x1 * y2 - y1 * x2
    )

    denominator = (
        x1**2 + y1**2
    )**1.5

    curvature = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0,
    )

    return float(np.max(curvature))