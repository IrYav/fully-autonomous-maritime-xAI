import random
import numpy as np

from scipy.interpolate import make_interp_spline, BSpline
from shapely.geometry.base import BaseGeometry
from shapely.geometry import LineString, Polygon, MultiPolygon
from shapely.affinity import rotate
from shapely.ops import unary_union

Point  = np.ndarray  # shape (2,)
Points = np.ndarray  # shape (n,2)
Boundary = np.ndarray # shape (2,2)

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
  result.append(p1)

  #Inner control points.
  for n in range (2,n):
    # Random for now, so we have something working. Prolly needs to be better
    rpoint = np.array(
      (random.random() * width , 
       random.random() * height), dtype=float)
    result.append(rpoint + boundary[0])
  
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

def get_corridor(bsp: BSpline, samples: int, bufferSize: float) -> BaseGeometry:
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
                    boundary: Boundary, 
                    max_attempts: int = 10_000) -> MultiPolygon:
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
  def island(pair, R=10, n=60):

    #--Island Key Parameters--
    alpha = np.random.uniform(0.3, 0.6) #Symmetry of island sides, 0 is indepedent, 1 is symmetrical
    roughness = np.random.uniform(0.5, 0.8) # Amplitude of Noise
    fpower = np.random.uniform(1.5, 1.9) # Smoothing out higher frequency noise
    #--Create Island Backbone--
    start, end = pair

    direction = end - start
    L = np.linalg.norm(direction)

    normal = np.array([-direction[1], direction[0]]) / L

    #u lets midpoint travel from A to B
    u = np.random.uniform(0.2, 0.8)
    #v shifts midpoint orthogonal AB
    vmax = min(0.2, L/(2*R)) # Prevent bad curvature causing self cuts.
    v = np.random.uniform(-vmax, vmax)

    midPoint = start + direction * u + normal * (R * v)

    #--Curvature Correction--
    A = start
    C = midPoint
    B = end

    tau = np.linalg.norm(C-A) / (np.linalg.norm(C-A) + np.linalg.norm(B-C))

    a = (C - A - tau*(B-A)) / (tau*(tau-1))
    b = (B-A) - a

    t_star = -(a @ b) / (2 * (a @ a))
    vel = 2*a*t_star + b

    kappa_max = abs(b[0]*a[1] - b[1]*a[0]) * 2 / np.linalg.norm(vel)**3

    if R * kappa_max > 1:
        scale = 0.8 / (R * kappa_max)
        v *= scale
        midPoint = start + direction * u + normal * (R * v)

    # ---------------------------------------------------------------
    
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
    r1    = R * noise_taper * (1 + (amps[:,None] * noise1).sum(axis=0))

    r_new = R * noise_taper * (1 + (amps[:,None] * noise2).sum(axis=0))
    
    # Partial correlation of r1 and r2, to weakly couple both island sides.
    r2 = alpha * r1 + (1-alpha) * r_new

    #Cap R to alleviate Self-Cut island slightly.
    r_cap = 1.5 * R
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
    angle = np.random.uniform(0, 360)
    polygon = rotate(polygon, angle, origin='centroid')
    return polygon, curve

  islands = []
  scale = np.abs(boundary[1] - boundary[0])

  islandCount = 15#random.randint(3,3)

  #Generate and Validate
  while len(islands) < islandCount:
    islePoints = np.random.random((2,2)) * scale + boundary[0] #Enforce min distance later

    radius = np.clip(
      np.random.uniform(0.2, 0.35) * np.linalg.norm(islePoints[1] - islePoints[0]),
      0.05 * np.mean(scale),
      0.25 * np.mean(scale)
    )
    isle, curve = island(islePoints,R=radius)

    if isle.is_valid and not any(isle.intersects(i) for i in islands) and not isle.intersects(corridor):
      islands.append(isle)

  return MultiPolygon(islands), curve

#Different difficulty levels
#Maybe a few hand picked ones.

