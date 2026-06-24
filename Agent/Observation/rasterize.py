import numpy as np
import skimage as ski
import shapely

from shapely.geometry import MultiPolygon, Polygon, box
from shapely.affinity import rotate
from shapely.geometry.base import BaseGeometry

from Simulation.Auxilaries.shapely_aux import extract_geometries
from Simulation.Environment.agent import Ship, ShipState, get_ship_polygon
from Simulation.Environment.environment import Environment

def get_global_view(env: Environment, ship: Ship, 
                    boundary: np.ndarray, resolution: np.ndarray,
                    lview_size: np.ndarray, pixel_val = 255):
  '''
    Get a boundary sized render,
    rasterized to the graphical target resolution[0] * resolution[1] (rows, cols)
    Pixels are 0 iff empty, `pixel_val` otherwise.

    That includes 
    - scaling up/down,
    - discretizing,
    - flipping y axis,
    - swapping coordinate ordering.
    Out:
      A (3, resolution[0], resolution[1]) Numpy array, containing the following images:
      - Islands
      - End-Polygon
      - The local-view cutout
  '''
  #DEBUGGING
  return np.zeros(
    (3, resolution[0], resolution[0]),
    dtype=np.uint8,
  )
  #Boundary is the input space. We want:
  #boundary intersection islands (boundary obstacle is not included)
  #boundary intersection end
  #
  pos = ship.state.position
  heading = ship.state.heading #Radian
  p1, p2 = pos - lview_size/2, pos + lview_size/2
  pos = shapely.Point(pos)
  rows, cols = resolution

  #Get a box centered on ship
  lview = box(*p1,*p2)

  #Align it with the ship
  lview = rotate(lview, heading, pos, True)

  #Empty view
  poly_boundary = box(*boundary[0],*boundary[1])

  #Extract global views
  islands  = poly_boundary.intersection(env.islands)
  end_poly = poly_boundary.intersection(env.end_poly)
  lview    = poly_boundary.intersection(lview)

  #Prepare for rasterization, still using mathematical references.
  target_box = np.array([
      [-0.5, -0.5],
      [cols - 0.5, rows - 0.5],
  ])

  #Switching to image coordinates
  islands  = map_box_coordinates(islands, boundary, target_box)
  end_poly = map_box_coordinates(end_poly, boundary, target_box)
  lview    = map_box_coordinates(lview, boundary, target_box)

  isl_mask   = rasterize(islands, resolution)
  end_mask   = rasterize(end_poly, resolution)
  lview_mask = rasterize(lview, resolution)

  return np.stack([
    isl_mask,
    end_mask,
    lview_mask,
  ], axis=0).astype(np.uint8) * pixel_val

def get_local_view(env: Environment, ship: Ship, size: np.ndarray, resolution: np.ndarray, pixel_val = 255):
  '''
    Get a geometrical size[0] * size[1] (x,y) cutout centered around the ship, 
    rasterized to the graphical target resolution[0] * resolution[1] (rows, cols)
    Pixels are 0 iff empty, `pixel_val` otherwise.

    That includes 
    - scaling up/down,
    - discretizing,
    - flipping y axis,
    - swapping coordinate ordering.
    Out:
      A (3, resolution[0], resolution[1]) Numpy array, containing the following images:
      - Obstacles (Islands and Boundary)
      - End-Polygon
      - The ship
  '''
  pos = ship.state.position
  heading = ship.state.heading #Radian
  p1, p2 = pos - size/2, pos + size/2
  pos = shapely.Point(pos)
  rows, cols = resolution

  #Get a box centered on ship
  window = box(*p1,*p2)

  #Align it with the ship
  window = rotate(window, heading, pos, True)

  #Extract local views
  obstacles = window.intersection(env.islands).union(window.difference(env.boundary))
  end_poly = window.intersection(env.end_poly)
  ship_poly = get_ship_polygon(ship)

  if not window.covers(ship_poly):
    print(f"Ship: {ship_poly}")
    print(f"Window: {window}")
    print(f"Ship part outside of window: {ship_poly.difference(window)}")
    raise ValueError("Local view does not contain ship.")
  
  #Rotate back
  obstacles = rotate(obstacles, -heading, pos, True)
  end_poly  = rotate(end_poly , -heading, pos, True)
  ship_poly = rotate(ship_poly, -heading, pos, True)

  #Prepare for rasterization, still using mathematical references.
  source_box = (p1,p2)
  target_box = np.array([
      [-0.5, -0.5],
      [cols - 0.5, rows - 0.5],
  ])
  #Switching to image coordinates
  obstacles = map_box_coordinates(obstacles, source_box, target_box)
  end_poly  = map_box_coordinates(end_poly , source_box, target_box)
  ship_poly = map_box_coordinates(ship_poly, source_box, target_box)

  obs_mask  = rasterize(obstacles, resolution)
  end_mask  = rasterize(end_poly , resolution)
  ship_mask = rasterize(ship_poly, resolution)

  return np.stack([
    obs_mask,
    end_mask,
    ship_mask,
  ], axis=0).astype(np.uint8) * pixel_val

def rasterize(geom: BaseGeometry, resolution):
  '''
    Rasterizes the areal part of the geometry into a binary mask
    of shape `resolution`.

    A pixel is `True` iff it is considered occupied by the geometry
    according to the underlying rasterization method.

    Uses graphical indexing, hence 
      `rows, cols = resolution`
    Args:
      geom:
        Any geometry to rasterize, draws nothing if geom.area is too small.
        Only draws polygons contained in geom, omitting the rest.
      resolution:
        rows, cols of the window cutout to rasterize on.
        Any polygon parts outside of rectangle from (0,0) to (rows,cols) are omitted.
  '''
  rows, cols = resolution
  mask = np.zeros((rows, cols), dtype=bool)

  #Return 0 mask, because there is (basically) nothing.
  if geom.area < 1e-12:
    return mask
  
  #We can only render polygons one by one.
  polys = extract_geometries(geom, ["Polygon"])

  #Collects exteriors and interiors.
  #Shapely, uses x,y coordinates
  #skimage uses rows, cols, that is y,x coordinates, 
  #hence necessitating a coordinate ordering swap.
  exs, ins = [], []
  for poly in polys:
    exs.append(
      np.asarray(poly.exterior.coords[:-1])[:, [1,0]]
    )
    for interior in poly.interiors:
      ins.append(
        np.asarray(interior.coords[:-1])[:, [1,0]]
      )

  #Collect all positive pixels, we have a pixel, if we have any polygon there.
  for e in exs:
    mask |= ski.draw.polygon2mask(resolution,e)

  #Remove holes, we only keep pixels that have no hole
  for i in ins:
    mask &= ~ski.draw.polygon2mask(resolution,i)

  return mask

def map_box_coordinates(geom: BaseGeometry, source_box: np.ndarray, target_box: np.ndarray, y_flip = True):
  '''
    Given an affine transformation fully defined by mapping `source_box` to `target_box`
    and optionally flipping the y axis,
    this method applies that transformation to `geom`.
  
    `source_box` and `target_box` are both ((xmin, ymin), (xmax, ymax))

    Note that geom does not need to be contained in source_box, the mapping is well defined in either case.
    Args:
      geom: Geometry to transform, relation to the boxes is irrelevant for correctness.
      source_box: ((xmin,ymin), (xmax, ymax)), axis aligned
      target_box: ((xmin,ymin), (xmax, ymax)), axis aligned
      y_flip: Iff y_flip, additionally converts to image-style basis with downward-pointing y axis.
  '''
  def make_transform_fn(source_box: np.ndarray, target_box: np.ndarray, y_flip):
    s1, s2 = source_box
    t1, t2 = target_box
    #Given a point p to transform:
    def fn(p):
      p = p.copy()
      p -= s1 # Shift to origin
      p /= s2 - s1 # Normalize into unit square
      if y_flip: # flip y inside unit square
        p[:,1] = 1 - p[:,1]
      p *= t2 - t1 # scale up to target.
      p += t1 # move to target area
      return p
    
    return fn
  return shapely.transform(geom, make_transform_fn(source_box, target_box, y_flip))