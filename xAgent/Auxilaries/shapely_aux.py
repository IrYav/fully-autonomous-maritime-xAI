from shapely.geometry import LineString, Polygon
from shapely.geometry.base import BaseGeometry

def flatten_lines(lines) -> list[LineString]:
  '''
    Given a (Multi)LineString, turns them into a list of LineStrings.
    `None` for other Geometries
  '''
  if lines.geom_type in ["LineString", "MultiLineString"]:
    if lines.is_empty:
      return []
    if lines.geom_type == "LineString":
      return [lines]
    if lines.geom_type == "MultiLineString":
      return list(lines.geoms)
  else:
    return []

def flatten_polygons(polys) -> list[Polygon]:
  if polys.geom_type in ["Polygon", "MultiPolygon"]:
    if polys.is_empty:
      return []
    if polys.geom_type == "Polygon":
      return [polys]
    if polys.geom_type == "MultiPolygon":
      return list(polys.geoms)
  else:
    return []

def debug_geometry(g: BaseGeometry, first_msg = "Debugged Geometry:" ):
  print(first_msg)
  print(
    f"Type: {type(g)}\n"
    f"Valid: {getattr(g, 'is_valid', 'n/a')}\n"
    f"Area: {getattr(g, 'area', 'n/a')}"
  )
  print(g)