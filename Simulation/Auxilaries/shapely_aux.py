
from typing import Collection
from shapely.geometry import LineString, Polygon
from shapely.geometry.base import BaseGeometry

def flatten_lines(lines) -> list[LineString]:
  '''
    Given a (Multi)LineString, turns them into a list of LineStrings.
    `[]` otherwise.
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
  '''
    Given a (Multi)Polygon, turns them into a list of Polygon.
    `[]` otherwise.
  '''
  if polys.geom_type in ["Polygon", "MultiPolygon"]:
    if polys.is_empty:
      return []
    if polys.geom_type == "Polygon":
      return [polys]
    if polys.geom_type == "MultiPolygon":
      return list(polys.geoms)
  else:
    return []
  
def extract_geometries(geom: BaseGeometry, geom_types: Collection[str]) -> list[BaseGeometry]:
  '''
    Collects all `geom_types` in `geom`. 
    If `geom_types` also has a container (e.g., MultiPolygon), unpacking is minimized.
    
    For example with `geom_types = ["Polygon","MultiPolygon"]`,
    every MultiPolygon is returned "as is", rather than being flattened to a list of polygons.
  '''
  if geom.is_empty:
    return []

  if geom.geom_type in geom_types:
    return [geom]

  if hasattr(geom, "geoms"):
    result = []

    for subgeom in geom.geoms:
      result.extend(
        extract_geometries(
          subgeom,
          geom_types,
        )
      )

    return result

  return []


def debug_geometry(g: BaseGeometry, first_msg = "Debugged Geometry:" ):
  print(first_msg)
  print(
    f"Type: {type(g)}\n"
    f"Valid: {getattr(g, 'is_valid', 'n/a')}\n"
    f"Area: {getattr(g, 'area', 'n/a')}"
  )
  print(g)