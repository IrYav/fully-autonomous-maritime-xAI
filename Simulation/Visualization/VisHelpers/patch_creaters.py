from matplotlib.path import Path
from matplotlib.patches import PathPatch
from shapely import LineString, LinearRing


def line_patch(
  line: LineString | LinearRing,
  *,
  color="#0072B2",
  alpha=1,
  linewidth=2,
) -> PathPatch:
  '''
    Returns a matplotlib PathPatch object, according to 
    the given `LineString` or `LinearRing`
  '''
  coords = list(line.coords)

  path = Path(
    coords,
    [Path.MOVETO] + [Path.LINETO] * (len(coords) - 1),
  )

  return PathPatch(
    path,
    facecolor="none",
    edgecolor=color,
    linewidth=linewidth,
    alpha=alpha,
  )