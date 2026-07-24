
from matplotlib.patches import PathPatch
from shapely import LineString, Point, Polygon
from shapely.affinity import translate

from Simulation.Environment.environment import Points
from Simulation.Visualization.Mouse.mouse_mode import MouseMode
from Simulation.Visualization.RightHalf.AdminTab.draft_model import AreaObject
from Simulation.Visualization.RightHalf.AdminTab.editor_controller import EditorController
from Simulation.Visualization.VisHelpers.patch_creaters import line_patch


class CreateMode(MouseMode):
  '''
    CreateMode is used to create a polygon by sequentially
    creating the exterior points of it.
  '''
  curr_points: list[Points]
  curr_pos : Point | None
  ec: EditorController

  def __init__(self, ec: EditorController):
    super().__init__()
    self.ec = ec

  def on_activate(self):
    self._clear()
  
  def on_deactivate(self):
    self._clear()
    self.ec.preview_changed.emit([]) #Wipe preview.

  def on_press(self, pos):
    self.curr_pos = Point(pos)

    if self._can_add:
      self.curr_points.append(self.curr_pos)
      self.ec.preview_changed.emit(self._get_mode_artist())

  def on_move(self, pos):
    self.curr_pos = Point(pos)
    artists = self._get_mode_artist()
    self.ec.preview_changed.emit(artists)

  def on_finish(self):
    if self.can_finish:
      self.ec.add_man_area(self.curr_points)
      self._clear()
      self.ec.preview_changed.emit([]) #Wipe preview

  @property
  def can_finish(self):
    return (
      len(self.curr_points) >= 3 and 
      Polygon(self.curr_points).is_valid
    )
  
  def _clear(self):
    self.curr_points = []
    self.curr_pos = None

  def _get_mode_artist(self) -> list[PathPatch]:

    if len(self.curr_points) == 0:
      #No draft at all
      return []
    
    if self._can_add:
      #Valid point
      last_part = line_patch(
        LineString([self.curr_points[-1], self.curr_pos]),
        color="#0072B2",
        alpha=0.5,
        linewidth=2,
      )
    else:
      #Invalid point
      last_part = line_patch(
        LineString([self.curr_points[-1], self.curr_pos]),
        color="#D55E00",
        alpha=0.5,
        linewidth=2,
      )
    if self.can_finish:
      #Valid draft
      parts = line_patch(
        LineString(self.curr_points),
        color="#0072B2",
        alpha=0.8,
        linewidth=3,
      )
    
    elif len(self.curr_points) > 1 :
      #Finish currently not possible, but additional points may correct it.
      parts = line_patch(
        LineString(self.curr_points),
        color="#E69F00",
        alpha=0.8,
        linewidth=3,
      )
    else: #No parts available
      return [last_part]
    
    return [parts, last_part]
    
  @property
  def _can_add(self) -> bool:

    #Basically no self-cuts
    return (
      len(self.curr_points) == 0 or
      LineString([*self.curr_points, self.curr_pos]).is_simple
    )
  
class SelectMode(MouseMode):
  ec: EditorController

  def __init__(self, ec: EditorController):
    super().__init__()
    self.ec = ec

  def on_move(self, pos):
    '''
      Highlight the currently hovered polygon, iff None is selected.
    '''
    if self.ec.selected is not None:
      return #Only highlight selected one
    elif self._top_ao(pos) is not None:
      self.ec.preview_changed.emit(self._get_mode_artist())
    else:
      self.ec.preview_changed.emit([])

  def on_press(self, pos):
    top_ao = self._top_ao(pos)
    if self.ec.selected is not top_ao:
      self.ec.selected = top_ao #Can assign None again.
      if self.ec.selected is not None:
        self.ec.to_front(self.ec.selected)
      self.ec.preview_changed.emit(self._get_mode_artist())

  def on_drag_begin(self, pos):
    self.ec.selected = self._top_ao(pos)

  def on_drag(self, delta):
    if self.ec.selected is None:
      return
    else:
      self.ec.selected.polygon = translate(self.ec.selected.polygon, *delta)
      #Update polygon itself and selection highlighter rendering
      self.ec.changed.emit()
      self.ec.preview_changed.emit(self._get_mode_artist())

  def _top_ao(self, pos) -> AreaObject | None:
    '''
      Returns first `AreaObject` from the top (or `None`)
    '''
    pos = Point(pos)

    #First the draft
    for ao in reversed(self.ec.draft):
      if pos.within(ao.polygon):
        return ao
      
    return None
  
  def _get_mode_artist(self) -> list[PathPatch]:
      if self.ec.selected is None:
        return []
      return [line_patch(
        self.ec.selected.polygon.exterior,
        color="#0072B2",
      )] 