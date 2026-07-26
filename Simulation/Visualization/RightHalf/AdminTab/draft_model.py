from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from collections.abc import Iterator
from shapely.geometry import Polygon

from Simulation.Auxilaries.shapely_aux import extract_geometries
from Simulation.Environment.environment import Environment


class Material(Enum):
  """Material assigned to an area object."""

  LAND = auto()
  VOID = auto()
  END = auto()

  @property
  def artist_name(self):
    return {
      Material.LAND: "islands",
      Material.VOID: "cutout",
      Material.END: "end",
    }[self]


@dataclass(slots=True)
class AreaObject:
  """
  Represents a single editable area within the scenario.

  Objects are uniquely identified by their ID. The rendering order is
  determined externally by the order list stored in the draft.
  """

  id: int
  polygon: Polygon
  material: Material = Material.LAND


@dataclass(slots=True)
class EnvironmentDraft:
  """
  Editable working copy of a scenario.

  The draft is independent from the simulation pipeline and only stores the
  high-level scene description. Expensive data such as meshes or potential
  fields are rebuilt only after the draft is committed.

  Objects are addressed by their unique ID, while the drawing order is stored
  separately. The last ID in ``order`` is rendered on top.
  """

  objects: dict[int, AreaObject] = field(default_factory=dict)
  order: list[int] = field(default_factory=list)

  def get(self, object_id: int) -> AreaObject:
    """Return the requested object."""
    return self.objects[object_id]

  def contains(self, object_id: int) -> bool:
    """Return whether the object exists."""
    return object_id in self.objects

  def add(self, obj: AreaObject) -> None:
    """
    Add a new object to the draft.

    Raises:
      ValueError: If the object ID already exists.
    """
    if obj.id in self.objects:
      raise ValueError(f"Object ID {obj.id} already exists.")

    self.objects[obj.id] = obj
    self.order.append(obj.id)

  def remove(self, object_id: int) -> None:
    """
    Remove an object from the draft.

    Raises:
      KeyError: If the object does not exist.
    """
    del self.objects[object_id]
    self.order.remove(object_id)

  def move_to_front(self, object_id: int) -> None:
    """Move an object to the front of the drawing order."""
    self.order.remove(object_id)
    self.order.append(object_id)

  def move_to_back(self, object_id: int) -> None:
    """Move an object to the back of the drawing order."""
    self.order.remove(object_id)
    self.order.insert(0, object_id)

  def clear(self) -> None:
    self.object.clear()
    self.order.clear()

  @staticmethod
  def draft_from_env(env: Environment) -> EnvironmentDraft:
    '''
      Create a draft copy of the environment.
    '''
    draft = EnvironmentDraft()
    islands = extract_geometries(env.islands,["Polygon"])
    for island in islands:
      draft.add(AreaObject(draft.next_id, island))
    draft.add(AreaObject(draft.next_id, env.end_poly, Material.END))
    return draft

  def draft_to_env(
      self, 
      old_env: Environment, 
      update_draft = True) -> Environment:
    '''
      Processes the current draft to an Environment ready for 
      further processing.

      Touching islands will be fused, cutouts will be removed. 
      Everything happens in iteration order.

      In case of multiple `AreaObject`s with `Material.END`, 
      only the final one will be kept.

      Iff `update_draft`, the result will be used to update 
      the current draft, setting 
      the current state to `draft_from_env(result)`
    '''
    islands = Polygon()
    end = Polygon()

    for ao in self:
      match ao.material:
        case Material.LAND:
          islands = islands.union(ao.polygon)
        case Material.VOID:
          islands = islands.difference(ao.polygon)
        case Material.END:
          end = ao.polygon

    result = Environment(
      islands,
      old_env.corridor,
      old_env.start, 
      end, 
      old_env.boundary)
    
    if update_draft:
      self.assign(self.draft_from_env(result))

    return result

  def assign(self, new_draft: EnvironmentDraft):
    self.objects = new_draft.objects
    self.order = new_draft.order

  def __iter__(self) -> Iterator[AreaObject]:
    '''
      Iterate over all objects in drawing order, back to front.
    '''
    for object_id in self.order:
      yield self.objects[object_id]

  def __reversed__(self) -> Iterator[AreaObject]:
    for object_id in reversed(self.order):
      yield self.objects[object_id]

  @property
  def next_id(self) -> int:
    '''Return the next unused object ID.'''
    if not self.objects:
      return 0
    return max(self.objects) + 1
  
