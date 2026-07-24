from __future__ import annotations

from abc import ABC
import numpy as np

class MouseMode(ABC):
  '''
    Abstract base class to implement different mouse modi.

    Intended to be used as a strategy pattern.

    Implementation differs from actual subclass present.

    Usage should not require too much knowledge about 
    the subclass.
  '''
  def on_activate(self):
    '''
      What will happen, if this mode is activated?
    '''
    pass

  def on_deactivate(self):
    '''
      What will happen, if this mode is deactivated?
    '''
    pass

  def on_press(self, pos):
    '''
      What will happen, on a click, at `pos`?
    '''
    pass

  def on_move(self, pos):
    '''
      What will happen, on move mousement, at `pos`?
    '''
    pass

  def on_release(self, pos):
    '''
      What will happen, on a mouse release, at `pos`?
    '''
    pass

  @property
  def can_finish(self) -> bool:
    '''
      Does the mouse mode support a meaningful finish button?

      Can it be currently used? This can depend on the current state,
      in addition to the choice of the current `MouseMode` instance.
      (e.g., think of finishing a "polygon" with n points)
    '''
    return False
  
  def on_finish(self):
    '''
      How will the current mode react on a finish button click?
    '''
    pass

  def create_artist(self):
    '''
      How (if at all) will this mode be visualized? 
    '''
    pass

  def on_drag_begin(self, pos):
    '''
      What will happen when dragging begins
    '''
    pass

  def on_drag(self, delta):
    '''
      What happens on each quantified drag unit?
    '''
    pass

  def on_drag_end(self, pos):
    '''
      What happens on drag end?
    '''
    pass