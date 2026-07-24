from PySide6.QtGui import QAction, QActionGroup


from enum import Enum, auto

from Simulation.Visualization.Mouse.mouse_mode import MouseMode
from Simulation.Visualization.Mouse.mouse_tool import MouseTool
from Simulation.Visualization.RightHalf.AdminTab.cds_modes import CreateMode, SelectMode
from Simulation.Visualization.RightHalf.AdminTab.editor_controller import EditorController

class Tool(Enum):
    SELECT = auto()
    CREATE = auto()

class MouseToolActions:
  '''
    Central interface to change MouseMode in MouseTool
  '''    
  actions: dict[Tool, QAction]
  modes: dict[Tool, MouseMode]
  def __init__(self, mouse_tool: MouseTool, parent=None):
    self.mouse_tool = mouse_tool

    self._group = QActionGroup(parent)
    self._group.setExclusionPolicy(
      QActionGroup.ExclusionPolicy.ExclusiveOptional
    )

    self.actions = {}
    self.modes = {}

  def register(
    self,
    tool: Tool,
    text: str,
    mode: MouseMode,
  ) -> QAction:
    action = QAction(text)
    action.setCheckable(True)

    self._group.addAction(action)

    self.actions[tool] = action
    self.modes[tool] = mode

    action.toggled.connect(
      lambda checked, t=tool:
        self.mouse_tool.set_mode(self.modes[t] if checked else None)
    )
    
    return action

  def set_editor(self, ec : EditorController):
    '''
      Activates all Tools requiring an `EditorController`
    '''
    if Tool.CREATE not in self.modes: #Avoid double setting
      self.register(Tool.CREATE, "Create", CreateMode(ec))
      self.register(Tool.SELECT, "Select", SelectMode(ec)) # TODO change later.

  def clear(self):
    action = self._group.checkedAction()
    if action is not None:
        action.setChecked(False)

  def activate(self, tool: Tool):
    print("Activated: ", tool)
    self.actions[tool].setChecked(True)
