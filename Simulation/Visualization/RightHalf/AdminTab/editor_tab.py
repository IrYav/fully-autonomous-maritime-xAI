from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
  QAbstractButton, QButtonGroup, QGridLayout, QGroupBox, QHBoxLayout, QPushButton, QRadioButton, QToolButton, QVBoxLayout,
  QWidget
  )
from matplotlib.patches import PathPatch

from Simulation.Visualization.Mouse.mouse_tool_actions import MouseToolActions, Tool
from Simulation.Visualization.RightHalf.AdminTab.editor_controller import EditorController
from Simulation.Visualization.RightHalf.AdminTab.mouse_modes import CreateMode, TransformMode
from Simulation.Visualization.RightHalf.AdminTab.draft_model import Material
from Simulation.Visualization.RightHalf.AdminTab.reset import reset_ship, reset_state

from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from Simulation.Visualization.visualization import EnvironmentViewer


class EditorTab(QWidget):
  ec : EditorController
  
  def __init__(self, 
               ec : EditorController, 
               mta: MouseToolActions,
               env_view: EnvironmentViewer,
               ):
    super().__init__()
    self.mta = mta
    self.ec = ec
    self.env_view = env_view
    sections = QVBoxLayout(self)

    sections.addWidget(self._islands_section())
    sections.addWidget(self._selection_section())
    sections.addWidget(self._reset_section())
    sections.addStretch(1)
    sections.addWidget(self._bottom_buttons())

    self._connect_signals()

  #Called on tab leave
  def hideEvent(self, event):
    #Closed editor, so no edit mode is active.
    #Also clears various creation doodles.
    self.mta.clear()
    self.ec.clear_pv()
    super().hideEvent(event)
  
  def _islands_section(self) -> QWidget:
    #---LAYOUT---
    #Local main container
    section = QWidget()
    section_layout = QVBoxLayout(section)

    button_widget = QGroupBox("Creation")
    self.button_lay = QHBoxLayout(button_widget)
    self.mat_widget = QGroupBox("Material")
    self.mat_lay = QVBoxLayout(self.mat_widget)

    section_layout.addWidget(button_widget)
    section_layout.addWidget(self.mat_widget)
    section_layout.addStretch(1)

    #---BUTTONS---

    # Manual mode
    self.buttons_man = QHBoxLayout()
    self.manual = QToolButton()
    self.manual.setText("Manual")
    self.manual.setCheckable(True)

    self.finish = QPushButton("Finish")
    self.buttons_man.addWidget(self.manual)
    self.buttons_man.addWidget(self.finish)

    # Auto mode
    self.autom = QPushButton("Automatic Island")

    self.button_lay.addLayout(self.buttons_man)
    self.button_lay.addWidget(self.autom)

    #---MATERIAL SELECTOR---

    land = QRadioButton("Land")
    void = QRadioButton("Void")
    end = QRadioButton("End")

    land.setProperty("material", Material.LAND)
    void.setProperty("material", Material.VOID)
    end.setProperty("material", Material.END)
    end.setToolTip("Only the last end will be retained on commit." \
      " To have a meaningful navigation task," \
      " there has to be exactly one end within the boundary.")

    self.group = QButtonGroup(section)
    self.group.addButton(land)
    self.group.addButton(void)
    self.group.addButton(end)

    land.setChecked(True)

    self.mat_lay.addWidget(land)
    self.mat_lay.addWidget(void)
    self.mat_lay.addWidget(end)

    #--- CREATE MOUSE MODES ---
    self.mta.set_editor(self.ec)
    self.manual_action = self.mta.actions[Tool.CREATE]

    #--- CONNECT ---
    #Mouse modes need special treatment.
    self.manual.setDefaultAction(self.manual_action)
    #Connect signals, to update correctly on mouse input.



    #Normal buttons
    self.finish.clicked.connect(self._on_finish_clicked)
    self.autom.clicked.connect(self.ec.on_auto)

    self.group.buttonClicked.connect(self._on_material_changed)

    return section

  def _selection_section(self) -> QWidget:
    section = QWidget()
    section_layout = QVBoxLayout(section)

    group = QGroupBox("Selection")
    section_layout.addWidget(group)

    group_layout = QHBoxLayout(group)

    self.select = QToolButton()
    self.select.setDefaultAction(self.mta.actions[Tool.SELECT])

    self.delete = QPushButton()
    self.delete.setText("Delete")
    self.delete.clicked.connect(self.ec.on_delete)
    self.delete.setEnabled(False) # Nothing selected on start

    self.transform_group = QButtonGroup(section)
    self.transform_group.setExclusive(True)

    self.move_action = QRadioButton("Move")
    self.rotate_action = QRadioButton("Rotate")
    self.scale_action = QRadioButton("Scale")

    self.transform_group.addButton(self.move_action)
    self.transform_group.addButton(self.rotate_action)
    self.transform_group.addButton(self.scale_action)

    self.move_action.setChecked(True)
    self.move_action.toggled.connect(
      lambda checked: self.mta.modes[Tool.SELECT].set_transform_mode(
        TransformMode.MOVE if checked else TransformMode.MOVE
      )
    )
    self.rotate_action.toggled.connect(
      lambda checked: self.mta.modes[Tool.SELECT].set_transform_mode(
        TransformMode.ROTATE if checked else TransformMode.MOVE
      )
    )
    self.scale_action.toggled.connect(
      lambda checked: self.mta.modes[Tool.SELECT].set_transform_mode(
        TransformMode.SCALE if checked else TransformMode.MOVE
      )
    )

    group_layout.addWidget(self.select)
    group_layout.addWidget(self.move_action)
    group_layout.addWidget(self.rotate_action)
    group_layout.addWidget(self.scale_action)
    group_layout.addWidget(self.delete)
    return section

  def _bottom_buttons(self) -> QWidget:
    section = QWidget()
    section_layout = QHBoxLayout(section)

    self.commit = QPushButton()
    self.commit.setText("Commit changes and recalculate everything.")
    self.commit.setToolTip("Without a commit, " \
      "the agent won't see any changes in the environment")
    self.commit.clicked.connect(self.ec.on_commit)

    section_layout.addWidget(self.commit)

    return section

  def _reset_section(self) -> QWidget:
    section = QGroupBox("Reset")
    layout = QGridLayout(section)

    self.set_state = QPushButton("Set State")
    self.set_ship = QPushButton("Set Ship")
    self.reset_state = QPushButton("Reset State")
    self.reset_ship = QPushButton("Reset Ship")
    self.reset_all = QPushButton("Reset All")

    layout.addWidget(self.set_state,   0, 0)
    layout.addWidget(self.reset_state, 0, 1)

    layout.addWidget(self.set_ship,    1, 0)
    layout.addWidget(self.reset_ship,  1, 1)

    layout.addWidget(self.reset_all,   2, 0, 1, 2)

    self.set_state.clicked.connect(self.ec.set_reset_state)
    self.set_ship.clicked.connect(self.ec.set_reset_ship)
    self.reset_state.clicked.connect(lambda: reset_state(self.ec))
    self.reset_ship.clicked.connect(lambda: reset_ship(self.ec))
    self.reset_all.clicked.connect(self.ec.reset_all)
    return section
  
  def _connect_signals(self):
    # render draft + preview
    self.ec.draft_changed.connect(self._on_changed_signal)
    # render preview
    self.ec.preview_changed.connect(self._on_pv_changed_signal) 
    # Delete button toggle
    self.ec.selected_changed.connect(self._set_delete_toggle)

  def _on_changed_signal(self):
    self.env_view.update_draft()
    self.finish.setEnabled(self.mta.mouse_tool.can_finish)

  def _on_pv_changed_signal(self, artists: list[PathPatch]):
    self.env_view.set_preview(artists)
    self.finish.setEnabled(self.mta.mouse_tool.can_finish)

  def _set_delete_toggle(self):
    '''Ensure delete button is only conditionally available.'''
    can_delete, tooltip = self.ec.can_delete()
    self.delete.setEnabled(can_delete)
    self.delete.setToolTip(tooltip)

  def _on_finish_clicked(self) -> None:
    """
    Finish the current manual polygon.

    If the constructed polygon is valid, it is added to the current
    scenario draft using the currently selected material.
    """
    mt = self.mta.mouse_tool
    mt.on_finish()

  def _on_material_changed(self, button: QAbstractButton) -> None:
    """
    Update the currently selected material.
    """
    material = button.property("material")
    self.ec.on_set_material(material)
