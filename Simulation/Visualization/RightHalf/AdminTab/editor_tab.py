from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
  QAbstractButton, QAbstractSpinBox, QButtonGroup, QDoubleSpinBox, QFormLayout, QGridLayout, QGroupBox, QHBoxLayout, QPushButton, QRadioButton, QToolButton, QVBoxLayout,
  QWidget, QFileDialog, QMessageBox
  )
from matplotlib.patches import PathPatch

from Simulation.Visualization.Mouse.mouse_tool_actions import MouseToolActions, Tool
from Simulation.Visualization.RightHalf.AdminTab.editor_controller import EditorController
from Simulation.Visualization.RightHalf.AdminTab.mouse_modes import CreateMode, TransformMode
from Simulation.Visualization.RightHalf.AdminTab.draft_model import (
  Material, heading_degrees_to_radians, heading_radians_to_degrees,
  normalized_velocity_to_simulation,
)
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
    sections.addWidget(self._ship_section())
    sections.addWidget(self._reset_section())
    sections.addWidget(self._scenario_section())
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
    self.commit.clicked.connect(self._on_commit_clicked)

    section_layout.addWidget(self.commit)

    return section

  def _ship_section(self) -> QWidget:
    section = QGroupBox("Ship")
    layout = QFormLayout(section)

    self.ship_x = self._number_input()
    self.ship_y = self._number_input()
    self.ship_heading = self._number_input()
    self.ship_velocity = self._number_input()

    layout.addRow("Position X", self.ship_x)
    layout.addRow("Position Y", self.ship_y)
    layout.addRow("Heading (degrees)", self.ship_heading)
    layout.addRow("Velocity (× max)", self.ship_velocity)

    for field in (self.ship_x, self.ship_y, self.ship_heading, self.ship_velocity):
      field.editingFinished.connect(self._on_ship_field_edited)

    self._refresh_ship_controls()
    return section

  @staticmethod
  def _number_input() -> QDoubleSpinBox:
    field = QDoubleSpinBox()
    field.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
    # QDoubleSpinBox rejects NaN/infinity; this wide finite range also keeps
    # the editor from imposing a simulation-specific practical limit.
    field.setRange(-1e100, 1e100)
    field.setDecimals(12)
    field.setSingleStep(1.0)
    return field

  def _refresh_ship_controls(self) -> None:
    draft = getattr(self.ec, "draft", None)
    ship = getattr(draft, "ship", None)
    fields = (self.ship_x, self.ship_y, self.ship_heading, self.ship_velocity)
    enabled = ship is not None
    for field in fields:
      field.setEnabled(enabled)
      field.blockSignals(True)
    if ship is not None:
      self.ship_x.setValue(float(ship.state.position[0]))
      self.ship_y.setValue(float(ship.state.position[1]))
      self.ship_heading.setValue(heading_radians_to_degrees(ship.state.heading))
      self.ship_velocity.setValue(
        float(ship.state.velocity) / float(ship.params.max_velocity)
      )
    for field in fields:
      field.blockSignals(False)

  def _on_ship_field_edited(self) -> None:
    self._apply_ship_controls()

  def _apply_ship_controls(self) -> None:
    draft = getattr(self.ec, "draft", None)
    ship = getattr(draft, "ship", None)
    if ship is None:
      return
    heading = self.ship_heading.value() % 360.0
    self.ship_heading.setValue(heading)
    ship.state.position[0] = self.ship_x.value()
    ship.state.position[1] = self.ship_y.value()
    ship.state.heading = heading_degrees_to_radians(heading)
    ship.state.velocity = normalized_velocity_to_simulation(
      self.ship_velocity.value(), ship.params.max_velocity
    )
    self.ec.draft_changed.emit()

  def _on_commit_clicked(self) -> None:
    self._apply_ship_controls()
    self.ec.on_commit()

  def _scenario_section(self) -> QWidget:
    section = QGroupBox("Scenario")
    layout = QHBoxLayout(section)

    self.save_scenario = QPushButton("Save Scenario")
    self.load_scenario = QPushButton("Load Scenario")
    self.export_screenshot = QPushButton("Export Screenshot")
    layout.addWidget(self.save_scenario)
    layout.addWidget(self.load_scenario)
    layout.addWidget(self.export_screenshot)

    self.save_scenario.clicked.connect(self._on_save_scenario)
    self.load_scenario.clicked.connect(self._on_load_scenario)
    self.export_screenshot.clicked.connect(self._on_export_screenshot)
    return section

  def _on_save_scenario(self) -> None:
    path, _ = QFileDialog.getSaveFileName(
      self, "Save Scenario", "scenario.json", "Scenario files (*.json);;All files (*)"
    )
    if not path:
      return
    try:
      self.ec.save_scenario(path)
    except (OSError, ValueError) as exc:
      QMessageBox.critical(self, "Could not save scenario", str(exc))

  def _on_load_scenario(self) -> None:
    path, _ = QFileDialog.getOpenFileName(
      self, "Load Scenario", "", "Scenario files (*.json);;All files (*)"
    )
    if not path:
      return
    try:
      self.ec.load_scenario(path)
    except (OSError, ValueError, TypeError) as exc:
      QMessageBox.critical(self, "Could not load scenario", str(exc))

  def _on_export_screenshot(self) -> None:
    path, _ = QFileDialog.getSaveFileName(
      self, "Export Screenshot", "scenario.png", "PNG images (*.png);;All files (*)"
    )
    if not path:
      return
    try:
      self.env_view.export_boundary_screenshot(path)
    except (OSError, ValueError, RuntimeError) as exc:
      QMessageBox.critical(self, "Could not export screenshot", str(exc))

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
    self._refresh_ship_controls()
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
