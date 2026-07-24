from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
  QAbstractButton, QButtonGroup, QGroupBox, QHBoxLayout, QPushButton, QRadioButton, QToolButton, QVBoxLayout,
  QWidget
  )
from matplotlib.patches import PathPatch

from Simulation.Visualization.Mouse.mouse_tool_actions import MouseToolActions, Tool
from Simulation.Visualization.RightHalf.AdminTab.editor_controller import EditorController
from Simulation.Visualization.RightHalf.AdminTab.cds_modes import CreateMode
from Simulation.Visualization.RightHalf.AdminTab.draft_model import Material

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
    sections.addStretch()
    sections.addWidget(self._bottom_buttons())
    
    #
    self._connect_signals()

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

    group_layout.addWidget(self.select)
    group_layout.addWidget(self.delete)
    #TODO Rotate and scale
    return section

  def _bottom_buttons(self) -> QWidget:
    section = QWidget()
    section_layout = QHBoxLayout(section)

    self.commit = QPushButton()
    self.commit.setText("Commit changes and recalculate everything.")
    self.commit.clicked.connect(self.ec.on_commit)

    section_layout.addWidget(self.commit)

    return section
  
  def _connect_signals(self):
    # render draft + preview
    self.ec.changed.connect(self._on_changed_signal)
    # render preview
    self.ec.preview_changed.connect(self._on_pv_changed_signal) 

  def _on_env_changed_signal(self):
    return
    #self.env_view._draw

  def _on_changed_signal(self):
    self.env_view.update_draft()
    self.finish.setEnabled(self.mta.mouse_tool.can_finish)

  def _on_pv_changed_signal(self, artists: list[PathPatch]):
    self.env_view.set_preview(artists)
    self.finish.setEnabled(self.mta.mouse_tool.can_finish)

  def _on_manual_clicked(self) -> None:
    """
    Enter manual island creation mode.

    Mouse clicks are interpreted as polygon vertices until the
    creation is finished or cancelled.
    """
    #TODO
    pass

  def _on_finish_clicked(self) -> None:
    """
    Finish the current manual polygon.

    If the constructed polygon is valid, it is added to the current
    scenario draft using the currently selected material.
    """
    mt = self.mta.mouse_tool
    mt.on_finish()
    pass

  def _on_material_changed(self, button: QAbstractButton) -> None:
    """
    Update the currently selected material.
    """
    material = button.property("material")
    self.ec.on_set_material(material)