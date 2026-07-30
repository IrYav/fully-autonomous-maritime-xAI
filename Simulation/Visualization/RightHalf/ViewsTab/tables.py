from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
  QAbstractItemView, QGroupBox, QHBoxLayout, QHeaderView, QTableWidgetItem, QTableWidget, 
  QVBoxLayout, QWidget, QCheckBox,
  )
from PySide6.QtGui import QColor

from Simulation.Visualization.VisHelpers.helpers import pretty_str

class TablesTab(QWidget):
  def __init__(self):
    super().__init__()
    tab_layout = QVBoxLayout(self)

    state = QGroupBox("State")
    state_layout = QVBoxLayout(state)
    reward = QGroupBox("Reward")
    reward_layout = QVBoxLayout(reward)

    tab_layout.addWidget(state,12) # 12 rows
    tab_layout.addWidget(reward,6) # 6 rows

    self._state = CounterfactualStateTable()
    self._reward = RewardVector()

    state_layout.addWidget(self._state)
    reward_layout.addWidget(self._reward)

  def set_state(self, state: dict):
    if self._state._keys is None or self._state._keys != list(state.keys()):
      self._state.set_variables(state)
    else:
      self._state.update_true_values(state)

  def set_reward(self, reward: dict):
    self._reward._set_reward(reward)

  @property
  def agent_observation(self) -> dict[str, float]:
    '''
      Get the observation of the agent, 
      overwritten with manipulated values where the box has been checked.
    '''
    return self._state.get_observation()

class VariableTable(QTableWidget):
  def __init__(self):
    super().__init__()

    self.setColumnCount(2)
    self.setHorizontalHeaderLabels(
      ["Variable", "Value"]
    )
    self.setEditTriggers(
      QAbstractItemView.EditTrigger.NoEditTriggers
    )

  def _set_values(self, values: dict):
    self.setRowCount(len(values))

    for row, (key, value) in enumerate(values.items()):
      
      if isinstance(value, (int, float)):
        text = f"{value:.6g}"
      else:
        text = str(value)

      self.setItem(
        row,
        0,
        QTableWidgetItem(pretty_str(key))
      )

      self.setItem(
        row,
        1,
        QTableWidgetItem(text)
      )

class RewardVector(VariableTable):
  def _set_reward(self, reward: dict):
    self._set_values(reward)


class CounterfactualStateTable(QTableWidget):
  override_changed = Signal(str, bool)

  def __init__(self):
    super().__init__()
    self._keys: list[str] | None = None # Placeholder until set.
    self.setColumnCount(4)
    self.setHorizontalHeaderLabels([
      "Name",
      "True Value",
      "Override",
      "Override active?",
    ])

    self.verticalHeader().hide()

    header = self.horizontalHeader()
    header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)        # Name
    header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
    header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
    header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
    
    self.cellClicked.connect(self._toggle_checkbox)

  def set_variables(self, true_values: dict):
    '''
      Call once to set number of variables.
    '''
    self.setRowCount(len(true_values))
    self._keys = list(true_values.keys())
    for row, (key, value) in enumerate(true_values.items()):

      self.setItem(row, 0, self._readonly_item(pretty_str(key)))
      self.setItem(row, 1, self._readonly_item(self._format(value)))
      self.setItem(row, 2, QTableWidgetItem(self._format(0.0)))

      checkbox = QCheckBox()
      checkbox.setChecked(False)

      checkbox.toggled.connect(
        lambda checked, r=row, k=key: (
          self._update_highlighting(r),
          self.override_changed.emit(k, checked),
        )
      )

      wrapper = QWidget()
      layout = QHBoxLayout(wrapper)
      layout.setContentsMargins(0, 0, 0, 0)
      layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
      layout.addWidget(checkbox)

      self.setCellWidget(row, 3, wrapper)

      self._update_highlighting(row)

  def update_true_values(self, true_values: dict):
    for row, key in enumerate(self._keys):
      self.item(row, 1).setText(self._format(true_values[key]))
      self._update_highlighting(row)

  def get_observation(self) -> dict:
    observation = {}

    for row, key in enumerate(self._keys):
      wrapper = self.cellWidget(row, 3)
      checkbox = wrapper.findChild(QCheckBox)

      column = 2 if checkbox.isChecked() else 1
      value = float(self.item(row, column).text())

      observation[key] = value

    return observation

  @staticmethod
  def _format(value):
    if isinstance(value, (int, float)):
      return f"{value:.6g}"
    return str(value)

  def _toggle_checkbox(self, row: int, column: int):
    if column != 3:
      return

    wrapper = self.cellWidget(row, 3)
    checkbox = wrapper.findChild(QCheckBox)
    checkbox.toggle()

  def _readonly_item(self, text: str) -> QTableWidgetItem:
    item = QTableWidgetItem(text)
    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
    return item

  def _update_highlighting(self, row: int):
    wrapper = self.cellWidget(row, 3)
    checkbox = wrapper.findChild(QCheckBox)

    if checkbox.isChecked():
        active_col = 2
        inactive_col = 1
    else:
        active_col = 1
        inactive_col = 2

    active = QColor(235, 250, 235)   # leichtes Grün
    inactive = QColor(Qt.GlobalColor.white)

    for col in (1, 2):
        item = self.item(row, col)
        if item is None:
            continue

        item.setBackground(active if col == active_col else inactive)