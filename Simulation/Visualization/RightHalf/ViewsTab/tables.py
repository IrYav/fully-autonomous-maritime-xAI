from PySide6.QtWidgets import (
  QAbstractItemView, QGroupBox, QTableWidgetItem, QTableWidget, QVBoxLayout, QWidget
  )

from Simulation.Visualization.VisHelpers.helpers import pretty_str

class TablesTab(QWidget):
  def __init__(self):
    super().__init__()
    tab_layout = QVBoxLayout(self)

    state = QGroupBox("State")
    state_layout = QVBoxLayout(state)
    reward = QGroupBox("Reward")
    reward_layout = QVBoxLayout(reward)

    tab_layout.addWidget(state)
    tab_layout.addWidget(reward)

    self._state = StateVector()
    self._reward = RewardVector()

    state_layout.addWidget(self._state)
    reward_layout.addWidget(self._reward)

  def set_state(self, state: dict):
    self._state._set_state(state)

  def set_reward(self, reward: dict):
    self._reward._set_reward(reward)


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

class StateVector(VariableTable):
  def _set_state(self, state: dict):
    self._set_values(state)


class RewardVector(VariableTable):
  def _set_reward(self, reward: dict):
    self._set_values(reward)