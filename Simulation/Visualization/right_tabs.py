from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
  QAbstractItemView, QLabel, QTableWidgetItem, QVBoxLayout,
  QTableWidget, QWidget
  )
from PySide6.QtGui import QImage, QPixmap

import numpy as np

class PixLabel(QLabel):
  '''
    QLabel but keeps aspect ratio and retains sharp edges.
  '''
  def __init__(self):
    super().__init__()
    self._pixmap = None

  def setPixmap(self, pixmap : QPixmap):
    self._pixmap = pixmap
    self.setMinimumSize(self._pixmap.size())
    self._update_pixmap()
  
  def resizeEvent(self, event):
    super().resizeEvent(event)
    self._update_pixmap()

  def _update_pixmap(self):
    if self._pixmap is None:
      return

    super().setPixmap(
      self._pixmap.scaled(
        self.size(),
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.FastTransformation,
      )
    )

class ImagePanel(QWidget):
  def __init__(self, title: str):
    super().__init__()

    self.title_label = QLabel(title)
    self.pix_label = PixLabel()

    policy = self.sizePolicy()
    policy.setHeightForWidth(True)

    self.setSizePolicy(policy)
    self.title_label.setAlignment(
      Qt.AlignmentFlag.AlignCenter
    )

    self.pix_label.setAlignment(
      Qt.AlignmentFlag.AlignCenter
    )

    layout = QVBoxLayout(self)

    layout.addWidget(self.pix_label, 1)
    layout.addWidget(self.title_label)

  def set_pixmap(self, pixmap):
    self.pix_label.setPixmap(pixmap)

  def set_title(self, title: str):
    self.title_label.setText(title)
  
class ImageView(QWidget):
  def __init__(self, image_names: list[str]):
    super().__init__()
    self.content_layout = QVBoxLayout(self)

    #Create an image slot for every image, with a more natural name.
    self.panels = {
      name: ImagePanel(_pretty_str(name))
      for name in image_names
    }

    self.content_layout.addStretch()

    for panel in self.panels.values():
      self.content_layout.addWidget(panel, 1)

    self.content_layout.addStretch()
  def set_views(self, images: np.ndarray):
    '''
    images:
      Tensor of shape
      (n_channels, rows, cols)
      n must be equal to `len(image_names)` picked during __init__
    Channel i is mapped to the i-th panel.
    '''

    if len(images) > len(self.panels):
      raise ValueError(
        f"Expected at most {len(self.panels)} channels, "
        f"got {len(images)}."
      )

    for panel, image in zip(
      self.panels.values(),
      images,
    ):
      image = image.astype(np.uint8)

      rows, cols = image.shape

      qimg = QImage(
        image.data,
        cols,
        rows,
        cols,
        QImage.Format.Format_Grayscale8,
      )

      pix = QPixmap.fromImage(qimg)

      panel.set_pixmap(pix)

class StateVector(QTableWidget):
  def __init__(self):
    super().__init__()

    self.setColumnCount(2)

    self.setHorizontalHeaderLabels(
      ["Variable", "Value"]
    )
    self.setEditTriggers(
      QAbstractItemView.EditTrigger.NoEditTriggers
    )

  def set_state(self, state: dict):
    self.setRowCount(len(state))
    for row, (key, value) in enumerate(state.items()):
      self.setItem(
        row,
        0,
        QTableWidgetItem(_pretty_str(key))
      )

      self.setItem(
        row,
        1,
        QTableWidgetItem(str(f"{value:.6g}"))
      )

def _pretty_str(text: object) -> str:
  '''
    Slightly prettier visually represented strings, 
    e.g., `image_view` becomes `Image View`
  '''
  return str(text).replace("_", " ").title()