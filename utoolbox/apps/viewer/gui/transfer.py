import logging

import numpy as np
from PySide2.QtCore import Qt
from PySide2.QtWidgets import QDialog, QLabel, QSlider, QVBoxLayout
from vispy.scene import SceneCanvas, PanZoomCamera

from utoolbox.apps.viewer.gui.histogram import Histogram

__all__ = ["TransferFunctionDialog"]

logger = logging.getLogger(__name__)


class TransferFunctionDialog(QDialog):
    def __init__(self, n_bins=256):
        self._n_bins = n_bins

        super().__init__()
        self.setWindowModality(Qt.NonModal)
        self.setWindowTitle("Transfer Function")

        layout = QVBoxLayout()

        histogram = self._create_histogram_widget()
        layout.addWidget(histogram)

        minimum = QSlider(Qt.Horizontal)
        minimum.setStyleSheet(
            """
            QSlider::groove:horizontal {
                border: 1px solid #999999;
                height: 12px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #B1B1B1, stop:1 #c4c4c4);
            }
            QSlider::handle:horizontal {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #b4b4b4, stop:1 #8f8f8f);
                border: 1px solid #5c5c5c;
                width: 12px;view.camera = scene.PanZoomCamera(aspect=1)

                margin: -2px 0;
            }
            """
        )
        minimum_label = QLabel("Minimum")
        minimum_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(minimum)
        layout.addWidget(minimum_label)

        maximum = QSlider(Qt.Horizontal)
        maximum_label = QLabel("Maximum")
        maximum_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(maximum)
        layout.addWidget(maximum_label)

        brightness = QSlider(Qt.Horizontal)
        brightness_label = QLabel("Brightness")
        brightness_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(brightness)
        layout.addWidget(brightness_label)

        contrast = QSlider(Qt.Horizontal)
        contrast_label = QLabel("Contrast")
        contrast_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(contrast)
        layout.addWidget(contrast_label)

        transparency = QSlider(Qt.Horizontal)
        transparency_label = QLabel("Transparency")
        transparency_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(transparency)
        layout.addWidget(transparency_label)

        self.setLayout(layout)

    ##

    @property
    def histogram(self):
        pass

    @property
    def n_bins(self):
        return self._n_bins

    ##

    def _create_histogram_widget(self, height=128):
        # create canvas
        canvas = SceneCanvas(size=(self.n_bins, height), bgcolor="white")
        viewbox = canvas.central_widget.add_view(border_width=0)
        viewbox.camera = PanZoomCamera(interactive=False)

        # dummy data
        edges = np.arange(self.n_bins + 1, dtype=np.int)
        data = np.zeros_like(edges, shape=(self.n_bins,))

        # create histogram
        histogram = Histogram((data, edges), orientation="h", color="gray")
        viewbox.add(histogram)
        viewbox.camera.set_range(margin=0)

        # save accessors
        self._viewbox, self._histogram = viewbox, histogram

        return canvas.native


if __name__ == "__main__":
    import sys
    from PySide2.QtWidgets import QApplication

    app = QApplication(sys.argv)

    # generate random data
    np.random.seed(42)

    n = 100000
    data = np.empty((n,))
    lasti = 0
    for i in range(1, 20):
        nexti = lasti + (n - lasti) // 2
        scale = np.abs(np.random.randn(1)) + 0.1
        data[lasti:nexti] = np.random.normal(
            size=(nexti - lasti,), loc=np.random.randn(1), scale=scale / i
        )
        lasti = nexti
    data = data[:lasti]

    histogram = np.histogram(data, bins=256)

    tf = TransferFunctionDialog()
    tf.show()

    print(".. pass tf.show()")

    sys.exit(app.exec_())
