import numpy as np
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from qgis.PyQt.QtCore import QCoreApplication
from qgis.PyQt.QtWidgets import QWidget, QVBoxLayout, QTreeWidget


class SurferPropertyGrid(QTreeWidget):
    """Property Grid in stile Golden Surfer."""
    def tr(self, text):
        return QCoreApplication.translate("KrigingStudio", text)
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setHeaderLabels([self.tr("Parameter"), self.tr("Value")])
        self.setColumnWidth(0, 180)


class MatplotlibVariogramCanvas(FigureCanvas):
    def tr(self, text):
        return QCoreApplication.translate("KrigingStudio", text)

    """Matplotlib variogram plot (X=Distance, Y=Variogram)."""
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(6, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        self.plot_empty()

    def plot_empty(self):
        self.ax.clear()
        self.ax.set_xlabel(self.tr("Distance"))
        self.ax.set_ylabel(self.tr("Variogram"))
        self.ax.grid(True, which='both', linestyle='-', linewidth=0.5, )
        self.ax.set_xlim(0, 15000)
        self.ax.set_ylim(0, 180000)
        self.fig.tight_layout()
        self.draw()

    def update_plot(self, x_pts, y_pts, x_model, y_model, auto_vscale=True, y_max_fixed=None, counts=None, experimental_color="#d62728", model_color="#1f77b4", point_size=36):
        """
        Update the variogram graph.

        :param x_pts: Experimental distances (lags)
        :param y_pts: Experimental semivariance values
        :param x_model: Distances for the model curve
        :param y_model: Semivariance values ​​for the model curve
        :param auto_vscale: If True scales the Y-axis automatically, otherwise uses y_max_fixed
        :param y_max_fixed: Custom maximum limit for Y-axis
        """
        self.ax.clear()

        # Experimental points are intentionally drawn above the model curve.
        self.ax.scatter(x_pts, y_pts, s=point_size, color=experimental_color, label=self.tr('Experimental'), zorder=3)

        # The model curve starts at the origin and is drawn below the points.
        if x_model is not None and y_model is not None:
            self.ax.plot(x_model, y_model, '-', color=model_color, label=self.tr('Model'), zorder=2)

        self.ax.set_xlabel(self.tr('Distance (Lag)'))
        self.ax.set_ylabel(self.tr('Variance'))
        self.ax.grid(True)

        # Vertical scale management
        if not auto_vscale and y_max_fixed is not None:
            self.ax.set_ylim(bottom=0, top=y_max_fixed)
        else:
            self.ax.set_ylim(bottom=0)

        self.ax.legend()
        self.fig.tight_layout()
        self.draw()

    def clear_plot(self):
        self.plot_empty()


class CrossValidationCanvas(FigureCanvas):
    def tr(self, text):
        return QCoreApplication.translate("KrigingStudio", text)

    """Cross Validation Graph (X=Estimated, Y=Measured)."""
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(5, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        self.clear_plot()

    def update_plot(self, estimated, measured, value_color="#7f7f7f", reference_color="#d62728", best_fit_color="#2ca02c", point_size=36):
        self.ax.clear()

        if len(estimated) == 0 or len(measured) == 0:
            self.clear_plot()
            return

        # Values are deliberately drawn first so both diagnostic lines remain visible.
        self.ax.scatter(estimated, measured, s=point_size, color=value_color, label=self.tr('Value'), zorder=1)

        # Reference line
        min_v = min(min(estimated), min(measured))
        max_v = max(max(estimated), max(measured))
        self.ax.plot([min_v, max_v], [min_v, max_v], '--', color=reference_color, label=self.tr('Reference line'), zorder=3)

        # Regression line (Best Fit)
        m, b = np.polyfit(estimated, measured, 1)
        order = np.argsort(np.asarray(estimated))
        x_sorted = np.asarray(estimated)[order]
        self.ax.plot(x_sorted, m * x_sorted + b, '-', color=best_fit_color, label=self.tr('Best Fit'), zorder=4)

        self.ax.set_xlabel(self.tr('Estimated'))
        self.ax.set_ylabel(self.tr('Measured'))
        self.ax.grid(True)

        # External legend at the top
        self.ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False)
        self.fig.tight_layout()
        self.draw()

    def clear_plot(self):
        """Completely clears the chart area and removes the legend."""
        self.ax.clear()
        self.ax.set_xlabel(self.tr('Estimated'))
        self.ax.set_ylabel(self.tr('Measured'))
        self.ax.grid(True)
        self.fig.tight_layout()
        self.draw()
