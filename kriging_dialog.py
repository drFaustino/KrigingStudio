from qgis.PyQt.QtWidgets import (
    QFormLayout, QListWidget, QDialogButtonBox, QHBoxLayout, QTabWidget,
    QWidget, QHeaderView, QTreeWidget, QTreeWidgetItem, QLineEdit, QCheckBox,
    QFileDialog, QGroupBox, QProgressBar, QMessageBox, QTextEdit, QColorDialog,
)
from qgis.PyQt.QtCore import QVariant, QCoreApplication, QObject, pyqtSignal, QTimer
import csv
import copy
import os
import numpy as np
import matplotlib.pyplot as plt
import processing

from qgis.PyQt.QtWidgets import (
    QDialog, QVBoxLayout, QPushButton, QLabel, QComboBox, QTableWidget,
    QTableWidgetItem, QSpinBox, QDoubleSpinBox,
)
from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtGui import QColor
from qgis.gui import QgsMapLayerComboBox
from osgeo import gdal

from qgis.core import QgsTask, QgsApplication, QgsMessageLog

from qgis.core import (
    QgsProject, QgsVectorLayer, QgsRasterLayer, QgsPointXY,
    QgsFeature, QgsGeometry, QgsFields, QgsField,
    QgsVectorFileWriter, QgsWkbTypes, QgsRasterFileWriter, Qgis
)

from .kriging_engine import KrigingEngine
from .layer_source import (
    GEOMETRY_Z_KEY, ZSourceComboBox, extract_points, geometry_kind,
)
from .variogram_widget import (
    MatplotlibVariogramCanvas, CrossValidationCanvas, SurferPropertyGrid
)


class AutoFitDialog(QDialog):
    """Dialog for defining and selecting automatic variogram-fit parameters."""
    def tr(self, text):
        return QCoreApplication.translate("AutoFitDialog", text)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Auto Fit Parameters"))
        self.resize(520, 420)

        layout = QVBoxLayout(self)

        form = QFormLayout()

        self.fit_criterion = QComboBox()
        self.fit_criterion.addItem(self.tr("Default"), "default")
        self.fit_criterion.addItem(self.tr("Least Squares"), "least_squares")
        self.fit_criterion.addItem(self.tr("Least Absolute Value"), "least_absolute")
        # We set Least Squares to default as required
        self.fit_criterion.setCurrentIndex(1)
        form.addRow(self.tr("Fit Criterion:"), self.fit_criterion)

        self.target_precision = QDoubleSpinBox()
        self.target_precision.setDecimals(8)
        self.target_precision.setRange(0.0, 100.0)
        self.target_precision.setSingleStep(0.0001)
        self.target_precision.setValue(0.0001)
        form.addRow(self.tr("Target Precision (%):"), self.target_precision)

        self.maximum_iterations = QSpinBox()
        self.maximum_iterations.setRange(1, 1000000)
        self.maximum_iterations.setValue(50)
        form.addRow(self.tr("Maximum iterations:"), self.maximum_iterations)

        # We use QLineEdit to also allow scientific notation (e.g. 1E+38)
        self.maximum_distance = QLineEdit("1E+38")
        form.addRow(self.tr("Maximum distance:"), self.maximum_distance)

        layout.addLayout(form)

        layout.addWidget(QLabel(self.tr("<b>Linear</b>")))

        # Horizontal layout to place list and table side by side
        list_table_layout = QHBoxLayout()

        self.linear_list = QListWidget()
        self.linear_list.addItems([
            self.tr("Slope: 9.58887"),
            self.tr("Aniso Ratio: 1"),
            self.tr("Aniso Angle: 0"),
        ])
        self.linear_list.currentRowChanged.connect(self._select_parameter)
        list_table_layout.addWidget(self.linear_list, stretch=1)

        self.bounds = QTableWidget(3, 1)
        self.bounds.setHorizontalHeaderLabels([self.tr("Value")])
        self.bounds.setVerticalHeaderLabels([self.tr("Minimum"), self.tr("Initial"), self.tr("Maximum")])
        self.bounds.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        list_table_layout.addWidget(self.bounds, stretch=1)

        layout.addLayout(list_table_layout)

        self._values = {
            self.tr("Slope"): (0.0, 9.588866606224663, 1e38),
            self.tr("Aniso Ratio"): (0.5, 1.0, 2.0),
            self.tr("Aniso Angle"): (-360.0, 0.0, 360.0),
        }
        self.linear_list.setCurrentRow(0)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _select_parameter(self, row):
        if row < 0:
            return
        key = ("Slope", "Aniso Ratio", "Aniso Angle")[row]
        vals = self._values[key]
        for r, value in enumerate(vals):
            item = QTableWidgetItem(format(value, ".15g"))
            item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.bounds.setItem(r, 0, item)

    def get_parameters(self):
        try:
            max_dist_val = float(self.maximum_distance.text().replace(',', '.'))
        except ValueError:
            max_dist_val = 1e38

        return {
            "fit_criterion": self.fit_criterion.currentData() or "least_squares",
            "target_precision": self.target_precision.value(),
            "maximum_iterations": self.maximum_iterations.value(),
            "maximum_distance": max_dist_val,
            "linear": dict(self._values),
        }




class _KrigingGridTask(QgsTask):
    """CPU-heavy interpolation executed off the GUI thread."""
    def __init__(self, description, engine, coords, values, grid_x, grid_y,
                 return_variance, max_neighbors, callback):
        super().__init__(description, QgsTask.CanCancel)
        self.engine = engine
        self.coords = np.asarray(coords, dtype=float)
        self.values = np.asarray(values, dtype=float)
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.return_variance = return_variance
        self.max_neighbors = max_neighbors
        self.callback = callback
        self.predictions = self.variances = None
        self.error_message = ""
        self.progress_signal = _TaskSignals()

    def run(self):
        try:
            def progress(v):
                if self.isCanceled():
                    raise RuntimeError("Operation cancelled.")
                self.setProgress(float(v))
                self.progress_signal.progress.emit(int(v), "Kriging: %d%%" % int(v))
            self.predictions, self.variances = self.engine.fit_and_predict(
                self.coords, self.values, self.grid_x, self.grid_y,
                return_variance=self.return_variance,
                max_neighbors=self.max_neighbors,
                progress_callback=progress
            )
            return not self.isCanceled()
        except Exception as exc:
            self.error_message = str(exc)
            return False

    def finished(self, result):
        # QgsTask.finished runs on the main thread; only UI/QGIS project
        # operations are performed here.
        self.callback(self, bool(result))


class _TaskSignals(QObject):
    progress = pyqtSignal(int, str)


class _BackgroundOperationTask(QgsTask):
    """Run CPU/I/O-heavy plugin operations away from QGIS' GUI thread."""
    def __init__(self, description, operation, finished_callback):
        super().__init__(description, QgsTask.CanCancel)
        self.operation = operation
        self.finished_callback = finished_callback
        self.signals = _TaskSignals()
        self.result_data = None
        self.error_message = ""

    def run(self):
        try:
            def report(value, message="Working…"):
                if self.isCanceled():
                    raise RuntimeError("Operation cancelled.")
                value = max(0, min(100, int(value)))
                self.setProgress(value)
                self.signals.progress.emit(value, str(message))
            self.result_data = self.operation(report, self.isCanceled)
            return not self.isCanceled()
        except Exception as exc:
            self.error_message = str(exc)
            return False

    def finished(self, result):
        self.finished_callback(self, bool(result))


class KrigingDialog(QDialog):
    def tr(self, text):
        translated = QCoreApplication.translate("KrigingDialog", text)
        if translated != text:
            return translated
        # Runtime fallback keeps recently added messages translated even when the
        # installed .qm file was built from an older translation catalog.
        from qgis.PyQt.QtCore import QLocale
        if QLocale.system().name().lower().startswith("it"):
            return {
                "Ready": "Pronto",
                "Calculating experimental variogram…": "Calcolo del variogramma sperimentale…",
                "Calculating kriging grid…": "Calcolo della griglia di kriging…",
                "Calculation failed or cancelled": "Calcolo non riuscito o annullato",
                "Cross-validation running…": "Convalida incrociata in corso…",
                "Cross-validation: %1%": "Convalida incrociata: %1%",
                "Experimental variogram": "Variogramma sperimentale",
                "Fitting variogram model…": "Adattamento del modello di variogramma…",
                "Generating contour lines…": "Generazione delle curve di livello…",
                "Kriging interpolation": "Interpolazione kriging",
                "Operation failed or cancelled": "Operazione non riuscita o annullata",
                "Optimizing variogram parameters…": "Ottimizzazione dei parametri del variogramma…",
                "Output error": "Errore di output",
                "Output failed": "Generazione degli output non riuscita",
                "Output files completed": "File di output completati",
                "Save contour lines": "Salva curve di livello",
                "Save interpolated raster": "Salva raster interpolato",
                "Save kriging variance": "Salva varianza del kriging",
                "Updating variogram plot…": "Aggiornamento del grafico del variogramma…",
                "Variogram AutoFit": "Adattamento automatico del variogramma",
                "Variogram failed": "Calcolo del variogramma non riuscito",
                "Writing interpolated raster…": "Scrittura del raster interpolato…",
                "Writing kriging outputs": "Scrittura degli output del kriging",
                "Writing raster and contour outputs…": "Scrittura del raster e delle curve di livello…",
                "Writing variance raster…": "Scrittura del raster della varianza…",
                "A calculation is already running.": "È già in corso un calcolo.",
                "Another operation is already running.": "È già in corso un'altra operazione.",
                "AutoFit failed": "AutoFit non riuscito",
                "Wait for the current operation to finish before clearing results.": "Attendi il completamento dell'operazione corrente prima di cancellare i risultati.",
                "Contour vector output requires Save Raster File.": "L'output vettoriale delle curve di livello richiede di selezionare il salvataggio del raster.",
                "Contour vector output requires 'Save Raster File' to be selected.": "Per generare le curve di livello vettoriali è necessario selezionare 'Save Raster File'.",
                "GeoPackage (*.gpkg);;ESRI Shapefile (*.shp);;All files (*.*)": "GeoPackage (*.gpkg);;ESRI Shapefile (*.shp);;Tutti i file (*.*)",
                "GeoTIFF raster (*.tif *.tiff);;All files (*.*)": "Raster GeoTIFF (*.tif *.tiff);;Tutti i file (*.*)",
                "Kriging": "Kriging",
                "Valid samples: %1 of %2 features": "Campioni validi: %1 su %2 feature",
                "Z from geometry (2D surface kriging)": "Z dalla geometria (kriging di superficie 2D)",
                "Numeric attribute field, or the Z coordinate of the geometry (single-part PointZ, PointZM and Point25D layers).": "Campo attributo numerico oppure coordinata Z della geometria (layer a punto singolo PointZ, PointZM e Point25D).",
                "Please select a valid point or line layer and a Z source (numeric field or geometry Z).": "Seleziona un layer puntuale o lineare valido e una sorgente Z (campo numerico o Z della geometria).",
                "Next >": "Avanti >",
                "< Back": "< Indietro",
                "Variogram AutoFit": "Adattamento automatico del variogramma",
            }.get(text, text)
        return translated

    """UI main with 3 Tabs and advanced Surfer options for variogram."""

    DESCRIPTIONS = {
        "GENERAL": "General variogram parameters and trend options.",
        "Trend removal": "Specifies polynomial trend removal (None, Linear, or Quadratic) prior to variogram computation.",
        "Auto vertical scale": "Automatically rescales the Y-axis based on variogram data range.",
        "AutoFit Settings": "Opens the UI to define and configure automatic variogram-fit parameters.",
        "AutoFit": "Calculate fit parameters and refresh variogram.",
        "EXPERIMENTAL": "Experimental variogram calculation parameters.",
        "Estimator type": "Type of variogram estimator used.",
        "Point size": "Sets the marker size for the experimental variogram data points on the plot.",
        "Auto max lag distance": "Automatically computes maximum lag distance based on dataset extent.",
        "Max lag distance": "Maximum distance up to which pair lag distances are calculated.",
        "Number of lags": "Total number of lag intervals.",
        "Auto lag width": "Automatically calculates lag width as Max Lag / Number of Lags.",
        "Lag width": "Width of each lag bin.",
        "LAG DIRECTION": "Directional variogram parameters.",
        "Direction (°)": "Azimuth direction for directional variogram computation.",
        "Tolerance (°)": "Angular tolerance around the specified direction.",
        "Step amount (°)": "Angular step increment for stepping through directions."
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Kriging Studio"))
        self.resize(1000, 850)
        self.engine = KrigingEngine()
        self.last_y_max = None
        self.auto_fit_parameters = None

        self.unified_btn_style = """
            QPushButton {
                background-color: #2b5b84;
                color: #ffffff;
                font-weight: bold;
                padding: 6px;
                border: 1px solid #1f415f;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #3a71a4;
            }
        """
        self.autofit_btn_style = """
            QPushButton {
                background-color: #5c3c88;
                color: #ffffff;
                font-weight: bold;
                padding: 6px;
                border: 1px solid #432b64;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #714b9f;
            }
        """

        self.setup_ui()

    def _tree_item(self, parent, text):
        item = QTreeWidgetItem(parent, [self.tr(text), ""])
        item.setData(0, Qt.ItemDataRole.UserRole, text)
        return item

    def setup_ui(self):
        main_layout = QVBoxLayout(self)

        top_panel = QHBoxLayout()
        top_panel.addWidget(QLabel(self.tr("Input layer:")))
        self.layer_cb = QgsMapLayerComboBox()
        self._apply_layer_filters()
        top_panel.addWidget(self.layer_cb, stretch=1)

        top_panel.addWidget(QLabel(self.tr("Z field:")))
        self.field_cb = ZSourceComboBox()
        self.field_cb.setToolTip(self.tr(
            "Numeric attribute field, or the Z coordinate of the geometry "
            "(single-part PointZ, PointZM and Point25D layers)."))
        top_panel.addWidget(self.field_cb, stretch=1)
        self.layer_cb.layerChanged.connect(self.field_cb.setLayer)
        self.layer_cb.layerChanged.connect(self.on_input_layer_changed)
        self.field_cb.setLayer(self.layer_cb.currentLayer())

        self.btn_refresh = QPushButton(self.tr("Refresh"))
        self.btn_refresh.setToolTip(self.tr("Refresh project layers and fields"))
        self.btn_refresh.setStyleSheet(self.unified_btn_style)
        self.btn_refresh.clicked.connect(self.refresh_layers_and_fields)
        top_panel.addWidget(self.btn_refresh)

        main_layout.addLayout(top_panel)
        self.lbl_source_info = QLabel("")
        self.lbl_source_info.setStyleSheet("color: #555555;")
        main_layout.addWidget(self.lbl_source_info)

        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabBar::tab:nth-child(1) { background-color: #dbeafe; color: #1e3a8a; font-weight: bold; }
            QTabBar::tab:nth-child(2) { background-color: #dcfce7; color: #166534; font-weight: bold; }
            QTabBar::tab:nth-child(3) { background-color: #fef3c7; color: #92400e; font-weight: bold; }
            QTabBar::tab {
                font-weight: bold;
                padding: 7px 14px;
                border: 1px solid #b8b8b8;
                border-bottom: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:nth-child(1) { background-color: #dcecff; color: #17365d; }
            QTabBar::tab:nth-child(2) { background-color: #dff2df; color: #1f5c2a; }
            QTabBar::tab:nth-child(3) { background-color: #f6e1e6; color: #6b2434; }
            QTabBar::tab:selected { border-bottom: 2px solid #555555; }
            /* Combo-box popups inside the three tabs: keep selected text readable. */
            QComboBox QAbstractItemView {
                selection-background-color: #2b5b84;
                selection-color: #ffffff;
            }

            QComboBox QAbstractItemView::item:selected {
                background-color: #2b5b84;
                color: #ffffff;
            }

            QComboBox QAbstractItemView::item:hover {
                background-color: #dbeafe;
                color: #17365d;
            }
            QTreeWidget::item:selected,
            QListWidget::item:selected {
                background-color: #2b5b84;
                color: #ffffff;
            }
        """)

        self.tab_variogram = QWidget()
        self.setup_variogram_tab()
        self.tab_variogram.setStyleSheet("background-color: #f7f9fa;")
        self.tabs.addTab(self.tab_variogram, self.tr("1. Variogram"))

        self.tab_options = QWidget()
        self.setup_options_tab()
        self.tab_options.setStyleSheet("background-color: #f5f8f5;")
        self.tabs.addTab(self.tab_options, self.tr("2. Kriging and Validation"))

        self.tab_export = QWidget()
        self.setup_export_tab()
        self.tab_export.setStyleSheet("background-color: #f9f6f7;")
        self.tabs.addTab(self.tab_export, self.tr("3. Results"))

        main_layout.addWidget(self.tabs)

        # Persistent progress indicator, independent of the currently selected tab.
        self.progress_label_bottom = QLabel(self.tr("Ready"))
        self.progress_bar_bottom = QProgressBar()
        self.progress_bar_bottom.setRange(0, 100)
        self.progress_bar_bottom.setValue(0)
        self.progress_bar_bottom.setTextVisible(True)
        main_layout.addWidget(self.progress_label_bottom)
        main_layout.addWidget(self.progress_bar_bottom)

        btn_layout = QHBoxLayout()
        self.btn_back = QPushButton(self.tr("< Back"))
        self.btn_next = QPushButton(self.tr("Next >"))
        self.btn_cancel = QPushButton(self.tr("Cancel"))
        self.btn_back.setStyleSheet("""
            QPushButton { background-color: #d9e8f7; color: #17365d; font-weight: bold; padding: 6px 14px; border: 1px solid #9bb8d4; border-radius: 4px; }
            QPushButton:hover { background-color: #c5dcf0; }
        """)
        self.btn_next.setStyleSheet("""
            QPushButton { background-color: #dff0df; color: #205b2a; font-weight: bold; padding: 6px 14px; border: 1px solid #a8c9a8; border-radius: 4px; }
            QPushButton:hover { background-color: #cce6cc; }
        """)

        self.btn_cancel.clicked.connect(self.reject)
        self.btn_next.clicked.connect(self.next_tab)
        self.btn_back.clicked.connect(self.prev_tab)

        btn_layout.addWidget(self.btn_back)
        btn_layout.addWidget(self.btn_next)
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_cancel)

        main_layout.addLayout(btn_layout)

    def next_tab(self):
        curr = self.tabs.currentIndex()
        if curr < self.tabs.count() - 1:
            self.tabs.setCurrentIndex(curr + 1)

    def prev_tab(self):
        curr = self.tabs.currentIndex()
        if curr > 0:
            self.tabs.setCurrentIndex(curr - 1)

    def _apply_layer_filters(self):
        """Show only vector layers that can supply samples: point and line layers."""
        try:
            flt = Qgis.LayerFilter.PointLayer | Qgis.LayerFilter.LineLayer
        except AttributeError:  # older API
            from qgis.core import QgsMapLayerProxyModel
            flt = QgsMapLayerProxyModel.PointLayer | QgsMapLayerProxyModel.LineLayer
        self.layer_cb.setFilters(flt)
        self.layer_cb.setAllowEmptyLayer(False)

    def on_input_layer_changed(self, layer):
        """Warn and reject polygon layers; only point and line inputs are supported."""
        if not layer or not isinstance(layer, QgsVectorLayer):
            return
        if geometry_kind(layer) == "polygon":
            QMessageBox.warning(
                self,
                self.tr("Unsupported layer type"),
                self.tr("Polygon layers are not supported. Select a point layer or a line layer with a numeric elevation field.")
            )
            self.field_cb.setLayer(None)

    def refresh_layers_and_fields(self, preferred_layer=None):
        """Re-read project layers and the numeric fields of the selected layer.

        ``preferred_layer`` (e.g. the active QGIS layer) is selected when it is a
        point or line vector layer that the selector lists.
        """
        self.layer_cb.setExceptedLayerList([])
        self._apply_layer_filters()
        if isinstance(preferred_layer, QgsVectorLayer):
            self.layer_cb.setLayer(preferred_layer)  # ignored if filtered out
        self.field_cb.setLayer(self.layer_cb.currentLayer())
        self.recalculate_lag_parameters()

    def open_autofit_settings_dialog(self):
        dialog = AutoFitDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.auto_fit_parameters = dialog.get_parameters()

    def _style_tab_combo(self, combo):
        combo.setStyleSheet("""
            QComboBox {
                background-color: #ffffff;
                color: #17365d;
                border: 1px solid #9aa7b2;
                padding: 3px 6px;
            }

            QComboBox:hover {
                border: 1px solid #2b5b84;
            }

            QComboBox QAbstractItemView {
                background-color: #ffffff;
                color: #17365d;
                border: 1px solid #9aa7b2;
                outline: none;
                selection-background-color: #2b5b84;
                selection-color: #ffffff;
            }

            QComboBox QAbstractItemView::item {
                background-color: #ffffff;
                color: #17365d;
                padding: 5px 8px;
            }

            QComboBox QAbstractItemView::item:hover {
                background-color: #dbeafe;
                color: #17365d;
            }

            QComboBox QAbstractItemView::item:selected {
                background-color: #2b5b84;
                color: #ffffff;
            }
        """)

        # Forza anche i colori della palette del popup.
        view = combo.view()
        palette = view.palette()
        palette.setColor(palette.ColorRole.Base, QColor("#ffffff"))
        palette.setColor(palette.ColorRole.Text, QColor("#17365d"))
        palette.setColor(palette.ColorRole.Highlight, QColor("#2b5b84"))
        palette.setColor(palette.ColorRole.HighlightedText, QColor("#ffffff"))
        view.setPalette(palette)

    def setup_variogram_tab(self):
        layout = QHBoxLayout(self.tab_variogram)

        left_layout = QVBoxLayout()
        self.var_plot = MatplotlibVariogramCanvas()
        left_layout.addWidget(self.var_plot)

        btn_var_layout = QHBoxLayout()

        self.btn_calc_variogram = QPushButton(self.tr("Calculate Variogram"))
        self.btn_calc_variogram.setStyleSheet(self.unified_btn_style)
        self.btn_calc_variogram.clicked.connect(self.on_calculate_variogram)

        self.btn_step_cw = QPushButton(self.tr("Step CW (►)"))
        self.btn_step_cw.setToolTip(self.tr("Rotate the direction clockwise"))
        self.btn_step_cw.clicked.connect(lambda: self.step_direction(-1))

        self.btn_step_ccw = QPushButton(self.tr("Step CCW (◄)"))
        self.btn_step_ccw.setToolTip(self.tr("Rotate the direction counterclockwise"))
        self.btn_step_ccw.clicked.connect(lambda: self.step_direction(1))

        self.btn_clear_variogram = QPushButton(self.tr("Clear"))
        self.btn_clear_variogram.setStyleSheet("padding: 6px;")
        self.btn_clear_variogram.clicked.connect(self.on_clear_variogram)

        btn_var_layout.addWidget(self.btn_calc_variogram)
        btn_var_layout.addWidget(self.btn_step_ccw)
        btn_var_layout.addWidget(self.btn_step_cw)
        btn_var_layout.addWidget(self.btn_clear_variogram)

        left_layout.addLayout(btn_var_layout)

        right_layout = QVBoxLayout()
        self.prop_grid = SurferPropertyGrid()
        self.prop_grid.header().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)

        # ==================== 1. GENERAL ====================
        general = self._tree_item(self.prop_grid, "GENERAL")
        item_trend = self._tree_item(general, "Trend removal")
        self.cb_trend = QComboBox()
        self.cb_trend.addItem(self.tr("None"), "None")
        self.cb_trend.addItem(self.tr("Linear"), "Linear")
        self.cb_trend.addItem(self.tr("Quadratic"), "Quadratic")
        self._style_tab_combo(self.cb_trend)

        item_auto_vscale = self._tree_item(general, "Auto vertical scale")
        self.chk_auto_vscale = QCheckBox()
        self.chk_auto_vscale.setChecked(True)

        item_autofit_settings = self._tree_item(general, "AutoFit Settings")
        self.btn_autofit_settings = QPushButton(self.tr("AutoFit Settings..."))
        self.btn_autofit_settings.setStyleSheet(self.autofit_btn_style)
        self.btn_autofit_settings.clicked.connect(self.open_autofit_settings_dialog)

        item_autofit = self._tree_item(general, "AutoFit")
        self.btn_autofit = QPushButton(self.tr("AutoFit..."))
        self.btn_autofit.setStyleSheet(self.unified_btn_style)
        self.btn_autofit.clicked.connect(self.open_autofit_dialog)

        # ==================== 2. EXPERIMENTAL ====================
        exp = self._tree_item(self.prop_grid, "EXPERIMENTAL")
        estimator_item = self._tree_item(exp, "Estimator type")
        estimator_item.setText(1, self.tr("Variogram"))

        item_model = self._tree_item(exp, "Model")
        self.cb_model = QComboBox()
        self.cb_model.addItem(self.tr("Linear"), "linear")
        self.cb_model.addItem(self.tr("Spherical"), "spherical")
        self.cb_model.addItem(self.tr("Exponential"), "exponential")
        self.cb_model.addItem(self.tr("Gaussian"), "gaussian")
        self.cb_model.addItem(self.tr("Power"), "power")
        self.cb_model.addItem(self.tr("Wave (Hole Effect)"), "hole_effect")
        self._style_tab_combo(self.cb_model)
        self.cb_model.currentIndexChanged.connect(self._set_model_from_ui)

        item_nugget = self._tree_item(exp, "Nugget effect")
        self.spin_nugget = QDoubleSpinBox()
        self.spin_nugget.setRange(0.0, 1e30)
        self.spin_nugget.setDecimals(8)
        self.spin_nugget.setValue(0.0)
        self.spin_nugget.valueChanged.connect(lambda v: setattr(self.engine, "nugget", float(v)))

        item_auto_max_dist = self._tree_item(exp, "Auto max lag distance")
        self.chk_auto_max_dist = QCheckBox()
        self.chk_auto_max_dist.setChecked(True)

        item_max_dist = self._tree_item(exp, "Max lag distance")
        self.txt_max_dist = QLineEdit("15000.0")

        item_n_lags = self._tree_item(exp, "Number of lags")
        self.spin_n_lags = QSpinBox()
        self.spin_n_lags.setRange(1, 500)
        self.spin_n_lags.setValue(15)

        item_auto_lag = self._tree_item(exp, "Auto lag width")
        self.chk_auto_lag = QCheckBox()
        self.chk_auto_lag.setChecked(True)

        item_lag_size = self._tree_item(exp, "Lag width")
        self.txt_lag_size = QLineEdit("1000.0")
        self.txt_lag_size.setReadOnly(True)

        # ==================== 3. LAG DIRECTION ====================
        dir_group = self._tree_item(self.prop_grid, "LAG DIRECTION")

        item_dir = self._tree_item(dir_group, "Direction (°)")
        self.spin_dir = QDoubleSpinBox()
        self.spin_dir.setRange(-360.0, 360.0)
        self.spin_dir.setValue(0.0)

        item_tol = self._tree_item(dir_group, "Tolerance (°)")
        self.spin_tol = QDoubleSpinBox()
        self.spin_tol.setRange(0.0, 90.0)
        self.spin_tol.setValue(90.0)

        item_step_amt = self._tree_item(dir_group, "Step amount (°)")
        self.spin_step_amt = QDoubleSpinBox()
        self.spin_step_amt.setRange(0.1, 180.0)
        self.spin_step_amt.setValue(15.0)

        item_aniso_ratio = self._tree_item(dir_group, "Anisotropy ratio")
        self.spin_aniso_ratio = QDoubleSpinBox()
        self.spin_aniso_ratio.setRange(0.01, 100.0)
        self.spin_aniso_ratio.setValue(1.0)
        self.spin_aniso_ratio.valueChanged.connect(lambda v: setattr(self.engine, "anis_ratio", float(v)))

        item_aniso_angle = self._tree_item(dir_group, "Anisotropy angle (°)")
        self.spin_aniso_angle = QDoubleSpinBox()
        self.spin_aniso_angle.setRange(-360.0, 360.0)
        self.spin_aniso_angle.setValue(0.0)
        self.spin_aniso_angle.valueChanged.connect(lambda v: setattr(self.engine, "anis_angle", float(v)))

        # ==================== 4. GRAPH STYLING ====================
        style_group = self._tree_item(self.prop_grid, "GRAPH STYLING")
        self.graph_style_controls = {}

        style_items = [
            ("Experimental", "#d62728", False),
            ("Model", "#1f77b4", False),
            ("Value", "#7f7f7f", True),
            ("Reference line", "#d62728", False),
            ("Best Fit", "#2ca02c", False),
        ]
        for label, color, is_point in style_items:
            item = self._tree_item(style_group, label)
            button = QPushButton()
            button.setProperty("graph_color", color)
            self._set_color_button(button, color)
            button.clicked.connect(lambda checked=False, b=button, n=label: self.choose_graph_color(n, b))
            self.prop_grid.setItemWidget(item, 1, button)
            self.graph_style_controls[label] = button

            if label == "Experimental":
                item_point_size_exp = self._tree_item(style_group, "Point size")
                self.spin_point_size_exp = QSpinBox()
                self.spin_point_size_exp.setRange(1, 100)
                self.spin_point_size_exp.setValue(26)
                self.spin_point_size_exp.valueChanged.connect(self.on_point_size_changed)
                self.prop_grid.setItemWidget(item_point_size_exp, 1, self.spin_point_size_exp)

        item_point_size = self._tree_item(style_group, "Value point size")
        self.spin_value_point_size = QSpinBox()
        self.spin_value_point_size.setRange(1, 100)
        self.spin_value_point_size.setValue(26)
        self.spin_value_point_size.valueChanged.connect(self.on_point_size_changed)
        self.prop_grid.setItemWidget(item_point_size, 1, self.spin_value_point_size)

        self.prop_grid.expandAll()

        # Assigning Widgets to the Property Grid
        self.prop_grid.setItemWidget(item_trend, 1, self.cb_trend)
        self.prop_grid.setItemWidget(item_auto_vscale, 1, self.chk_auto_vscale)
        self.prop_grid.setItemWidget(item_autofit_settings, 1, self.btn_autofit_settings)
        self.prop_grid.setItemWidget(item_autofit, 1, self.btn_autofit)
        self.prop_grid.setItemWidget(item_model, 1, self.cb_model)
        self.prop_grid.setItemWidget(item_nugget, 1, self.spin_nugget)
        self.prop_grid.setItemWidget(item_auto_max_dist, 1, self.chk_auto_max_dist)
        self.prop_grid.setItemWidget(item_max_dist, 1, self.txt_max_dist)
        self.prop_grid.setItemWidget(item_n_lags, 1, self.spin_n_lags)
        self.prop_grid.setItemWidget(item_auto_lag, 1, self.chk_auto_lag)
        self.prop_grid.setItemWidget(item_lag_size, 1, self.txt_lag_size)
        self.prop_grid.setItemWidget(item_dir, 1, self.spin_dir)
        self.prop_grid.setItemWidget(item_tol, 1, self.spin_tol)
        self.prop_grid.setItemWidget(item_step_amt, 1, self.spin_step_amt)
        self.prop_grid.setItemWidget(item_aniso_ratio, 1, self.spin_aniso_ratio)
        self.prop_grid.setItemWidget(item_aniso_angle, 1, self.spin_aniso_angle)

        right_layout.addWidget(self.prop_grid, stretch=2)

        # ==================== OBJECT DESCRIPTION PANEL ====================
        self.grp_desc = QGroupBox()
        self.grp_desc.setMaximumHeight(85)
        desc_lay = QVBoxLayout(self.grp_desc)
        desc_lay.setContentsMargins(4, 2, 4, 2)

        self.lbl_desc_title = QLabel(self.tr("<b>Trend removal</b>"))
        self.txt_desc_body = QTextEdit()
        self.txt_desc_body.setReadOnly(True)
        self.txt_desc_body.setMaximumHeight(45)
        self.txt_desc_body.setStyleSheet("background-color: transparent; border: none;")
        self.txt_desc_body.setText(self.tr(self.DESCRIPTIONS["Trend removal"]))

        desc_lay.addWidget(self.lbl_desc_title)
        desc_lay.addWidget(self.txt_desc_body)
        right_layout.addWidget(self.grp_desc, stretch=1)

        self.prop_grid.currentItemChanged.connect(self.on_property_item_changed)

        self.chk_auto_max_dist.toggled.connect(self.on_auto_max_dist_toggled)
        self.chk_auto_lag.toggled.connect(self.on_auto_lag_toggled)
        self.spin_n_lags.valueChanged.connect(self.recalculate_lag_parameters)
        self.txt_max_dist.textChanged.connect(self.recalculate_lag_parameters)
        self.layer_cb.layerChanged.connect(self.recalculate_lag_parameters)
        self.field_cb.sourceChanged.connect(self.recalculate_lag_parameters)

        layout.addLayout(left_layout, stretch=2)
        layout.addLayout(right_layout, stretch=1)

        self.recalculate_lag_parameters()

    def on_point_size_changed(self, size):
        # Connected to size controls; plot refresh occurs when the associated
        # variogram/value plot is redrawn. Keep this slot intentionally harmless.
        return None

    def _set_color_button(self, button, color_name):
        color = QColor(color_name)
        button.setProperty("graph_color", color.name())
        button.setStyleSheet(
            f"QPushButton {{ background-color: {color.name()}; border: 1px solid #555; min-width: 70px; min-height: 20px; }}"
        )
        button.setToolTip(self.tr("Choose graph color"))

    def choose_graph_color(self, name, button):
        current = QColor(button.property("graph_color") or "#000000")
        color = QColorDialog.getColor(current, self, self.tr("Choose color for %1").replace("%1", self.tr(name)))
        if color.isValid():
            self._set_color_button(button, color.name())

    def graph_style(self, name, default):
        button = self.graph_style_controls.get(name) if hasattr(self, "graph_style_controls") else None
        return str(button.property("graph_color")) if button and button.property("graph_color") else default

    def on_property_item_changed(self, current, previous):
        if not current:
            return

        # Testo visualizzato, eventualmente già tradotto
        item_text = current.text(0)

        self.lbl_desc_title.setText(f"<b>{item_text}</b>")

        # Recupera il testo originale inglese salvato nel UserRole
        original_text = current.data(0, Qt.ItemDataRole.UserRole)

        # Cerca la descrizione usando la chiave originale
        desc = self.DESCRIPTIONS.get(
            original_text,
            "No description available for this parameter."
        )

        # Traduce la descrizione
        self.txt_desc_body.setText(self.tr(desc))

    def on_auto_max_dist_toggled(self, checked):
        self.txt_max_dist.setReadOnly(checked)
        if checked:
            self.recalculate_lag_parameters()

    def on_auto_lag_toggled(self, checked):
        self.txt_lag_size.setReadOnly(checked)
        if checked:
            self.recalculate_lag_parameters()

    def calculate_auto_max_distance(self, coords):
        if coords is None or len(coords) < 2:
            return 15000.0

        min_x, min_y = np.min(coords, axis=0)
        max_x, max_y = np.max(coords, axis=0)

        diagonal = np.hypot(max_x - min_x, max_y - min_y)
        return diagonal * 0.35

    def recalculate_lag_parameters(self):
        coords, _ = self.get_layer_points()

        if self.chk_auto_max_dist.isChecked():
            if coords is not None and len(coords) >= 2:
                max_dist = self.calculate_auto_max_distance(coords)
                self.txt_max_dist.setText(f"{max_dist:.4f}")
            else:
                max_dist = 15000.0
        else:
            try:
                max_dist = float(self.txt_max_dist.text().replace(',', '.'))
            except ValueError:
                max_dist = 15000.0

        if self.chk_auto_lag.isChecked():
            n_lags = self.spin_n_lags.value()
            if n_lags > 0 and max_dist > 0:
                lag_width = max_dist / n_lags
                self.txt_lag_size.setText(f"{lag_width:.4f}")

    def calculate_surfer_lag_intervals(self, max_lag_distance, number_of_lags, lag_width):
        intervals = []
        for lag in range(number_of_lags):
            center = (max_lag_distance / number_of_lags) * (lag + 0.5)
            from_dist = max(center - (lag_width / 2.0), 0.0)
            to_dist = min(center + (lag_width / 2.0), max_lag_distance)
            intervals.append((center, from_dist, to_dist))
        return intervals

    def step_direction(self, direction_factor):
        step = self.spin_step_amt.value()
        current_dir = self.spin_dir.value()
        new_dir = (current_dir + (direction_factor * step)) % 360.0
        self.spin_dir.setValue(new_dir)
        self.on_calculate_variogram()

    def open_autofit_dialog(self):
        if getattr(self, "_operation_task", None) is not None:
            return
        coords, vals = self.get_layer_points()
        if coords is None or len(coords) < 3:
            QMessageBox.warning(self, self.tr("AutoFit"), self.tr("At least three valid points are required to fit a variogram."))
            return
        self._configure_engine()
        engine = copy.deepcopy(self.engine)
        residuals = self.remove_trend(coords, vals, self.cb_trend.currentData() or "None")
        params = (self.spin_n_lags.value(), float(self.txt_max_dist.text().replace(",", ".")),
                  float(self.txt_lag_size.text().replace(",", ".")), self.spin_dir.value(), self.spin_tol.value())
        self._set_operation_busy(True, self.tr("Fitting variogram model…"))
        def operation(report, cancelled):
            report(5, self.tr("Calculating experimental variogram…"))
            lags, gamma, counts = engine.calculate_experimental_variogram(coords, residuals,
                n_lags=params[0], max_dist=params[1], lag_width=params[2], direction=params[3], tolerance=params[4])
            if len(lags) < 2:
                raise ValueError("Not enough experimental variogram pairs are available.")
            report(60, self.tr("Optimizing variogram parameters…"))
            if cancelled(): raise RuntimeError("Operation cancelled.")
            if not engine.autofit(lags, gamma, counts):
                raise ValueError("The variogram fit did not converge.")
            return (engine.nugget, engine.sill, engine.range_val, engine.slope, engine.scale, engine.exponent)
        self._start_background_operation(self.tr("Variogram AutoFit"), operation, self._finish_autofit)

    def _finish_autofit(self, task, success):
        self._operation_task = None
        self._set_operation_busy(False, self.tr("Completed") if success else self.tr("AutoFit failed"))
        if not success:
            if task.error_message and task.error_message != "Operation cancelled.":
                QMessageBox.critical(self, self.tr("AutoFit"), task.error_message)
            return
        self.engine.nugget, self.engine.sill, self.engine.range_val, self.engine.slope, self.engine.scale, self.engine.exponent = task.result_data
        self.spin_nugget.setValue(float(self.engine.nugget))
        self.on_calculate_variogram()
        QMessageBox.information(self, self.tr("AutoFit"), self.tr("Variogram parameters fitted successfully."))

    def get_layer_points(self):
        """Return (coords, values) from the selected layer and Z source.

        The Z source is a numeric attribute field or, for single-part point
        layers with Z values, the geometry Z coordinate (``GEOMETRY_Z_KEY``).
        Features with empty, multi-part (geometry Z), missing or non-finite
        XY/Z values are ignored.
        """
        layer = self.layer_cb.currentLayer()
        source = self.field_cb.currentField()

        if not layer or not source or not isinstance(layer, QgsVectorLayer):
            self._show_source_info(None)
            return None, None
        try:
            coords, vals, stats = extract_points(layer, source)
        except ValueError as exc:
            self._show_source_info(None, str(exc))
            return None, None
        self._show_source_info(stats)
        return coords, vals

    def _show_source_info(self, stats, message=""):
        if not hasattr(self, "lbl_source_info"):
            return
        if message:
            self.lbl_source_info.setText(message)
        elif stats is None:
            self.lbl_source_info.setText("")
        else:
            text = self.tr("Valid samples: %1 of %2 features").replace("%1", str(stats["used"])).replace("%2", str(stats["total"]))
            if self.field_cb.isGeometryZ():
                text += " — " + self.tr("Z from geometry (2D surface kriging)")
            self.lbl_source_info.setText(text)

    def remove_trend(self, coords, vals, trend_type):
        if trend_type == "None" or len(coords) < 6:
            return vals

        x = coords[:, 0]
        y = coords[:, 1]

        if trend_type == "Linear":
            A = np.column_stack([x, y, np.ones_like(x)])
        elif trend_type == "Quadratic":
            A = np.column_stack([x**2, y**2, x*y, x, y, np.ones_like(x)])
        else:
            return vals

        try:
            coeffs, _, _, _ = np.linalg.lstsq(A, vals, rcond=None)
            trend = A @ coeffs
            return vals - trend
        except np.linalg.LinAlgError:
            return vals

    def on_calculate_variogram(self):
        if getattr(self, "_operation_task", None) is not None:
            return
        coords, vals = self.get_layer_points()
        self.recalculate_lag_parameters()
        try:
            max_dist = float(self.txt_max_dist.text().replace(',', '.'))
            lag_width = float(self.txt_lag_size.text().replace(',', '.'))
        except ValueError:
            max_dist, lag_width = 15000.0, 1000.0
        if coords is None or len(coords) < 2:
            self.on_clear_variogram()
            QMessageBox.warning(self, self.tr("Variogram"), self.tr("Select a valid point or line layer with numeric values before calculating the variogram."))
            return
        self._configure_engine()
        engine = copy.deepcopy(self.engine)
        residuals = self.remove_trend(coords, vals, self.cb_trend.currentData() or "None")
        args = (self.spin_n_lags.value(), max_dist, lag_width, self.spin_dir.value(), self.spin_tol.value())
        self._set_operation_busy(True, self.tr("Calculating experimental variogram…"))
        def operation(report, cancelled):
            report(5, self.tr("Calculating experimental variogram…"))
            result = engine.calculate_experimental_variogram(coords, residuals, n_lags=args[0],
                max_dist=args[1], lag_width=args[2], direction=args[3], tolerance=args[4])
            report(90, self.tr("Updating variogram plot…"))
            return result
        self._start_background_operation(self.tr("Experimental variogram"), operation, self._finish_variogram)

    def _finish_variogram(self, task, success):
        self._operation_task = None
        self._set_operation_busy(False, self.tr("Completed") if success else self.tr("Variogram failed"))
        if not success:
            if task.error_message and task.error_message != "Operation cancelled.":
                QMessageBox.critical(self, self.tr("Variogram"), task.error_message)
            return
        result = task.result_data
        lags, gamma, counts = result
        if len(lags) == 0:
            self.on_clear_variogram()
            return
        self._configure_engine()
        lags_model = np.insert(lags, 0, 0.0)
        model_y = self.engine.variogram_function(lags_model)
        auto_vscale = self.chk_auto_vscale.isChecked()
        if auto_vscale or self.last_y_max is None:
            self.last_y_max = max(float(np.max(gamma)), float(np.max(model_y))) * 1.1
        kwargs = dict(auto_vscale=auto_vscale, y_max_fixed=self.last_y_max, counts=counts,
            experimental_color=self.graph_style("Experimental", "#d62728"),
            model_color=self.graph_style("Model", "#1f77b4"), point_size=self.spin_point_size_exp.value())
        if hasattr(self.var_plot, 'update_plot'):
            try:
                self.var_plot.update_plot(lags, gamma, lags_model, model_y, **kwargs)
            except TypeError:
                self.var_plot.update_plot(lags, gamma, lags_model, model_y)
        self._progress_callback(100)
        self.progress_label_bottom.setText(self.tr("Completed"))

    def on_clear_variogram(self):
        self._reset_progress_bars()
        self.last_y_max = None
        if hasattr(self.var_plot, 'clear_plot'):
            self.var_plot.clear_plot()

    def setup_options_tab(self):
        root_layout = QVBoxLayout(self.tab_options)
        layout = QHBoxLayout()
        root_layout.addLayout(layout, stretch=1)

        left_lay = QVBoxLayout()

        grp_kriging = QGroupBox(self.tr("Kriging Parameters"))
        grp_kriging.setMaximumHeight(110)
        lay_k = QVBoxLayout(grp_kriging)
        lay_k.setContentsMargins(8, 6, 8, 6)

        h_drift = QHBoxLayout()
        h_drift.addWidget(QLabel(self.tr("Kriging drift type:")))
        self.cb_drift = QComboBox()
        self.cb_drift.addItem(self.tr("Ordinary"), "ordinary")
        self.cb_drift.addItem(self.tr("Universal (Linear)"), "universal_linear")
        self.cb_drift.addItem(self.tr("Universal (Quadratic)"), "universal_quadratic")
        self._style_tab_combo(self.cb_drift)
        h_drift.addWidget(self.cb_drift)
        lay_k.addLayout(h_drift)

        h_sec = QHBoxLayout()
        h_sec.addWidget(QLabel(self.tr("Search sectors:")))
        self.spin_sectors = QSpinBox()
        self.spin_sectors.setValue(4)
        h_sec.addWidget(self.spin_sectors)

        h_sec.addWidget(QLabel(self.tr("Maximum points per sector:")))
        self.spin_max_pts = QSpinBox()
        self.spin_max_pts.setValue(16)
        h_sec.addWidget(self.spin_max_pts)
        lay_k.addLayout(h_sec)

        left_lay.addWidget(grp_kriging)

        self.cross_plot = CrossValidationCanvas()
        left_lay.addWidget(self.cross_plot, stretch=1)

        btn_cross_lay = QHBoxLayout()
        self.btn_calc_cross = QPushButton(self.tr("Run Cross Validation"))
        self.btn_calc_cross.setStyleSheet(self.unified_btn_style)
        self.btn_calc_cross.clicked.connect(self.on_calculate_cross_validation)

        self.btn_clear_cross = QPushButton(self.tr("Clear"))
        self.btn_clear_cross.setStyleSheet("padding: 6px;")
        self.btn_clear_cross.clicked.connect(self.on_clear_cross_validation)

        btn_cross_lay.addWidget(self.btn_calc_cross)
        btn_cross_lay.addWidget(self.btn_clear_cross)

        left_lay.addLayout(btn_cross_lay)

        right_lay = QVBoxLayout()

        lbl_stats = QLabel(self.tr("Cross Validation Summary Statistics"))
        lbl_stats.setStyleSheet("font-weight: bold;")
        right_lay.addWidget(lbl_stats)

        self.tbl_cross_stats = QTableWidget(4, 1)
        self.tbl_cross_stats.setHorizontalHeaderLabels([self.tr("Value")])
        self.tbl_cross_stats.setVerticalHeaderLabels([
            self.tr("Mean Error"),
            self.tr("Root Mean Square"),
            self.tr("Mean Absolute Error"),
            self.tr("Correlation")
        ])

        self.tbl_cross_stats.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )

        # Riduce l'altezza delle righe
        self.tbl_cross_stats.verticalHeader().setDefaultSectionSize(27)

        # Altezza complessiva della tabella
        self.tbl_cross_stats.setFixedHeight(140)

        right_lay.addWidget(self.tbl_cross_stats)

        lbl_val = QLabel(self.tr("Cross Validation Values"))
        lbl_val.setStyleSheet("font-weight: bold;")
        right_lay.addWidget(lbl_val)

        self.tbl_cross_val = QTableWidget(8, 4)
        self.tbl_cross_val.setHorizontalHeaderLabels([self.tr("Row"), self.tr("Measured"), self.tr("Estimated"), self.tr("Error")])
        self.tbl_cross_val.verticalHeader().hide()
        self.tbl_cross_val.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        right_lay.addWidget(self.tbl_cross_val)

        lbl_dataset_stats = QLabel(self.tr("Dataset Statistics"))
        lbl_dataset_stats.setStyleSheet("font-weight: bold;")
        right_lay.addWidget(lbl_dataset_stats)

        self.tbl_dataset_stats = QTableWidget(6, 3)
        self.tbl_dataset_stats.setHorizontalHeaderLabels([self.tr("X"), self.tr("Y"), self.tr("Z")])
        self.tbl_dataset_stats.setVerticalHeaderLabels([self.tr("Minimum"), self.tr("Maximum"), self.tr("Mean"), self.tr("Variance"), self.tr("Std. Dev."), self.tr("Skewness")])
        self.tbl_dataset_stats.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        right_lay.addWidget(self.tbl_dataset_stats)

        self.btn_save_report = QPushButton(self.tr("Save the report (TXT / CSV)"))
        self.btn_save_report.setStyleSheet(self.unified_btn_style)
        self.btn_save_report.clicked.connect(self.export_validation_report)
        right_lay.addWidget(self.btn_save_report)

        layout.addLayout(left_lay, stretch=1)
        layout.addLayout(right_lay, stretch=1)

    def export_validation_report(self):
        filepath, selected_filter = QFileDialog.getSaveFileName(
            self, self.tr("Save Report Validation"), "", self.tr("CSV Files (*.csv);;Text Files (*.txt)")
        )
        if not filepath:
            return

        try:
            is_csv = filepath.endswith(".csv") or "CSV" in selected_filter
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                if is_csv:
                    writer = csv.writer(f)

                    # 1. CROSS VALIDATION SUMMARY STATISTICS
                    writer.writerow([self.tr("--- CROSS VALIDATION SUMMARY STATISTICS ---")])
                    for row in range(self.tbl_cross_stats.rowCount()):
                        label = self.tbl_cross_stats.verticalHeaderItem(row).text() if self.tbl_cross_stats.verticalHeaderItem(row) else ""
                        item = self.tbl_cross_stats.item(row, 0)
                        val = item.text() if item else ""
                        writer.writerow([label, val])

                    writer.writerow([])

                    # 2. CROSS VALIDATION VALUES
                    writer.writerow([self.tr("--- CROSS VALIDATION VALUES ---")])
                    writer.writerow([self.tr("Row"), self.tr("Measured"), self.tr("Estimated"), self.tr("Error")])
                    for row in range(self.tbl_cross_val.rowCount()):
                        r_vals = [self.tbl_cross_val.item(row, col).text() if self.tbl_cross_val.item(row, col) else "" for col in range(4)]
                        writer.writerow(r_vals)

                    writer.writerow([])

                    # 3. DATASET STATISTICS
                    writer.writerow([self.tr("--- DATASET STATISTICS ---")])
                    writer.writerow([self.tr("Statistic"), self.tr("X"), self.tr("Y"), self.tr("Z")])
                    for row in range(self.tbl_dataset_stats.rowCount()):
                        label = self.tbl_dataset_stats.verticalHeaderItem(row).text() if self.tbl_dataset_stats.verticalHeaderItem(row) else ""
                        r_vals = [self.tbl_dataset_stats.item(row, col).text() if self.tbl_dataset_stats.item(row, col) else "" for col in range(3)]
                        writer.writerow([label] + r_vals)

                else:
                    # 1. CROSS VALIDATION SUMMARY STATISTICS
                    f.write("=== CROSS VALIDATION SUMMARY STATISTICS ===\n")
                    for row in range(self.tbl_cross_stats.rowCount()):
                        label = self.tbl_cross_stats.verticalHeaderItem(row).text() if self.tbl_cross_stats.verticalHeaderItem(row) else ""
                        item = self.tbl_cross_stats.item(row, 0)
                        val = item.text() if item else ""
                        f.write(f"{label}: {val}\n")

                    f.write("\n=== CROSS VALIDATION VALUES ===\n")
                    f.write("\t".join([self.tr("Row"), self.tr("Measured"), self.tr("Estimated"), self.tr("Error")]) + "\n")
                    for row in range(self.tbl_cross_val.rowCount()):
                        r_vals = [self.tbl_cross_val.item(row, col).text() if self.tbl_cross_val.item(row, col) else "" for col in range(4)]
                        f.write("\t".join(r_vals) + "\n")

                    f.write("\n=== DATASET STATISTICS ===\n")
                    f.write("\t".join([self.tr("Statistic"), self.tr("X"), self.tr("Y"), self.tr("Z")]) + "\n")
                    for row in range(self.tbl_dataset_stats.rowCount()):
                        label = self.tbl_dataset_stats.verticalHeaderItem(row).text() if self.tbl_dataset_stats.verticalHeaderItem(row) else ""
                        r_vals = [self.tbl_dataset_stats.item(row, col).text() if self.tbl_dataset_stats.item(row, col) else "" for col in range(3)]
                        f.write(f"{label}\t" + "\t".join(r_vals) + "\n")

            QMessageBox.information(self, self.tr("Export"), self.tr("Report successfully exported with all three tables."))
        except Exception as exc:
            QMessageBox.critical(self, self.tr("Export Error"), self.tr("Unable to save report:") + f" {exc}")

    def _reset_progress_bars(self):
        """Reset the single persistent progress bar."""
        if hasattr(self, "progress_bar_bottom"):
            self.progress_bar_bottom.setValue(0)

    def _schedule_progress_reset(self):
        """Reset the persistent progress bar after completion, once no task is active."""
        def reset_when_idle():
            if getattr(self, "_operation_task", None) is not None:
                QTimer.singleShot(500, reset_when_idle)
                return
            if hasattr(self, "progress_bar_bottom") and self.progress_bar_bottom.value() >= 100:
                self.progress_bar_bottom.setValue(0)
                if hasattr(self, "progress_label_bottom"):
                    self.progress_label_bottom.setText(self.tr("Ready"))
        QTimer.singleShot(1200, reset_when_idle)

    def _progress_callback(self, value):
        value = max(0, min(100, int(value)))
        if hasattr(self, "progress_bar_bottom"):
            self.progress_bar_bottom.setValue(value)
        if value >= 100:
            self._schedule_progress_reset()

    def _fill_dataset_statistics(self, coords, values):
        """Populate dataset descriptive statistics for X, Y and Z."""
        coords = np.asarray(coords, dtype=float)
        values = np.asarray(values, dtype=float)
        columns = (coords[:, 0], coords[:, 1], values)
        for col, data in enumerate(columns):
            data = data[np.isfinite(data)]
            if data.size == 0:
                stats = ["—"] * 6
            else:
                mean = float(np.mean(data))
                variance = float(np.var(data, ddof=1)) if data.size > 1 else 0.0
                stddev = float(np.std(data, ddof=1)) if data.size > 1 else 0.0
                centered = data - mean
                skewness = (float(np.mean(centered ** 3)) / (float(np.std(data)) ** 3)
                            if data.size > 2 and float(np.std(data)) > 0 else 0.0)
                stats = [
                    f"{np.min(data):.6g}", f"{np.max(data):.6g}",
                    f"{mean:.6g}", f"{variance:.6g}",
                    f"{stddev:.6g}", f"{skewness:.6g}",
                ]
            for row, value in enumerate(stats):
                item = QTableWidgetItem(value)
                item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.tbl_dataset_stats.setItem(row, col, item)

    def _fill_cross_validation_statistics(self, measured, estimated, errors):
        """Populate summary metrics from finite cross-validation observations."""
        measured = np.asarray(measured, dtype=float)
        estimated = np.asarray(estimated, dtype=float)
        errors = np.asarray(errors, dtype=float)
        valid = np.isfinite(measured) & np.isfinite(estimated) & np.isfinite(errors)
        measured, estimated, errors = measured[valid], estimated[valid], errors[valid]
        if errors.size == 0:
            values = ["—", "—", "—", "—"]
        else:
            correlation = (float(np.corrcoef(measured, estimated)[0, 1])
                           if errors.size > 1 and np.std(measured) > 0 and np.std(estimated) > 0
                           else float("nan"))
            values = [
                f"{np.mean(errors):.6g}",
                f"{np.sqrt(np.mean(errors ** 2)):.6g}",
                f"{np.mean(np.abs(errors)):.6g}",
                f"{correlation:.6g}" if np.isfinite(correlation) else "—",
            ]
        for row, value in enumerate(values):
            item = QTableWidgetItem(value)
            item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.tbl_cross_stats.setItem(row, 0, item)

    def on_calculate_cross_validation(self):
        if getattr(self, "_operation_task", None) is not None:
            return
        coords, values = self.get_layer_points()
        if coords is None or len(coords) < 4:
            QMessageBox.warning(self, self.tr("Cross-validation"), self.tr("At least four valid points are required for cross-validation."))
            return
        self._fill_dataset_statistics(coords, values)
        self._configure_engine()
        engine = copy.deepcopy(self.engine)
        max_neighbors = self.spin_max_pts.value()
        self._set_operation_busy(True, self.tr("Cross-validation running…"))
        def operation(report, cancelled):
            def progress(v): report(v, self.tr("Cross-validation: %1%").replace("%1", str(v)))
            return engine.cross_validate(coords, values, max_neighbors=max_neighbors, progress_callback=progress)
        self._start_background_operation(self.tr("Cross-validation"), operation, self._finish_cross_validation)

    def _finish_cross_validation(self, task, success):
        self._operation_task = None
        self._set_operation_busy(False, self.tr("Completed") if success else self.tr("Operation failed or cancelled"))
        if not success:
            if task.error_message and task.error_message != "Operation cancelled.":
                QMessageBox.critical(self, self.tr("Cross-validation"), task.error_message)
            return
        measured, estimated, errors = task.result_data
        self._fill_cross_validation_statistics(measured, estimated, errors)
        if hasattr(self.cross_plot, "update_plot"):
            self.cross_plot.update_plot(estimated, measured,
                value_color=self.graph_style("Value", "#7f7f7f"),
                reference_color=self.graph_style("Reference line", "#d62728"),
                best_fit_color=self.graph_style("Best Fit", "#2ca02c"),
                point_size=self.spin_value_point_size.value())
        self.tbl_cross_val.setRowCount(len(measured))
        for i, (m, e, err) in enumerate(zip(measured, estimated, errors)):
            for col, value in enumerate((str(i + 1), f"{m:.6g}", f"{e:.6g}", f"{err:.6g}")):
                item = QTableWidgetItem(value)
                item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.tbl_cross_val.setItem(i, col, item)

    def on_clear_cross_validation(self):
        """Clear cross-validation results without disturbing a running task."""
        if getattr(self, "_operation_task", None) is not None:
            QMessageBox.information(
                self,
                self.tr("Cross-validation"),
                self.tr("Wait for the current operation to finish before clearing results."),
            )
            return

        if hasattr(self, "cross_plot") and hasattr(self.cross_plot, "clear_plot"):
            self.cross_plot.clear_plot()
        if hasattr(self, "tbl_cross_val"):
            self.tbl_cross_val.setRowCount(0)
        for name in ("tbl_cross_stats", "tbl_dataset_stats"):
            table = getattr(self, name, None)
            if table is not None:
                table.clearContents()
        self._reset_progress_bars()
        if hasattr(self, "progress_label_bottom"):
            self.progress_label_bottom.setText(self.tr("Ready"))

    def _start_background_operation(self, description, operation, callback):
        task = _BackgroundOperationTask(description, operation, callback)
        task.signals.progress.connect(self._on_background_progress)
        self._operation_task = task
        QgsApplication.taskManager().addTask(task)

    def _on_background_progress(self, value, message):
        self.progress_label_bottom.setText(message)
        self.progress_bar_bottom.setValue(value)
        self._progress_callback(value)

    def _set_operation_busy(self, busy, message):
        self.progress_label_bottom.setText(message)
        if not busy:
            if message == self.tr("Completed"):
                self._progress_callback(100)
            else:
                self._reset_progress_bars()
        for name in ("btn_calc_variogram", "btn_autofit", "btn_calc_cross", "btn_run_interpolation"):
            widget = getattr(self, name, None)
            if widget is not None:
                widget.setEnabled(not busy)

    def _configure_engine(self):
        drift = self.cb_drift.currentData() or "ordinary"
        self.engine.drift_type = {
            "ordinary": "None",
            "universal_linear": "Linear",
            "universal_quadratic": "Quadratic",
        }.get(drift, "None")
        if hasattr(self, "cb_model"):
            self.engine.model_type = self.cb_model.currentData() or "linear"
        self.engine.anis_ratio = max(0.01, float(self.spin_aniso_ratio.value())) if hasattr(self, "spin_aniso_ratio") else 1.0
        self.engine.anis_angle = float(self.spin_aniso_angle.value()) if hasattr(self, "spin_aniso_angle") else 0.0

    def _set_model_from_ui(self):
        self._configure_engine()

    def run_kriging_execution(self):
        """Validate inputs quickly, then run interpolation in a cancellable QgsTask."""
        if getattr(self, "_operation_task", None) is not None:
            QMessageBox.information(self, self.tr("Kriging"), self.tr("Another operation is already running."))
            return
        if getattr(self, "_kriging_task", None) is not None:
            QMessageBox.information(self, self.tr("Kriging"), self.tr("A calculation is already running."))
            return
        self._reset_progress_bars()
        if not any((self.chk_save_raster.isChecked(),
                    self.chk_save_variance.isChecked(),
                    self.chk_save_vector.isChecked())):
            QMessageBox.warning(self, self.tr("Warning"), self.tr("Select at least one output option: raster, variance, or contour line."))
            return
        if self.chk_save_vector.isChecked() and not self.chk_save_raster.isChecked():
            QMessageBox.warning(self, self.tr("Warning"), self.tr("Contour vector output requires 'Save Raster File' to be selected."))
            return

        layer = self.layer_cb.currentLayer()
        field_name = self.field_cb.currentField()
        if not layer or not field_name:
            QMessageBox.critical(self, self.tr("Error"), self.tr("Please select a valid point or line layer and a Z source (numeric field or geometry Z)."))
            return
        coords, vals = self.get_layer_points()
        if coords is None or len(coords) < 3:
            QMessageBox.critical(self, self.tr("Error"), self.tr("At least three valid input points are required for kriging."))
            return

        extent = layer.extent()
        xmin, xmax, ymin, ymax = extent.xMinimum(), extent.xMaximum(), extent.yMinimum(), extent.yMaximum()
        if not (xmax > xmin and ymax > ymin):
            QMessageBox.critical(self, self.tr("Error"), self.tr("The input layer has an invalid extent."))
            return
        raster_path = self.txt_raster_path.text().strip()
        if self.chk_save_raster.isChecked() and not raster_path:
            QMessageBox.warning(self, self.tr("Warning"), self.tr("Please specify a valid raster output path."))
            return
        save_variance = bool(self.chk_save_variance.isChecked())
        variance_path = self.txt_variance_path.text().strip() if hasattr(self, "txt_variance_path") else ""
        if save_variance and not variance_path:
            base, ext = os.path.splitext(raster_path)
            if raster_path:
                variance_path = base + "_variance" + (ext or ".tif")
                self.txt_variance_path.setText(variance_path)
            else:
                QMessageBox.warning(self, self.tr("Warning"), self.tr("Please specify a valid Kriging variance output path."))
                return

        self._configure_engine()
        self.engine.nugget = float(self.spin_nugget.value()) if hasattr(self, "spin_nugget") else 0.0
        engine = copy.deepcopy(self.engine)
        cols = rows = 200
        gx = xmin + (np.arange(cols, dtype=float) + 0.5) * ((xmax - xmin) / cols)
        gy = ymax - (np.arange(rows, dtype=float) + 0.5) * ((ymax - ymin) / rows)
        grid_x, grid_y = np.meshgrid(gx, gy)

        self._kriging_context = {
            "raster_path": raster_path, "variance_path": variance_path,
            "save_raster": self.chk_save_raster.isChecked(),
            "save_variance": save_variance, "save_vector": self.chk_save_vector.isChecked(),
            "add_to_project": self.chk_add_to_project.isChecked(),
            "extent": extent, "crs": layer.crs(),
            "vector_outputs": [
                (self.txt_vector_path_main.text().strip(), self.spin_step_main.value()),
                (self.txt_vector_path_sec.text().strip(), self.spin_step_sec.value())
            ]
        }
        self.progress_label_bottom.setText(self.tr("Calculating kriging grid…"))
        self.progress_bar_bottom.setValue(0)
        self.btn_run_interpolation.setEnabled(False)
        task = _KrigingGridTask(
            self.tr("Kriging interpolation"), engine, coords, vals, grid_x, grid_y,
            save_variance, self.spin_max_pts.value(), self._on_kriging_task_finished
        )
        task.progress_signal.progress.connect(self._on_background_progress)
        self._kriging_task = task
        QgsApplication.taskManager().addTask(task)

    def _on_kriging_task_finished(self, task, success):
        self.btn_run_interpolation.setEnabled(True)
        self._kriging_task = None
        ctx = self._kriging_context
        if not success:
            self.progress_label_bottom.setText(self.tr("Calculation failed or cancelled"))
            self.progress_bar_bottom.setValue(0)
            if task.error_message and task.error_message != "Operation cancelled.":
                QMessageBox.critical(self, self.tr("Kriging Error"), task.error_message)
            return
        # Raster writing and contour generation are expensive too: do them in a
        # second background task. Only adding layers/messages happens on the GUI thread.
        grid_z = np.asarray(task.predictions, dtype=np.float32)
        grid_var = np.asarray(task.variances, dtype=np.float32)
        extent = ctx["extent"]
        extent_values = (extent.xMinimum(), extent.xMaximum(), extent.yMinimum(), extent.yMaximum())
        crs_wkt = ctx["crs"].toWkt() if ctx["crs"] and ctx["crs"].isValid() else ""
        output_ctx = dict(ctx)
        output_ctx["extent_values"] = extent_values
        output_ctx["crs_wkt"] = crs_wkt
        self._set_operation_busy(True, self.tr("Writing raster and contour outputs…"))
        def operation(report, cancelled):
            outputs = []
            if output_ctx["save_raster"]:
                report(5, self.tr("Writing interpolated raster…"))
                self._write_geotiff_values(
                    output_ctx["raster_path"], grid_z, extent_values, crs_wkt,
                    progress_callback=lambda fraction: report(
                        5 + int(20 * fraction), self.tr("Writing interpolated raster…")
                    ), cancelled=cancelled,
                )
                outputs.append(("raster", output_ctx["raster_path"]))
            if output_ctx["save_variance"]:
                report(30, self.tr("Writing variance raster…"))
                self._write_geotiff_values(
                    output_ctx["variance_path"], grid_var, extent_values, crs_wkt,
                    progress_callback=lambda fraction: report(
                        30 + int(15 * fraction), self.tr("Writing variance raster…")
                    ), cancelled=cancelled,
                )
                outputs.append(("raster", output_ctx["variance_path"]))
            if output_ctx["save_vector"]:
                for i, (path, interval) in enumerate(output_ctx["vector_outputs"]):
                    if cancelled():
                        raise RuntimeError("Operation cancelled.")
                    if not path:
                        continue
                    report(50 + i * 20, self.tr("Generating contour lines…"))
                    processing.run("gdal:contour", {"INPUT": output_ctx["raster_path"], "BAND": 1,
                        "INTERVAL": float(interval), "FIELD_NAME": "ELEV", "OUTPUT": path})
                    outputs.append(("vector", path))
            report(100, self.tr("Output files completed"))
            return outputs
        self._start_background_operation(self.tr("Writing kriging outputs"), operation, self._finish_output_task)
        self._output_ctx = output_ctx

    def _finish_output_task(self, task, success):
        ctx = getattr(self, "_output_ctx", {})
        self._operation_task = None
        self._set_operation_busy(False, self.tr("Completed") if success else self.tr("Output failed"))
        if not success:
            if task.error_message and task.error_message != "Operation cancelled.":
                QMessageBox.critical(self, self.tr("Output error"), task.error_message)
            return
        if ctx.get("add_to_project"):
            for kind, path in task.result_data or []:
                if kind == "raster":
                    layer = QgsRasterLayer(path, os.path.basename(path))
                else:
                    layer = QgsVectorLayer(path, os.path.basename(path), "ogr")
                if layer.isValid():
                    QgsProject.instance().addMapLayer(layer)
        self.progress_bar_bottom.setValue(100)
        self.progress_label_bottom.setText(self.tr("Completed"))
        QMessageBox.information(self, self.tr("Kriging"), self.tr("Kriging completed successfully."))

    @staticmethod
    def _write_geotiff_values(path, array, extent_values, crs_wkt,
                              progress_callback=None, cancelled=None):
        """Write GeoTIFF by row chunks to report progress and support cancellation."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        xmin, xmax, ymin, ymax = extent_values
        rows, cols = array.shape
        driver = gdal.GetDriverByName("GTiff")
        if driver is None:
            raise RuntimeError("The GDAL GeoTIFF driver is not available.")
        ds = driver.Create(path, cols, rows, 1, gdal.GDT_Float32,
                           options=["COMPRESS=DEFLATE", "TILED=YES", "BIGTIFF=IF_SAFER"])
        if ds is None:
            raise RuntimeError(f"Unable to create '{path}'.")
        try:
            ds.SetGeoTransform((xmin, (xmax-xmin)/cols, 0.0, ymax, 0.0, -(ymax-ymin)/rows))
            if crs_wkt:
                ds.SetProjection(crs_wkt)
            band = ds.GetRasterBand(1)
            nodata = -3.4028235e38
            band.SetNoDataValue(nodata)
            data = np.asarray(array, dtype=np.float32)
            data = np.where(np.isfinite(data), data, nodata)
            chunk_rows = max(1, min(128, rows // 20 or 1))
            for yoff in range(0, rows, chunk_rows):
                if cancelled and cancelled():
                    raise RuntimeError("Operation cancelled.")
                yend = min(rows, yoff + chunk_rows)
                band.WriteArray(data[yoff:yend, :], 0, yoff)
                if progress_callback:
                    progress_callback(yend / max(rows, 1))
            band.FlushCache()
            ds.FlushCache()
        finally:
            ds = None

    def _write_geotiff(self, path, array, extent, crs):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        rows, cols = array.shape
        driver = gdal.GetDriverByName("GTiff")
        if driver is None:
            raise RuntimeError("The GDAL GeoTIFF driver is not available.")
        ds = driver.Create(path, cols, rows, 1, gdal.GDT_Float32,
                           options=["COMPRESS=DEFLATE", "TILED=YES"])
        if ds is None:
            raise RuntimeError(f"Unable to create '{path}'.")
        xres = (extent.xMaximum() - extent.xMinimum()) / cols
        yres = (extent.yMaximum() - extent.yMinimum()) / rows
        ds.SetGeoTransform((extent.xMinimum(), xres, 0.0,
                            extent.yMaximum(), 0.0, -yres))
        if crs and crs.isValid():
            ds.SetProjection(crs.toWkt())
        band = ds.GetRasterBand(1)
        band.WriteArray(np.asarray(array, dtype=np.float32))
        band.SetNoDataValue(-3.4028235e38)
        band.FlushCache()
        ds.FlushCache()
        ds = None


    def browse_raster(self):
        """Choose the destination GeoTIFF path without starting any processing."""
        current = self.txt_raster_path.text().strip()
        filename, _ = QFileDialog.getSaveFileName(
            self,
            self.tr("Save interpolated raster"),
            current or os.path.join(os.path.expanduser("~"), "kriging_result.tif"),
            self.tr("GeoTIFF raster (*.tif *.tiff);;All files (*.*)")
        )
        if filename:
            if not filename.lower().endswith((".tif", ".tiff")):
                filename += ".tif"
            self.txt_raster_path.setText(filename)

    def browse_variance(self):
        """Choose the destination GeoTIFF path for kriging variance."""
        current = self.txt_variance_path.text().strip()
        filename, _ = QFileDialog.getSaveFileName(
            self,
            self.tr("Save kriging variance"),
            current or os.path.join(os.path.expanduser("~"), "kriging_variance.tif"),
            self.tr("GeoTIFF raster (*.tif *.tiff);;All files (*.*)")
        )
        if filename:
            if not filename.lower().endswith((".tif", ".tiff")):
                filename += ".tif"
            self.txt_variance_path.setText(filename)

    def browse_vector(self, target_edit):
        """Choose the destination file for a contour vector layer."""
        current = target_edit.text().strip()
        filename, _ = QFileDialog.getSaveFileName(
            self,
            self.tr("Save contour lines"),
            current or os.path.join(os.path.expanduser("~"), "contours.gpkg"),
            self.tr("GeoPackage (*.gpkg);;ESRI Shapefile (*.shp);;All files (*.*)")
        )
        if filename:
            lower = filename.lower()
            if not lower.endswith((".gpkg", ".shp")):
                filename += ".gpkg"
            target_edit.setText(filename)

    def toggle_raster_controls(self, checked):
        """Enable or disable raster export controls."""
        checked = bool(checked)
        for name in ("txt_raster_path", "btn_browse_r"):
            widget = getattr(self, name, None)
            if widget is not None:
                widget.setEnabled(checked)

        # Variance is a raster output too, so it cannot be saved without the
        # primary raster output being enabled.
        variance_check = getattr(self, "chk_save_variance", None)
        if variance_check is not None:
            variance_check.setEnabled(checked)
            if not checked:
                variance_check.setChecked(False)

        self.toggle_variance_controls(
            checked and bool(variance_check and variance_check.isChecked())
        ) if variance_check is not None else None

        vector_check = getattr(self, "chk_save_vector", None)
        if vector_check is not None:
            vector_check.setEnabled(checked)
            if not checked:
                vector_check.setChecked(False)
        self.toggle_vector_controls(
            checked and bool(vector_check and vector_check.isChecked())
        ) if vector_check is not None else None

    def toggle_variance_controls(self, checked):
        """Enable the optional kriging-variance output path."""
        enabled = bool(checked) and bool(
            getattr(self, "chk_save_raster", None)
            and self.chk_save_raster.isChecked()
        )
        for name in ("txt_variance_path", "btn_browse_variance"):
            widget = getattr(self, name, None)
            if widget is not None:
                widget.setEnabled(enabled)

    def toggle_vector_controls(self, checked):
        """Enable contour controls only when raster output is enabled."""
        enabled = bool(checked) and bool(
            getattr(self, "chk_save_raster", None)
            and self.chk_save_raster.isChecked()
        )
        for name in (
            "txt_vector_path_main", "btn_browse_v_main", "spin_step_main",
            "txt_vector_path_sec", "btn_browse_v_sec", "spin_step_sec",
        ):
            widget = getattr(self, name, None)
            if widget is not None:
                widget.setEnabled(enabled)

    def setup_export_tab(self):
        layout = QVBoxLayout(self.tab_export)

        box_raster = QGroupBox(self.tr("Interpolated Raster Output (GeoTIFF)"))
        lay_r = QVBoxLayout(box_raster)

        self.chk_save_raster = QCheckBox(self.tr("Save Raster File"))
        self.chk_save_raster.setChecked(True)
        self.chk_save_raster.toggled.connect(self.toggle_raster_controls)
        lay_r.addWidget(self.chk_save_raster)

        h_r = QHBoxLayout()
        self.txt_raster_path = QLineEdit()
        self.btn_browse_r = QPushButton(self.tr("Browse..."))
        self.btn_browse_r.clicked.connect(self.browse_raster)
        h_r.addWidget(self.txt_raster_path)
        h_r.addWidget(self.btn_browse_r)
        lay_r.addLayout(h_r)

        self.chk_save_variance = QCheckBox(self.tr("Save Kriging Variance"))
        self.chk_save_variance.setChecked(False)
        self.chk_save_variance.toggled.connect(self.toggle_variance_controls)
        lay_r.addWidget(self.chk_save_variance)

        h_var = QHBoxLayout()
        self.txt_variance_path = QLineEdit()
        self.btn_browse_variance = QPushButton(self.tr("Browse..."))
        self.btn_browse_variance.clicked.connect(self.browse_variance)
        h_var.addWidget(self.txt_variance_path)
        h_var.addWidget(self.btn_browse_variance)
        lay_r.addLayout(h_var)

        layout.addWidget(box_raster)

        box_vector = QGroupBox(self.tr("Contour Line Output"))
        lay_v = QVBoxLayout(box_vector)

        self.chk_save_vector = QCheckBox(self.tr("Save contour vector files"))
        self.chk_save_vector.setToolTip(
            self.tr("Contour vector output requires Save Raster File.")
        )
        self.chk_save_vector.setChecked(False)
        self.chk_save_vector.toggled.connect(self.toggle_vector_controls)
        lay_v.addWidget(self.chk_save_vector)

        lay_v.addWidget(QLabel(self.tr("<b>Major contours:</b>")))
        h_v1 = QHBoxLayout()
        self.txt_vector_path_main = QLineEdit()
        self.btn_browse_v_main = QPushButton(self.tr("Browse major..."))
        self.btn_browse_v_main.clicked.connect(lambda: self.browse_vector(self.txt_vector_path_main))
        h_v1.addWidget(self.txt_vector_path_main)
        h_v1.addWidget(self.btn_browse_v_main)
        lay_v.addLayout(h_v1)

        h_step1 = QHBoxLayout()
        h_step1.addWidget(QLabel(self.tr("Major contour interval (map units):")))
        self.spin_step_main = QDoubleSpinBox()
        self.spin_step_main.setRange(0.01, 10000.0)
        self.spin_step_main.setValue(50.0)
        h_step1.addWidget(self.spin_step_main)
        lay_v.addLayout(h_step1)

        lay_v.addWidget(QLabel(self.tr("<b>Minor contours:</b>")))
        h_v2 = QHBoxLayout()
        self.txt_vector_path_sec = QLineEdit()
        self.btn_browse_v_sec = QPushButton(self.tr("Browse minor..."))
        self.btn_browse_v_sec.clicked.connect(lambda: self.browse_vector(self.txt_vector_path_sec))
        h_v2.addWidget(self.txt_vector_path_sec)
        h_v2.addWidget(self.btn_browse_v_sec)
        lay_v.addLayout(h_v2)

        h_step2 = QHBoxLayout()
        h_step2.addWidget(QLabel(self.tr("Minor contour interval (map units):")))
        self.spin_step_sec = QDoubleSpinBox()
        self.spin_step_sec.setRange(0.01, 10000.0)
        self.spin_step_sec.setValue(10.0)
        h_step2.addWidget(self.spin_step_sec)
        lay_v.addLayout(h_step2)

        layout.addWidget(box_vector)

        self.toggle_raster_controls(self.chk_save_raster.isChecked())
        self.toggle_vector_controls(self.chk_save_vector.isChecked())
        self.toggle_variance_controls(self.chk_save_variance.isChecked())

        self.chk_add_to_project = QCheckBox(self.tr("Add generated layers to the QGIS project"))
        self.chk_add_to_project.setChecked(True)
        layout.addWidget(self.chk_add_to_project)

        box_process = QGroupBox(self.tr("Kriging Execution"))
        lay_p = QVBoxLayout(box_process)

        self.btn_run_interpolation = QPushButton(self.tr("Run Kriging and Create Outputs"))
        self.btn_run_interpolation.setStyleSheet(self.unified_btn_style)
        self.btn_run_interpolation.clicked.connect(self.run_kriging_execution)
        lay_p.addWidget(self.btn_run_interpolation)

        layout.addWidget(box_process)
