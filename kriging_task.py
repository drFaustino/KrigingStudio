# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import numpy as np
from osgeo import gdal
from qgis.core import QgsTask, QgsMessageLog, Qgis, QgsRasterLayer


class KrigingCalculationTask(QgsTask):
    """Cancelable background task for point-kriging grid generation."""

    def __init__(
        self, description, engine, coords, values, grid_shape, grid_bounds,
        output_path, crs_wkt="", max_neighbors=None
    ):
        super().__init__(description, QgsTask.CanCancel)
        self.engine = engine
        self.coords = np.asarray(coords, dtype=float)
        self.values = np.asarray(values, dtype=float)
        self.grid_shape = tuple(grid_shape)
        self.grid_bounds = tuple(grid_bounds)
        self.output_path = output_path
        self.crs_wkt = crs_wkt
        self.max_neighbors = max_neighbors
        self.result_layer = None
        self.error_message = ""
        self.predictions = None
        self.variances = None

    def run(self):
        try:
            rows, cols = self.grid_shape
            xmin, xmax, ymin, ymax = self.grid_bounds
            if rows < 2 or cols < 2:
                raise ValueError("The output grid must contain at least two rows and columns.")
            if not (xmax > xmin and ymax > ymin):
                raise ValueError("The output extent is invalid.")

            gx = np.linspace(xmin, xmax, cols)
            gy = np.linspace(ymax, ymin, rows)
            grid_x, grid_y = np.meshgrid(gx, gy)
            self.setProgress(20)

            self.predictions, self.variances = self.engine.fit_and_predict(
                self.coords, self.values, grid_x, grid_y,
                return_variance=False, max_neighbors=self.max_neighbors
            )
            if self.isCanceled():
                return False
            self.setProgress(70)

            os.makedirs(os.path.dirname(os.path.abspath(self.output_path)), exist_ok=True)
            driver = gdal.GetDriverByName("GTiff")
            ds = driver.Create(self.output_path, cols, rows, 1, gdal.GDT_Float32,
                               options=["COMPRESS=DEFLATE", "TILED=YES"])
            if ds is None:
                raise RuntimeError("Unable to create the GeoTIFF output.")
            ds.SetGeoTransform((xmin, (xmax-xmin)/cols, 0.0,
                                ymax, 0.0, -(ymax-ymin)/rows))
            if self.crs_wkt:
                ds.SetProjection(self.crs_wkt)
            band = ds.GetRasterBand(1)
            band.WriteArray(np.asarray(self.predictions, dtype=np.float32))
            band.SetNoDataValue(-3.4028235e38)
            band.FlushCache()
            ds.FlushCache()
            ds = None
            self.setProgress(100)
            return True
        except Exception as exc:
            self.error_message = str(exc)
            QgsMessageLog.logMessage(
                f"Kriging task failed: {exc}", "KrigingStudio", Qgis.Critical
            )
            return False

    def finished(self, result):
        if result and os.path.exists(self.output_path):
            self.result_layer = QgsRasterLayer(self.output_path, "Kriging Surface")
            QgsMessageLog.logMessage(
                "Kriging task completed successfully.", "KrigingStudio", Qgis.Success
            )
        elif self.isCanceled():
            QgsMessageLog.logMessage(
                "Kriging task cancelled.", "KrigingStudio", Qgis.Warning
            )
        else:
            QgsMessageLog.logMessage(
                f"Kriging task failed: {self.error_message}",
                "KrigingStudio", Qgis.Critical
            )
