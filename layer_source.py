# -*- coding: utf-8 -*-
"""QGIS layer integration helpers for Kriging Studio.

* ``ZSourceComboBox``: selector of the observed-value (Z) source. It lists the
  numeric attribute fields of the current vector layer and, for single-part
  point layers with Z values (PointZ, PointZM, Point25D), an additional
  "geometry Z coordinate" entry. It refreshes itself when the layer fields
  change.
* ``extract_points``: reads the XY/Z samples from a layer.

The Z values of the geometry are interpolated along the layer XY coordinates:
this is 2D surface kriging, not 3D volumetric kriging. For PointZM data the M
coordinate is not used.
"""
from __future__ import annotations

import math

import numpy as np
from qgis.PyQt.QtCore import QCoreApplication, Qt, pyqtSignal
from qgis.PyQt.QtWidgets import QComboBox
from qgis.core import Qgis, QgsFeatureRequest, QgsProject, QgsWkbTypes

# Internal key of the "geometry Z coordinate" entry. Field names can never
# collide with it because it is not a valid QGIS field name.
GEOMETRY_Z_KEY = "__geometry_z__"


_IT = {
    "<Geometry Z coordinate>": "<Coordinata Z geometrica>",
    "Use the Z coordinate of each point as the observed value (M is ignored). 2D surface kriging.":
        "Usa la coordinata Z di ogni punto come valore osservato (la M è ignorata). Kriging di superficie 2D.",
    "The geometry Z coordinate requires a single-part point layer with Z values (PointZ, PointZM or Point25D).":
        "La coordinata Z geometrica richiede un layer a punto singolo con valori Z (PointZ, PointZM o Point25D).",
    "The selected Z field does not exist in the layer.": "Il campo Z selezionato non esiste nel layer.",
}


def _tr(text):
    translated = QCoreApplication.translate("KrigingLayerSource", text)
    if translated != text:
        return translated
    # Runtime fallback for strings missing from the compiled .qm catalog.
    from qgis.PyQt.QtCore import QLocale
    if QLocale.system().name().lower().startswith("it"):
        return _IT.get(text, text)
    return translated


# --------------------------------------------------------------------------
# Layer inspection
# --------------------------------------------------------------------------
def geometry_kind(layer):
    """Return 'point', 'line', 'polygon' or 'other' for a vector layer."""
    try:
        gt = layer.geometryType()
    except (AttributeError, RuntimeError):
        return "other"
    for name, kind in (("Point", "point"), ("Line", "line"), ("Polygon", "polygon")):
        try:
            if gt == getattr(Qgis.GeometryType, name):
                return kind
        except AttributeError:
            continue
    try:  # legacy enums
        if gt == QgsWkbTypes.PointGeometry:
            return "point"
        if gt == QgsWkbTypes.LineGeometry:
            return "line"
        if gt == QgsWkbTypes.PolygonGeometry:
            return "polygon"
    except AttributeError:
        pass
    return "other"


def layer_supports_geometry_z(layer):
    """True for single-part point layers carrying Z (PointZ, PointZM, Point25D)."""
    if layer is None or geometry_kind(layer) != "point":
        return False
    try:
        wkb = layer.wkbType()
        return bool(QgsWkbTypes.hasZ(wkb)) and not bool(QgsWkbTypes.isMultiType(wkb))
    except (AttributeError, RuntimeError):
        return False


def numeric_field_names(layer):
    """Names of the numeric attribute fields of a layer, in layer order."""
    names = []
    try:
        for field in layer.fields():
            if field.isNumeric():
                names.append(field.name())
    except (AttributeError, RuntimeError):
        pass
    return names


# --------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------
def _finite(*values):
    return all(math.isfinite(v) for v in values)


def extract_points(layer, source_key, is_cancelled=None):
    """Read the observations of ``layer``.

    ``source_key`` is either ``GEOMETRY_Z_KEY`` (use the Z coordinate of each
    single-part point) or the name of a numeric attribute field.

    Returns ``(coords, values, stats)`` where ``coords`` is an N x 2 float array,
    ``values`` an N float array and ``stats`` a dict with the keys ``total``,
    ``used`` and ``skipped``. Features with empty, multi-part (geometry Z only),
    missing or non-finite XY/Z values are skipped. Duplicate (x, y, z) samples
    are dropped.
    """
    kind = geometry_kind(layer)
    if kind not in ("point", "line"):
        return (np.empty((0, 2), dtype=float), np.empty(0, dtype=float),
                {"total": 0, "used": 0, "skipped": 0})

    use_geometry_z = source_key == GEOMETRY_Z_KEY
    request = QgsFeatureRequest()
    if use_geometry_z:
        if not layer_supports_geometry_z(layer):
            raise ValueError(_tr("The geometry Z coordinate requires a single-part point layer "
                                 "with Z values (PointZ, PointZM or Point25D)."))
        try:
            request.setSubsetOfAttributes([])  # geometry only: no attribute reading
        except (TypeError, AttributeError):
            pass
    else:
        if layer.fields().indexFromName(source_key) < 0:
            raise ValueError(_tr("The selected Z field does not exist in the layer."))
        try:
            request.setSubsetOfAttributes([source_key], layer.fields())
        except (TypeError, AttributeError):
            pass

    coords, vals = [], []
    seen = set()
    total = 0
    for n, feat in enumerate(layer.getFeatures(request)):
        total += 1
        if is_cancelled is not None and n % 2048 == 0 and is_cancelled():
            raise RuntimeError("Operation cancelled.")
        geom = feat.geometry()
        if geom is None or geom.isNull() or geom.isEmpty():
            continue

        if use_geometry_z:
            if geom.isMultipart():
                continue
            pt = geom.constGet()
            try:
                x, y, z = float(pt.x()), float(pt.y()), float(pt.z())
            except (AttributeError, TypeError, ValueError):
                continue
            samples = [(x, y, z)] if _finite(x, y, z) else []
        else:
            val = feat.attribute(source_key)
            if val is None:
                continue
            try:
                z = float(val)  # QGIS NULL raises TypeError here
            except (TypeError, ValueError):
                continue
            if not math.isfinite(z):
                continue
            try:
                if kind == "point":
                    points = geom.asMultiPoint() if geom.isMultipart() else [geom.asPoint()]
                else:
                    # vertices() also covers multipart and curved lines
                    points = list(geom.vertices())
            except (TypeError, ValueError, AttributeError):
                continue
            samples = []
            for p in points:
                x, y = float(p.x()), float(p.y())
                if _finite(x, y):
                    samples.append((x, y, z))

        for x, y, z in samples:
            key = (x, y, z)
            if key in seen:
                continue
            seen.add(key)
            coords.append((x, y))
            vals.append(z)

    coords_arr = np.asarray(coords, dtype=float).reshape((-1, 2))
    vals_arr = np.asarray(vals, dtype=float)
    stats = {"total": total, "used": len(vals_arr), "skipped": max(total - len(vals_arr), 0)}
    return coords_arr, vals_arr, stats


# --------------------------------------------------------------------------
# Selector widget
# --------------------------------------------------------------------------
class ZSourceComboBox(QComboBox):
    """Z-value source selector: numeric fields + optional geometry Z coordinate.

    API-compatible with the parts of ``QgsFieldComboBox`` used by the dialog
    (``setLayer``, ``currentField``). It keeps itself up to date when the fields
    of the layer change, and drops the layer when it is removed from the project.
    """

    # Emitted when the selected source changes because of the user or because
    # the layer fields changed (not on a plain setLayer call).
    sourceChanged = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._layer_id = None
        self._layer = None
        self.setMinimumContentsLength(14)
        self.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.activated.connect(lambda _i: self.sourceChanged.emit())
        project = QgsProject.instance()
        project.layersWillBeRemoved.connect(self._on_layers_will_be_removed)
        project.cleared.connect(self._on_project_cleared)

    # ----- public API -----
    def layer(self):
        if self._layer_id is None:
            return None
        return QgsProject.instance().mapLayer(self._layer_id)

    def currentField(self):
        """Selected attribute field name, GEOMETRY_Z_KEY, or '' when empty."""
        key = self.currentData(Qt.ItemDataRole.UserRole)
        return key or ""

    def isGeometryZ(self):
        return self.currentField() == GEOMETRY_Z_KEY

    def setLayer(self, layer):
        self._disconnect_layer()
        if layer is None:
            self._layer_id = None
            self._layer = None
        else:
            self._layer_id = layer.id()
            self._layer = layer
            try:
                layer.updatedFields.connect(self._on_fields_updated)
            except (AttributeError, TypeError):
                pass
        self._populate(preferred=None)

    def refresh(self):
        """Re-read the fields of the current layer, keeping the selection."""
        self._populate(preferred=self.currentField() or None)

    # ----- internals -----
    def _disconnect_layer(self):
        if self._layer is not None:
            try:
                self._layer.updatedFields.disconnect(self._on_fields_updated)
            except (TypeError, RuntimeError, AttributeError):
                pass
        self._layer = None

    def _populate(self, preferred):
        layer = self.layer()
        previous = self.currentField() if preferred is None and self.count() else preferred
        was_blocked = self.blockSignals(True)
        try:
            self.clear()
            if layer is not None:
                if layer_supports_geometry_z(layer):
                    self.addItem(_tr("<Geometry Z coordinate>"), GEOMETRY_Z_KEY)
                    self.setItemData(0, _tr("Use the Z coordinate of each point as the observed value "
                                            "(M is ignored). 2D surface kriging."),
                                     Qt.ItemDataRole.ToolTipRole)
                for name in numeric_field_names(layer):
                    self.addItem(name, name)
            index = self.findData(previous) if previous else -1
            if index < 0 and self.count():
                # Default: first numeric field; geometry Z only if there are none.
                index = 1 if (self.count() > 1 and self.itemData(0) == GEOMETRY_Z_KEY) else 0
            self.setCurrentIndex(index)
        finally:
            self.blockSignals(was_blocked)

    def _on_fields_updated(self):
        before = self.currentField()
        self.refresh()
        if self.currentField() != before:
            self.sourceChanged.emit()

    def _on_project_cleared(self):
        self.setLayer(None)
        self.sourceChanged.emit()

    def _on_layers_will_be_removed(self, layer_ids):
        if self._layer_id in layer_ids:
            self.setLayer(None)
            self.sourceChanged.emit()
