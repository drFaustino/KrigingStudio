# -*- coding: utf-8 -*-
import os

from qgis.PyQt.QtCore import QCoreApplication, QTranslator, QLocale
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtWidgets import QAction, QApplication
from qgis.core import QgsProject

from .kriging_dialog import KrigingDialog


class KrigingStudioPlugin:
    """QGIS plugin entry point for Kriging Studio."""

    def __init__(self, iface):
        self.iface = iface
        self.plugin_dir = os.path.dirname(__file__)
        self.action = None
        self.dlg = None
        self.translator = None

    def tr(self, text):
        return QCoreApplication.translate("KrigingStudioPlugin", text)

    def initGui(self):
        self._load_translator()
        icon_path = os.path.join(self.plugin_dir, "resources", "images", "icon.svg")
        self.action = QAction(QIcon(icon_path), self.tr("Kriging Studio"), self.iface.mainWindow())
        self.action.setToolTip(self.tr("Open Kriging Studio"))
        self.action.triggered.connect(self.run)
        self.iface.addPluginToMenu(self.tr("&Kriging Studio"), self.action)
        self.iface.addToolBarIcon(self.action)

    def _load_translator(self):
        locale = QLocale.system().name()
        language = locale.split("_")[0]

        base = os.path.join(
            self.plugin_dir,
            "resources",
            "i18n"
        )

        candidates = [
            f"KrigingStudio_{locale}.qm",
            f"KrigingStudio_{language}.qm",
        ]

        for filename in candidates:
            path = os.path.join(base, filename)

            if not os.path.isfile(path):
                continue

            translator = QTranslator()

            if translator.load(path):
                QApplication.installTranslator(translator)
                self.translator = translator
                return

    def _unload_translator(self):
        if self.translator is not None:
            QApplication.removeTranslator(self.translator)
            self.translator = None

    def unload(self):
        if self.action is not None:
            self.iface.removePluginMenu(self.tr("&Kriging Studio"), self.action)
            self.iface.removeToolBarIcon(self.action)
            self.action.deleteLater()
            self.action = None
        self._unload_translator()

    def run(self):
        if self.dlg is None:
            self.dlg = KrigingDialog(self.iface.mainWindow())
        # Pick up layers/fields changed in the project and prefer the active layer.
        self.dlg.refresh_layers_and_fields(self.iface.activeLayer())
        self.dlg.show()
        self.dlg.raise_()
        self.dlg.activateWindow()
