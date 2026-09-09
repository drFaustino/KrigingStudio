def classFactory(iface):
    from .kriging_plugin import KrigingStudioPlugin
    return KrigingStudioPlugin(iface)
