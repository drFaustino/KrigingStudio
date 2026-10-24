# KrigingStudio - UI responsiveness fixes

## Changes
- Moved leave-one-out cross-validation to a cancellable `QgsTask`.
- Moved experimental variogram calculation and AutoFit optimization to cancellable background tasks.
- Moved GeoTIFF writing and GDAL contour generation to a background task; layer registration and dialogs remain on the GUI thread.
- Routed task progress through Qt signals before updating widgets, rather than updating Qt widgets from worker threads.
- Connected kriging-task progress to the persistent bottom progress bar.
- Added task guards to prevent overlapping operations and disabled action buttons while background work is active.
- GeoTIFF output now uses `BIGTIFF=IF_SAFER` and converts non-finite cells to NoData.

## Validation
- Python syntax compilation passed for all plugin `.py` files.
- Runtime testing inside QGIS (including task cancellation, GDAL contour output and project layer registration) is still required. QGIS/GDAL are not available in this build environment, so no claim of in-app integration testing is made.


## Fix v3
- Added missing `browse_raster`, `browse_variance`, and `browse_vector` methods used by export-tab buttons.
- File dialogs now set output paths and add expected `.tif` / `.gpkg` extensions when omitted.
