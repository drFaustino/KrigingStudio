# Kriging Studio

**Advanced geostatistical interpolation for QGIS 4.x**

Kriging Studio is a professional geostatistical interpolation plugin for **QGIS 4.x and Qt6**, designed for the analysis and spatial interpolation of point datasets using **Ordinary Kriging** and **Universal Kriging**.

The plugin provides a complete workflow that goes beyond simple interpolation: users can analyse and fit variograms, configure anisotropy, model regional trends using linear or quadratic drift, perform cross-validation, estimate prediction uncertainty, restrict interpolation using polygon masks, generate georeferenced GeoTIFF rasters, and automatically extract vector contour lines from kriging predictions.

---

## Main features

### Ordinary Kriging

Kriging Studio provides Ordinary Kriging for spatial interpolation of point observations.

The method estimates values at unsampled locations from the spatial correlation structure of the input dataset, using a configurable variogram model.

---

### Universal Kriging

Universal Kriging allows spatial trends to be incorporated into the interpolation model.

Kriging Studio supports:

- **Linear regional drift**
- **Quadratic regional drift**

This makes the plugin suitable for datasets where the spatial variable contains a large-scale trend in addition to local spatial variability.

---

### Variogram analysis

The plugin includes tools for analysing and modelling spatial dependence through variograms.

Features include:

- Experimental variogram calculation
- Fitted variogram visualization
- Configurable variogram parameters
- Comparison between experimental and fitted models
- Nugget, sill and range handling

Supported variogram models include:

- Spherical
- Exponential
- Gaussian
- Linear
- Power
- Hole-effect

The variogram analysis provides an important diagnostic step before performing the final interpolation.

---

### Anisotropy

Kriging Studio supports directional spatial dependence through configurable anisotropy.

Users can specify:

- Anisotropy scaling
- Anisotropy direction/angle

This allows the interpolation model to account for datasets where spatial correlation changes according to direction.

---

### Cross-validation

The plugin includes cross-validation tools for evaluating the quality and stability of the interpolation model.

Supported validation workflows include:

- K-fold cross-validation
- Leave-one-out cross-validation
- Prediction diagnostics
- Comparison between observed and predicted values
- Best-fit diagnostics
- Reference-line visualization

Cross-validation is performed using the selected QGIS point dataset rather than synthetic validation data.

---

### Uncertainty estimation

Kriging Studio can calculate additional information about prediction uncertainty.

Available outputs include:

- **Kriging prediction**
- **Prediction variance**
- **Standard error**

The kriging variance can be exported as a separate raster and optionally loaded automatically into QGIS.

These outputs allow users to evaluate not only the predicted spatial distribution but also the uncertainty associated with the interpolation.

---

### Polygon masking

Interpolation can optionally be restricted to a user-defined polygon area.

This is useful when the interpolation grid must be limited to:

- Study areas
- Administrative boundaries
- Watersheds
- Geological units
- Project boundaries
- Other areas of interest

The polygon mask prevents predictions from being generated outside the selected area.

---

### GeoTIFF raster generation

Kriging Studio generates georeferenced **GeoTIFF** rasters directly from the interpolation results.

Generated rasters include appropriate:

- Coordinate Reference System (CRS)
- Spatial extent
- North-up georeferencing
- Pixel-centre coordinates
- Compression
- Float32 data representation

The raster output can be automatically added to the current QGIS project.

---

### Automatic contour-line generation

Kriging Studio can generate **vector contour lines** directly from the kriging prediction raster.

This allows users to move from a continuous raster surface to a vector representation suitable for:

- Topographic-style maps
- Isopleth maps
- Geological interpretation
- Environmental analysis
- Thematic cartography

Contour-line output can be automatically loaded into QGIS together with the prediction raster.

> **Note:** contour-line generation requires the kriging prediction raster to be saved. The plugin therefore enables the contour option only when raster output is selected.

---

## Performance and large datasets

Kriging Studio includes several optimizations designed to improve performance when processing large datasets and interpolation grids.

These include:

- Local-neighbourhood kriging
- Adaptive neighbourhood sizes
- Chunked prediction
- Memory-conscious raster generation
- Float32 GeoTIFF output
- Improved cancellation handling
- Robust processing of large interpolation grids

Adaptive local neighbourhood processing can use different neighbourhood sizes depending on the computational requirements of the dataset and grid.

---

## QGIS integration

Kriging Studio is designed as a native QGIS plugin.

It can work directly with:

- QGIS vector layers
- Point datasets
- Numeric attribute fields
- Project layers
- QGIS coordinate reference systems

The plugin can automatically refresh available project layers and numeric fields.

Generated outputs can optionally be added directly to the active QGIS project.

---

## Output management

Kriging Studio provides independent output controls for the main interpolation products.

Depending on the selected options, the plugin can generate:

| Output | Description |
|---|---|
| Prediction raster | Main kriging interpolation surface |
| Kriging variance | Spatial estimation uncertainty |
| Standard error | Standard deviation associated with prediction uncertainty |
| Contour lines | Vector isolines derived from the prediction raster |

The prediction raster is the basis for contour-line generation.

Therefore:

**Save Raster → required for contour-line output**

If raster output is disabled, the contour-line option is automatically disabled as well.

---

## Supported interpolation models

### Ordinary Kriging

Ordinary Kriging assumes an unknown but locally constant mean and estimates the spatially varying prediction surface from the spatial covariance/variogram structure.

### Universal Kriging

Universal Kriging extends the model by incorporating a deterministic spatial trend.

Kriging Studio supports:

```text
Universal Kriging
├── Linear drift
└── Quadratic drift
```

This is particularly useful when the sampled variable exhibits a large-scale directional or polynomial trend.

---

## Variogram models

The following variogram models are available:

| Model | Description |
|---|---|
| Spherical | Common bounded model with a finite range |
| Exponential | Smooth asymptotic approach toward the sill |
| Gaussian | Smooth model suitable for strongly continuous spatial behaviour |
| Linear | Linear variogram structure |
| Power | Flexible power-law spatial dependence |
| Hole-effect | Model capable of representing periodic or oscillatory spatial structures |

The experimental variogram can be compared visually with the fitted model before running the interpolation.

---

## Cross-validation diagnostics

Cross-validation provides an important quality-control step for the interpolation workflow.

The validation tools can be used to identify:

- Systematic prediction bias
- Poor model fitting
- Outliers
- Spatial prediction errors
- Differences between observed and predicted values
- General interpolation performance

Diagnostic plots include configurable visualization parameters such as point size and graph colours.

---

## Installation

### From a ZIP package

1. Download the Kriging Studio ZIP package.
2. Open **QGIS**.
3. Go to:

   **Plugins → Manage and Install Plugins…**

4. Select:

   **Install from ZIP**

5. Select the Kriging Studio ZIP file.
6. Confirm the installation.
7. Enable **Kriging Studio** from the Installed Plugins list.

---

## Requirements

Kriging Studio is designed for:

- **QGIS 4.x**
- **Qt6**

The plugin metadata declares:

```text
Minimum QGIS version: 4.00
Maximum QGIS version: 4.99
```

Kriging Studio does not provide a QGIS Processing provider.

---

## Python dependencies
Kriging Studio uses the following Python scientific-computing libraries for numerical calculations and spatial interpolation:

```text
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.distance import cdist
```

### NumPy
NumPy is used for numerical array operations, matrix calculations, vectorized computations, interpolation grids, and efficient handling of kriging data.

The plugin relies on NumPy for:

- Numerical arrays and matrices
- Vectorized distance and coordinate calculations
- Kriging system construction
- Grid generation and prediction
- Statistical and numerical operations
- Memory-efficient handling of large datasets

### SciPy
SciPy provides numerical optimization and spatial-distance functionality used by the geostatistical algorithms.

Kriging Studio uses:

- scipy.optimize.least_squares for nonlinear least-squares fitting of variogram models
- scipy.spatial.distance.cdist for efficient pairwise distance calculations between coordinate sets

These components are used particularly during variogram fitting, spatial-neighbourhood calculations, and kriging prediction.

The required imports are:

```text
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.distance import cdist
```

The plugin therefore requires a QGIS Python environment with NumPy and SciPy available.

## Project structure

The project is organised as a standard QGIS plugin and includes the main components required for:

- Plugin initialization
- User interface
- Kriging calculations
- Variogram analysis
- Cross-validation
- Raster generation
- Vector contour generation
- Translation support
- QGIS layer integration
- Resources and icons

---

## Internationalization

Kriging Studio includes Qt translation infrastructure for multilingual support.

User-facing elements are prepared for translation, including:

- Interface labels
- Dialogs
- Table headers
- Descriptions
- Menu values
- Plot labels
- User messages

Internal enumeration and data values remain stable so that translation does not affect the computational logic.

---

## Numerical robustness

Several improvements have been implemented to improve the stability of the interpolation workflow.

These include:

- Stable linear-system solving
- Duplicate-point handling
- Anisotropy-safe distance calculations
- Correct nugget handling
- Adaptive local neighbourhoods
- Chunked grid prediction
- Large-grid memory management
- Real leave-one-out cross-validation
- Robust cancellation and background processing

The implementation is intended to provide reliable interpolation while keeping memory and processing requirements manageable for large grids.

---

## Version

Current release:

**Kriging Studio 1.0.0**

Although the current metadata version is `1.0.0`, the changelog documents the development history and subsequent internal improvements associated with the project.

---

## Author

**Dr. Geol. Faustino Cetraro**

---

## License

See the `LICENSE` file included in the project repository for licensing information.

---

## Disclaimer

Kriging Studio is a scientific and geostatistical analysis tool. The quality of interpolation results depends on the characteristics of the input dataset, sampling density, spatial distribution of observations, selected variogram model, anisotropy parameters, drift specification, and other modelling assumptions.

Cross-validation and uncertainty outputs should be used as diagnostic tools and interpreted together with knowledge of the study area and the underlying physical or environmental process.

The software does not replace professional geostatistical analysis or domain-specific validation where results are used for scientific, engineering, environmental, regulatory, or decision-making purposes.

---

## Interfaces

<img width="1004" height="880" alt="img1" src="https://github.com/user-attachments/assets/62dc835a-906f-4b73-a40d-31ae11da1564" />

<img width="999" height="882" alt="img2" src="https://github.com/user-attachments/assets/e5e47839-86cf-4265-97f2-e8e8f79b6b45" />
