# SAR_GSD - SAR Ground Surface Deformation Detection

A Python package for detecting and quantifying ground surface changes using multi-temporal Sentinel-1 SAR (Synthetic Aperture Radar) imagery. The pipeline builds georeferenced datacubes, applies tensor-accelerated speckle filtering and trend analysis, and integrates the results with terrain and infrastructure data for applied geospatial analysis.

## Motivation

In engineering and infrastructure management, land use changes beneath transmission lines demand continuous attention. Erosion and landslides can expose tower foundations, potentially leading to power supply disruptions, structural damage, and fire hazards. Understanding terrain dynamics is critical for maintaining grid reliability and public safety.

SAR_GSD was developed as an exercise in addressing this challenge. It leverages SAR image datacubes for initial analysis of backscatter intensity changes over time, performing speckle reduction through multi-scale Gaussian smoothing followed by GPU-accelerated pixel-wise linear regression to detect temporal trends. The package also computes terrain slope, classifies it into slope classes, and overlays change detection results with infrastructure vector data for quantitative corridor-level assessment.

> **Note:** This project was initially created for course-related didactic purposes and therefore has some limitations and scope restrictions.

## Pipeline Overview

The full workflow, demonstrated in `SARGSD_Example.ipynb`, follows nine steps:

| Step | Section | Description |
|------|---------|-------------|
| 1 | **Imports & Configuration** | Load the package and set analysis parameters (AOI, dates, resolution, thresholds, polarization) |
| 2 | **Build Datacube** | Query Sentinel Hub for Sentinel-1 IW GRD acquisitions, download backscatter images, normalize by median, and stack into a 3D xarray DataArray saved as NetCDF |
| 3 | **Process Time Series** | Tensor-accelerated multi-scale Gaussian smoothing followed by pixel-wise linear regression to compute annualized trend and R-squared maps, then threshold-based change detection |
| 4 | **Visualizations** | Generate SAR intensity, change overlay, and trend map figures |
| 5 | **Export Outputs** | Save trend and change mask rasters as GeoTIFF, generate a text summary report |
| 6 | **DEM & Slope** | Download a DEM from OpenTopography, compute slope in degrees, classify into slope categories |
| 7 | **Raster-Vector Operations** | Buffer transmission line geometries, compute zonal statistics of change areas within infrastructure corridors, vectorize change masks and analyze by slope class |
| 8 | **Summary** | Build an interactive Folium map with raster and vector layers; perform NumPy-based infrastructure corridor analysis |
| 9 | **Google Earth Export** | Export transmission line buffers and change polygons as KML files |

## Package Structure

```
SAR_GSD/
├── SAR_GSD/                  # Main package
│   ├── __init__.py           # Public API exports
│   ├── config.py             # Central configuration (paths, credentials, parameters)
│   ├── download.py           # Sentinel-1 data acquisition and DEM download
│   ├── processing.py         # Tensor-accelerated SAR processing (smoothing, regression)
│   ├── visualization.py      # Matplotlib figures and Folium interactive maps
│   ├── export.py             # GeoTIFF and KML export
│   ├── vector_ops.py         # Buffering, zonal statistics, vectorization
│   └── utils.py              # Summary reports and Google Earth launcher
├── data/                     # Input vector data (e.g., TransmissionGrid.gpkg)
├── outputs/                  # Generated rasters, figures, KML, and reports
├── tests/                    # Unit tests
├── SARGSD_Example.ipynb      # Full pipeline example notebook
├── requirements.txt          # Python dependencies
├── LICENSE                   # GNU GPLv3
└── .env                      # API credentials (not tracked by git)
```

## Module Details

### `config.py` - Configuration

All parameters are managed through the `Config` class:

- **Project paths** - `PROJECT_ROOT`, `DATA_DIR`, `OUTPUT_DIR`, `CACHE_DIR`
- **API credentials** - Loaded from `.env` via `python-dotenv` (`SENTINEL_HUB_CLIENT_ID`, `SENTINEL_HUB_CLIENT_SECRET`, `KEY_OPEN_TOPOGRAPHY`)
- **Sentinel-1 parameters** - Platform, product type (GRD), sensor mode (IW), polarization (VV), resolution (20 m)
- **Temporal parameters** - Start date (`2017-01-01`), max acquisitions (40), revisit time (12 days)
- **Change detection** - Trend threshold (0.05 log-units/year), p-value threshold (0.05), Gaussian sigmas (`[1.5, 2.5]`), weights (`[0.8, 0.2]`)
- **AOI** - Default bounding box covering a region in Minas Gerais, Brazil: `[-42.85, -19.65, -42.48, -19.40]`

### `download.py` - Data Acquisition

| Function | Description |
|----------|-------------|
| `build_datacube()` | Queries available Sentinel-1 dates via Sentinel Hub Catalog, downloads each acquisition, normalizes by median, and assembles a georeferenced xarray DataArray with dimensions `(time, y, x)` and EPSG:4326 CRS. Supports even subsampling when `max_dates` is set. |
| `get_available_s1_dates()` | Queries the Sentinel Hub Catalog for available Sentinel-1 IW acquisition dates within a bounding box and time range. |
| `request_s1_image()` | Downloads a single Sentinel-1 GRD image for a given date using a custom Evalscript that extracts the specified polarization channel. |
| `get_dem()` | Retrieves a DEM raster from the OpenTopography SRTM GL3 API and saves it as GeoTIFF. |

### `processing.py` - Tensor-Accelerated SAR Processing

This module implements the core change detection analysis using PyTorch:

| Function | Description |
|----------|-------------|
| `process_sar_timeseries_tensor()` | Full pipeline: prepares tensors, applies multi-scale Gaussian smoothing, computes pixel-wise linear regression (annualized trend + R-squared), and generates change masks. Returns a dictionary with `trend`, `r_squared`, `smoothed`, `positive_mask`, `negative_mask`, and `processing_time`. |
| `tensor_smooth_timeseries()` | Applies multi-scale Gaussian smoothing via 2D convolution (`torch.nn.functional.conv2d`). Each sigma produces a smoothed copy of the time series; the final result is a weighted combination. Processes in batches to manage memory (16 frames for GPU, 8 for CPU). |
| `tensor_temporal_trend()` | Pixel-wise ordinary least-squares regression over the time dimension. Computes slope (annualized to units/year based on actual time span), R-squared, and returns the smoothed array. |
| `detect_changes()` | Applies trend and p-value thresholds to generate boolean masks for positive and negative changes. |
| `get_device()` | Auto-detects CUDA GPU availability and returns the appropriate `torch.device`. |
| `calculate_slope()` | Computes terrain slope from a projected DEM using `numpy.gradient`, with output in degrees or percent. |

**Processing pipeline summary:**

1. **Prepare tensor** - Extract backscatter values, replace NaNs with zero, move to GPU/CPU.
2. **Multi-scale Gaussian smoothing** - Convolve each time step with Gaussian kernels at multiple scales (default sigma = 1.5 and 2.5), then combine with configurable weights (default 0.8 / 0.2). This reduces speckle while preserving change signatures at different spatial scales.
3. **Pixel-wise linear regression** - Fit a trend line to each pixel's smoothed time series. The slope is annualized by dividing by the actual time span in years. R-squared measures goodness of fit.
4. **Change detection** - Classify pixels as positive change (trend > threshold), negative change (trend < -threshold), or stable.

### `visualization.py` - Visualization

| Function | Description |
|----------|-------------|
| `plot_sar_intensity()` | Displays a single SAR intensity image (optional log-scale) with colorbar and date annotation. |
| `plot_change_overlay()` | Overlays blue (positive) and red (negative) change masks on a grayscale SAR intensity image. |
| `plot_trend_map()` | Displays the annualized trend map with a symmetric diverging colormap and threshold markers on the colorbar. |
| `plot_time_series()` | Plots the backscatter time series for a single pixel. |
| `create_all_figures()` | Convenience function that generates and saves all standard figures (intensity, overlay, trend map). |
| `display_Folium_map()` | Creates an interactive Folium web map from a list of GeoDataFrame layer configurations with tooltips, popups, and custom styling. |
| `add_raster_to_folium()` | Adds a raster layer (xarray DataArray) to an existing Folium map as a base64-encoded PNG overlay, with automatic reprojection to WGS84 and configurable colormap. |

### `export.py` - Export

| Function | Description |
|----------|-------------|
| `save_all_outputs()` | Saves the trend map, positive mask, and negative mask as georeferenced GeoTIFF files. |
| `gdf_to_kml()` | Exports any GeoDataFrame to KML format for Google Earth, with automatic reprojection to WGS84, custom styling (line color, fill, width), and HTML description popups. Supports Point, LineString, and Polygon geometries. |

### `vector_ops.py` - Raster-Vector Operations

| Function | Description |
|----------|-------------|
| `buff_and_clip_geopackage()` | Opens a GeoPackage with Fiona, buffers each geometry by a given distance, clips to an AOI polygon, and saves the result as a new GeoPackage. |
| `zonal_statistics()` | Computes configurable zonal statistics (mean, min, max, std, count, sum, median) for a raster DataArray within polygon geometries using `rasterstats`. |
| `calculate_change_areas()` | Performs zonal statistics on positive and negative change masks within polygon features, computing pixel counts, areas in hectares, and percentages relative to total polygon area. |
| `vectorize_mask()` | Converts a binary raster mask to vector polygons using `rasterio.features.shapes`. |
| `vectorize_and_analyze_changes()` | Full vectorization pipeline: converts change masks to polygons, computes per-polygon zonal statistics against the trend raster and slope classes, and returns a combined GeoDataFrame with positive and negative change polygons. |

### `utils.py` - Utilities

| Function | Description |
|----------|-------------|
| `create_summary_report()` | Generates a text report with acquisition metadata, trend statistics, change detection counts, and output file paths. |
| `launch_google_earth()` | Attempts to open KML files in Google Earth Pro (supports Windows, macOS, Linux paths). Falls back to printing manual instructions. |

## Detailed Step-by-Step Guide

### Step 1 - Configuration

```python
from SAR_GSD import *

AOI_BBOX = Config.DEFAULT_AOI_BBOX        # [-42.85, -19.65, -42.48, -19.40]
START_DATE = Config.DEFAULT_START_DATE     # "2017-01-01"
END_DATE = None                            # None = today
MAX_DATES = Config.MAX_DATES              # 40
RESOLUTION = Config.DEFAULT_RESOLUTION     # 20 meters
CHANGE_THRESHOLD = Config.CHANGE_THRESHOLD # 0.05 log-units/year
POLARIZATION = Config.POLARIZATION         # "VV"
```

All parameters can be edited directly in the notebook or permanently in `SAR_GSD/config.py`. API credentials are loaded from a `.env` file in the project root.

### Step 2 - Build Datacube

```python
datacube, valid_dates = build_datacube(
    bbox_coords=AOI_BBOX,
    start_date=START_DATE,
    end_date=END_DATE,
    resolution=RESOLUTION,
    max_dates=MAX_DATES,
    polarization=POLARIZATION,
    normalize=True,
    verbose=True,
)
datacube.to_netcdf(Config.get_output_path(Config.DATACUBE_FILENAME))
```

This step:
1. Queries the Sentinel Hub Catalog for all available Sentinel-1 IW GRD dates within the time range.
2. Subsamples evenly to `max_dates` acquisitions if the catalog returns more.
3. Downloads each image via the Sentinel Hub Process API using a custom Evalscript.
4. Replaces invalid values (zero or negative backscatter) with NaN.
5. Normalizes each image by its median to reduce radiometric variation across acquisitions.
6. Stacks all images into a 3D xarray DataArray with dimensions `(time, y, x)`, geographic coordinates, and EPSG:4326 CRS.
7. Saves as NetCDF for later reuse.

In the example case (Ipatinga, Minas Gerais), this produces a datacube of shape `(46, 1358, 1961)` covering 2017-01-03 to 2025-11-29.

### Step 3 - Process Time Series

```python
results = process_sar_timeseries_tensor(datacube['backscatter'])

datacube["trend"] = (["y", "x"], results["trend"])
datacube["backscatter_smoothed_tensor"] = (["time", "y", "x"], results["smoothed"])
datacube["r_squared_tensor"] = (["y", "x"], results["r_squared"])
datacube["positive_mask"] = (["y", "x"], results["positive_mask"].astype(np.uint8))
datacube["negative_mask"] = (["y", "x"], results["negative_mask"].astype(np.uint8))
```

The tensor-accelerated pipeline:
1. Converts the backscatter array to a PyTorch tensor, replacing NaNs with zero.
2. Applies multi-scale Gaussian smoothing with configurable sigma values (default 1.5 and 2.5 pixels) and combination weights (default 0.8 and 0.2). The smoothing is performed via 2D convolution (`F.conv2d`) in batches to manage memory.
3. Computes pixel-wise linear regression over the time dimension, producing an annualized trend (slope in log-units/year) and R-squared map.
4. Classifies pixels exceeding the threshold as positive or negative change.

GPU acceleration (CUDA) significantly speeds up the smoothing step when available; the pipeline falls back to CPU otherwise.

### Step 4 - Visualizations

```python
figures = create_all_figures(
    datacube=datacube['backscatter'],
    trend=datacube['trend'],
    positive_mask=positive_mask,
    negative_mask=negative_mask,
    threshold=CHANGE_THRESHOLD,
    output_dir=str(Config.OUTPUT_DIR),
    dpi=Config.DPI,
    show=True,
)
```

Generates three figures:
- **SAR intensity image** - Latest acquisition in log-scale grayscale.
- **Change overlay** - Grayscale intensity with blue (positive trend) and red (negative trend) masks overlaid.
- **Trend map** - Annualized trend values with a symmetric diverging colormap (`RdBu_r`) and threshold markers.

Interpretation:
- **Blue areas** (positive trend): Increasing backscatter, possibly indicating construction, vegetation growth, or surface roughening.
- **Red areas** (negative trend): Decreasing backscatter, possibly indicating subsidence, deforestation, or surface smoothing.

### Step 5 - Export Outputs

```python
output_files = save_all_outputs(
    datacube["trend"],
    positive_mask=datacube["positive_mask"],
    negative_mask=datacube["negative_mask"],
    bbox=AOI_BBOX,
    output_dir=Config.OUTPUT_DIR,
    verbose=True,
)

report_path = create_summary_report(
    outputs=output_files,
    datacube=datacube['backscatter'],
    trend_da=datacube['trend'],
    positive_mask=datacube['positive_mask'],
    negative_mask=datacube['negative_mask'],
)
```

Saves:
- `sar_trend.tif` - Georeferenced trend raster.
- `positive_change_mask.tif` - Binary positive change mask.
- `negative_change_mask.tif` - Binary negative change mask.
- `summary.txt` - Text report with acquisition metadata, trend statistics, and change detection counts.

### Step 6 - DEM, Slope, and Slope Classes

```python
dem = get_dem(AOI_BBOX)
dem_reprojected = dem.rio.reproject("EPSG:31983")
slope = calculate_slope(dem_reprojected)
```

1. Downloads a SRTM GL3 DEM from OpenTopography for the AOI.
2. Reprojects the DEM to a metric CRS (EPSG:31983 - SIRGAS 2000 / UTM 23S) for slope computation.
3. Computes slope in degrees using `numpy.gradient`.
4. Creates slope classes: 0-5, 5-10, 10-20, 20-45, 45-75, and 75-90 degrees.
5. Reprojects DEM, slope, and slope classes to match the datacube grid and adds them as new variables.
6. Reprojects the entire datacube to EPSG:31983 for downstream vector analysis in meters.

### Step 7 - Raster-Vector Operations

```python
# Buffer and clip transmission lines
path_buff = buff_and_clip_geopackage(path_tl, path_buff, buffer_distance=50, clip_geom=aoi_projected)

# Zonal statistics: change areas within each corridor
gdf_tl = calculate_change_areas(gdf_buff, datacube["positive_mask"], datacube["negative_mask"], res, output_path=path_zs)

# Vectorize change masks and analyze by slope class
gdf_result = vectorize_and_analyze_changes(datacube)
```

This section performs integrated raster-vector analysis:
1. Loads a GeoPackage of transmission line geometries (from Brazil's EPE).
2. Buffers each line by 50 m and clips to the AOI using Fiona for low-level geometry processing.
3. Computes zonal statistics to determine the area of positive/negative SAR change within each transmission line corridor (in hectares and as a percentage).
4. Vectorizes the change masks into polygons and computes per-polygon zonal statistics against the trend raster and slope classes, producing a GeoDataFrame with area, mean trend, and slope-class breakdown for each change polygon.

### Step 8 - Interactive Folium Map and Infrastructure Analysis

```python
fmap = display_Folium_map(geojson_display_dicts, zoom_start=12)
add_raster_to_folium(fmap, sar_first, name=f"SAR Intensity ({first_date})", ...)
add_raster_to_folium(fmap, datacube["dem"], name="Elevation (m)", ...)
```

Builds a multi-layer interactive web map with:
- **Transmission line buffers** - Blue polygons with tooltips showing change area statistics.
- **SAR change polygons** - Colored by trend magnitude using a diverging Spectral colormap. Polygons intersecting infrastructure are highlighted with red outlines.
- **Raster overlays** - First and last SAR intensity images, DEM elevation, and slope classes.
- **Layer control** - Toggle layers on/off.

The map is saved as an HTML file for sharing or embedding.

A NumPy-based infrastructure corridor analysis is also performed, which rasterizes vector buffers and computes the percentage of detected changes occurring within transmission line corridors vs. the wider study area.

### Step 9 - Google Earth Export

```python
gdf_to_kml(gdf_tl, output_path="outputs/transmission_line_buffers.kml", color="ff00ff00", ...)
gdf_to_kml(gdf_result[gdf_result["change_type"] == "positive"], output_path="outputs/positive_change.kml", ...)
gdf_to_kml(gdf_result[gdf_result["change_type"] == "negative"], output_path="outputs/negative_change.kml", ...)
launch_google_earth([...])
```

Exports three KML files:
- **Transmission line buffers** - Green polygons with change area attributes.
- **Positive change polygons** - Blue polygons with trend and slope statistics.
- **Negative change polygons** - Red polygons with trend and slope statistics.

KML files can be viewed in Google Earth Pro, Google Earth Web, or any GIS application.

## Prerequisites

### API Credentials

1. **Sentinel Hub** (for SAR data download): Register for free credentials at https://apps.sentinel-hub.com/dashboard/
2. **OpenTopography** (for DEM data): Register for a free API key at https://portal.opentopography.org/requestService?service=api

Create a `.env` file in the project root:

```
SENTINEL_HUB_CLIENT_ID=your_client_id
SENTINEL_HUB_CLIENT_SECRET=your_client_secret
KEY_OPEN_TOPOGRAPHY=your_opentopography_key
```

> If you do not have API credentials, pre-built data files (datacube and DEM) are available at:
> https://drive.google.com/drive/folders/1lOqUf__jXbOH_zP7iyHVX51mvLmqrTRg?usp=sharing
>
> Download the contents to the `outputs/` folder and skip the download steps in the notebook.

### Installation

```bash
pip install -r requirements.txt
```

### Dependencies

| Category | Packages |
|----------|----------|
| Scientific computing | numpy, pandas, scipy |
| Geospatial | xarray, rioxarray, geopandas, fiona, shapely, rasterio, rasterstats, pyproj |
| Sentinel Hub API | sentinelhub |
| Visualization | matplotlib, folium, branca, Pillow |
| KML export | simplekml |
| GPU acceleration | torch (PyTorch) |
| HTTP & environment | requests, python-dotenv |

### GPU Support

The processing module uses PyTorch for tensor-accelerated spatial filtering. A CUDA-capable GPU significantly speeds up the Gaussian smoothing step. The pipeline automatically detects and uses GPU if available, otherwise falls back to CPU.

## Output Files

The pipeline generates the following outputs in the `outputs/` directory:

| File | Format | Description |
|------|--------|-------------|
| `sar_datacube.nc` | NetCDF | 3D SAR backscatter datacube (time, y, x) |
| `sar_trend.tif` | GeoTIFF | Annualized trend map (log-units/year) |
| `positive_change_mask.tif` | GeoTIFF | Binary mask of positive changes |
| `negative_change_mask.tif` | GeoTIFF | Binary mask of negative changes |
| `dem.tif` | GeoTIFF | Digital Elevation Model |
| `slope.tif` | GeoTIFF | Slope in degrees |
| `sar_intensity.png` | PNG | SAR intensity figure |
| `sar_change_overlay.png` | PNG | Change overlay figure |
| `sar_trend_map.png` | PNG | Trend map figure |
| `TransmissionGrid_50m_buffer.gpkg` | GeoPackage | Buffered transmission line corridors |
| `TransmissionGrid_zs.gpkg` | GeoPackage | Transmission lines with zonal statistics |
| `transmission_line_buffers.kml` | KML | Transmission buffers for Google Earth |
| `positive_change.kml` | KML | Positive change polygons for Google Earth |
| `negative_change.kml` | KML | Negative change polygons for Google Earth |
| `SARGSD_Example.html` | HTML | Interactive Folium web map |
| `summary.txt` | Text | Analysis summary report |

## Limitations

- This pipeline uses **intensity-based analysis** (backscatter log-ratio change detection with linear regression). It is sensitive to persistent temporal trends but less sensitive to rapid or seasonal changes.
- The analysis does not include **interferometric SAR (InSAR)** processing, which would provide phase-based deformation measurements.
- The change detection threshold is a single global value; adaptive thresholding per land cover type is not implemented.
- Sentinel-1 GRD products do not preserve phase information, limiting the analysis to amplitude-based methods.

This work establishes a scalable baseline for future InSAR implementation. The goal is to continue evolving the package with more sophisticated preprocessing and analysis capabilities.

## License

This project is licensed under the GNU General Public License v3.0. See [LICENSE](LICENSE) for details.

## Data Sources

- **SAR imagery**: Sentinel-1 IW GRD via [Sentinel Hub](https://www.sentinel-hub.com/)
- **DEM**: SRTM GL3 via [OpenTopography](https://portal.opentopography.org/)
- **Transmission lines**: Brazil's EPE (Empresa de Pesquisa Energetica) - https://gisepeprd2.epe.gov.br/webmapepe/