# SAR_GSD — SAR Ground Surface Deformation Detection

A Python package for detecting ground surface deformation using Sentinel-1 SAR imagery through temporal trend analysis of radar backscatter intensity. It combines cloud-based satellite data acquisition, GPU-accelerated tensor processing, and geospatial analysis to identify areas experiencing persistent change.

## Features

- **Sentinel-1 data acquisition** via Sentinel Hub API (cloud-free, all-weather radar imagery)
- **GPU-accelerated processing** using PyTorch tensor operations for multi-scale Gaussian smoothing and linear regression
- **Change detection** based on annualised temporal trends with configurable thresholds
- **DEM and slope integration** from OpenTopography
- **Raster-vector analysis** — zonal statistics, buffering, vectorisation of change masks
- **Multiple output formats** — GeoTIFF, NetCDF, GeoPackage, KML, PNG, interactive HTML maps (Folium)
- **Infrastructure corridor analysis** — overlay change maps with vector datasets (e.g. transmission lines)

## Installation

```bash
pip install -r requirements.txt
```

### API credentials

Create a `.env` file in the project root:

```
SENTINEL_HUB_CLIENT_ID=<your_id>
SENTINEL_HUB_CLIENT_SECRET=<your_secret>
KEY_OPEN_TOPOGRAPHY=<your_key>
```

- Sentinel Hub credentials: [https://www.sentinel-hub.com/](https://www.sentinel-hub.com/)
- OpenTopography API key: [https://opentopography.org/](https://opentopography.org/)

## Quick start

```python
from SAR_GSD import Config, build_datacube, process_sar_timeseries, get_device

# 1. Download Sentinel-1 time series
bbox = [-42.85, -19.65, -42.48, -19.40]
datacube, dates = build_datacube(bbox, "2017-01-01", resolution=20, max_dates=40)

# 2. Run tensor-accelerated processing pipeline
results = process_sar_timeseries(datacube)

# 3. Access results
trend         = results["trend"]            # Annualised trend (y, x)
positive_mask = results["positive_mask"]    # Areas with increasing backscatter
negative_mask = results["negative_mask"]    # Areas with decreasing backscatter
```

See [`SARGSD_Example.ipynb`](SARGSD_Example.ipynb) for a complete walkthrough including visualisation, DEM integration, vector analysis, and interactive mapping.

## Package structure

```
SAR_GSD/
├── SAR_GSD/
│   ├── __init__.py              # Public API exports
│   ├── config.py                # Centralised configuration and defaults
│   ├── download.py              # Sentinel Hub data acquisition and DEM download
│   ├── processing.py            # Trend analysis, smoothing, change detection (NumPy/SciPy)
│   ├── processing_tensor.py     # Tensor-accelerated alternative pipeline (PyTorch)
│   ├── visualization.py         # Plotting and interactive maps
│   ├── export.py                # GeoTIFF, KML export
│   ├── vector_ops.py            # Buffering, zonal statistics, vectorisation
│   └── utils.py                 # Google Earth launcher, summary reports
├── tests/
│   └── tests.py                 # Unit tests (synthetic datacube)
├── SARGSD_Example.ipynb         # Full example notebook
├── requirements.txt
├── LICENSE                      # GNU GPLv3
└── .env                         # API credentials (not tracked)
```

## Modules

### `download`

| Function | Description |
|---|---|
| `build_datacube()` | Download Sentinel-1 time series and build a 3-D xarray datacube |
| `get_available_s1_dates()` | Query available acquisition dates from Sentinel Hub |
| `request_s1_image()` | Download a single Sentinel-1 image |
| `get_dem()` | Retrieve DEM data from OpenTopography |

### `processing`

| Function | Description |
|---|---|
| `process_sar_timeseries()` | End-to-end pipeline: trend + smoothing + change detection |
| `compute_temporal_trend()` | Pixel-wise linear regression with p-values |
| `apply_spatial_smoothing()` | Gaussian + mean filter smoothing |
| `detect_changes()` | Generate binary change masks from trend and p-values |
| `calculate_slope()` | Compute slope from DEM |

### `processing_tensor`

GPU-accelerated alternative pipeline using PyTorch tensors throughout.

| Function | Description |
|---|---|
| `process_sar_timeseries_tensor()` | Full pipeline with multi-scale Gaussian smoothing and tensor regression |
| `tensor_smooth_timeseries()` | Batched multi-scale Gaussian convolution on GPU/CPU |
| `tensor_temporal_trend()` | Pixel-wise linear regression returning trend and R² |

### `visualization`

| Function | Description |
|---|---|
| `plot_sar_intensity()` | Display a single SAR intensity image |
| `plot_change_overlay()` | Overlay change masks on SAR imagery |
| `plot_trend_map()` | Diverging colourmap of temporal trend |
| `plot_time_series()` | Time series plot for a single pixel |
| `create_all_figures()` | Generate all standard figures at once |
| `display_Folium_map()` | Interactive web map with multiple layers |

### `vector_ops`

| Function | Description |
|---|---|
| `zonal_statistics()` | Compute statistics within polygon geometries |
| `buff_and_clip_geopackage()` | Buffer and clip vector features |
| `calculate_change_areas()` | Quantify change area within polygons |
| `vectorize_and_analyze_changes()` | Vectorise masks and compute zonal stats |

### `export`

| Function | Description |
|---|---|
| `save_all_outputs()` | Save trend and masks as GeoTIFF |
| `gdf_to_kml()` | Export GeoDataFrame to KML for Google Earth |

## Configuration

Default parameters are defined in `SAR_GSD/config.py` and can be overridden:

| Parameter | Default | Description |
|---|---|---|
| `RESOLUTION` | 20 | Spatial resolution in metres |
| `POLARIZATION` | `"VV"` | SAR polarization |
| `CHANGE_THRESHOLD` | 0.05 | Minimum trend magnitude (log-units/year) |
| `P_VALUE_THRESHOLD` | 0.05 | Statistical significance level |
| `GAUSSIAN_SIGMA` | `[1.5, 2.5]` | Multi-scale smoothing sigmas |
| `MEAN_FILTER_SIZE` | 3 | Mean filter kernel size |
| `MAX_DATES` | 40 | Maximum number of acquisitions |

## Testing

```bash
python -m pytest tests/
```

The test suite uses a synthetic datacube (10 time steps, 50 x 50 pixels) to validate array operations, tensor processing, linear regression, change detection, and NumPy-PyTorch consistency.

## License

This project is licensed under the [GNU General Public License v3.0](LICENSE).