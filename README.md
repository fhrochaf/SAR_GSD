# SAR_GSD: SAR Ground Surface Deformation Detection

A Python package for detecting ground surface changes using Sentinel-1 SAR imagery through temporal trend analysis.

## Features

- **Automated data acquisition** from Sentinel Hub API
- **rioxarray-based datacubes** with full geospatial metadata
- **Time series analysis** using linear regression
- **Change detection** with configurable thresholds
- **Publication-quality visualizations**
- **KML export** for Google Earth visualization
- **Modular architecture** for easy extension

## Installation

### Prerequisites

- Python 3.8+
- Sentinel Hub account (free tier available)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/your-username/SAR_GSD.git
cd SAR_GSD
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure credentials:
   - Copy `.env.example` to `.env` (if exists) or create `.env` file
   - Add your Sentinel Hub credentials:
   ```
   SENTINEL_HUB_CLIENT_ID=your_client_id
   SENTINEL_HUB_CLIENT_SECRET=your_client_secret
   ```
   - Get free credentials at: https://apps.sentinel-hub.com/dashboard/

## Quick Start

### Using the Jupyter Notebook

The easiest way to get started is with the example notebook:

```bash
jupyter notebook SARGSD_Example.ipynb
```

The notebook walks through the complete pipeline:
1. Build SAR datacube from Sentinel-1 imagery
2. Process time series to detect trends
3. Generate visualizations
4. Export results to KML and GeoTIFF

### Using the Python Package

```python
from SAR_GSD import Config, build_datacube, process_sar_timeseries

# Define area of interest
bbox = [-42.85, -19.65, -42.48, -19.40]  # [lon_min, lat_min, lon_max, lat_max]

# Build datacube
datacube, dates = build_datacube(
    bbox_coords=bbox,
    start_date="2020-01-01",
    resolution=20,
    max_dates=40
)

# Process time series
results = process_sar_timeseries(
    datacube=datacube,
    threshold=0.07
)

# Access results
trend = results['trend']
positive_mask = results['positive_mask']
negative_mask = results['negative_mask']
```

## Project Structure

```
SAR_GSD/
├── SAR_GSD/              # Python package
│   ├── __init__.py       # Package initialization
│   ├── config.py         # Configuration management
│   ├── download.py       # Data acquisition functions
│   ├── processing.py     # Time series analysis
│   ├── visualization.py  # Plotting functions
│   └── export.py         # Export utilities (KML, GeoTIFF)
├── outputs/              # Output directory (auto-created)
├── SARGSD_Example.ipynb  # Example notebook
├── .env                  # API credentials (not in git)
├── .gitignore
└── README.md
```

## Configuration

All configuration is managed through the `Config` class in `SAR_GSD/config.py`:

```python
from SAR_GSD import Config

# Access settings
print(Config.OUTPUT_DIR)
print(Config.CHANGE_THRESHOLD)
print(Config.DEFAULT_RESOLUTION)

# Use predefined study areas
area = Config.get_study_area('ipatinga_test')
datacube, dates = build_datacube(**area)
```

## Output Files

All outputs are saved to the `outputs/` directory:

| File | Description | Format |
|------|-------------|--------|
| `sar_datacube.nc` | Complete SAR time series | NetCDF |
| `sar_latest.tif` | Most recent acquisition | GeoTIFF |
| `sar_trend.tif` | Temporal trend map | GeoTIFF |
| `positive_change_mask.tif` | Positive change mask | GeoTIFF |
| `negative_change_mask.tif` | Negative change mask | GeoTIFF |
| `sar_intensity.png` | Intensity visualization | PNG |
| `sar_change_overlay.png` | Change overlay | PNG |
| `sar_trend_map.png` | Trend visualization | PNG |
| `sar_change_detection.kml` | Google Earth polygons | KML |
| `summary.txt` | Analysis summary | Text |

## Working with rioxarray

The datacubes are fully compatible with rioxarray for geospatial operations:

```python
import xarray as xr

# Load datacube
datacube = xr.open_dataarray('outputs/sar_datacube.nc')

# Select specific date
img = datacube.sel(time='2023-01-15')

# Clip to geometry
import geopandas as gpd
aoi = gpd.read_file('my_aoi.geojson')
clipped = datacube.rio.clip(aoi.geometry, aoi.crs)

# Reproject
reprojected = datacube.rio.reproject('EPSG:3857')

# Export
datacube.isel(time=0).rio.to_raster('output.tif')
```

## Methodology

The pipeline implements a log-ratio change detection approach:

1. **Data Acquisition**: Downloads Sentinel-1 GRD VV polarization data
2. **Normalization**: Each image is normalized by its median value
3. **Log Transformation**: Applies log10 to reduce multiplicative noise
4. **Trend Analysis**: Fits linear regression to time series at each pixel
5. **Spatial Smoothing**: Applies Gaussian + mean filters to reduce speckle
6. **Thresholding**: Identifies pixels with |trend| > threshold

**Interpretation:**
- **Positive trend (blue)**: Increasing backscatter → construction, vegetation growth, surface roughening
- **Negative trend (red)**: Decreasing backscatter → subsidence, deforestation, surface smoothing

## API Reference

### Main Functions

#### `build_datacube(bbox_coords, start_date, resolution=20, ...)`
Downloads and builds a georeferenced SAR datacube.

#### `process_sar_timeseries(datacube, threshold=0.07, ...)`
Analyzes time series and detects changes.

#### `create_all_figures(datacube, trend, positive_mask, negative_mask, ...)`
Generates all standard visualizations.

#### `save_all_outputs(datacube, trend_da, positive_mask_da, ...)`
Saves all output files (datacube, trend, masks, KML).

See docstrings in each module for detailed API documentation.

## Examples

### Custom Study Area

```python
from SAR_GSD import build_datacube, process_sar_timeseries

# Define custom AOI
my_bbox = [-48.5, -15.8, -48.3, -15.6]

# Build and process
datacube, _ = build_datacube(
    bbox_coords=my_bbox,
    start_date="2021-01-01",
    end_date="2023-12-31",
    resolution=10,  # Higher resolution
    max_dates=50
)

results = process_sar_timeseries(datacube, threshold=0.05)
```

### Time Series Analysis at Specific Location

```python
from SAR_GSD import plot_time_series

# Plot time series for a specific pixel
fig = plot_time_series(
    datacube,
    row=100,
    col=150,
    save_path="outputs/pixel_timeseries.png"
)
```

### Batch Processing Multiple AOIs

```python
from SAR_GSD import Config

# Process all predefined study areas
for area_name in Config.list_study_areas():
    area = Config.get_study_area(area_name)
    datacube, _ = build_datacube(**area)
    results = process_sar_timeseries(datacube)
    # Save with area-specific names...
```

## Requirements

Key dependencies:
- `numpy` - Numerical computing
- `xarray` - Multi-dimensional arrays
- `rioxarray` - Geospatial xarray extension
- `rasterio` - Raster I/O
- `sentinelhub` - Sentinel Hub API
- `matplotlib` - Plotting
- `scipy` - Scientific computing
- `torch` - Tensor operations (for convolutions)
- `simplekml` - KML generation
- `python-dotenv` - Environment variables

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Add your license here]

## Citation

If you use this package in your research, please cite:

```bibtex
[Add citation information]
```

## Acknowledgments

- Sentinel-1 data provided by ESA Copernicus programme
- Sentinel Hub for API access
- rioxarray developers

## Contact

[Add contact information]

## References

- [Sentinel-1 Product Specification](https://sentinel.esa.int/web/sentinel/missions/sentinel-1)
- [Sentinel Hub Documentation](https://docs.sentinel-hub.com/)
- [rioxarray Documentation](https://corteva.github.io/rioxarray/)
