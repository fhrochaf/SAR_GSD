# SAR_GSD: SAR Ground Surface Deformation Detection

A Python package for detecting ground surface changes using Sentinel-1 SAR imagery through temporal trend analysis.

## Features

- **Automated data acquisition** from Sentinel Hub API
- **rioxarray-based datacubes** with full geospatial metadata
- **Time series analysis** using linear regression with GPU acceleration
- **Change detection** with configurable thresholds and p-value filtering
- **DEM integration** with slope analysis from OpenTopography
- **Zonal statistics** for vector-raster analysis
- **Publication-quality visualizations** with matplotlib and Folium
- **KML export** for Google Earth visualization
- **Modular architecture** for easy extension

## Installation

### Prerequisites

- Python 3.9+
- CUDA-compatible GPU (optional, for accelerated processing)
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

3. Configure credentials in `.env` file:
```env
SENTINEL_HUB_CLIENT_ID=your_client_id
SENTINEL_HUB_CLIENT_SECRET=your_client_secret
KEY_OPEN_TOPOGRAPHY=your_opentopo_key
```

Get credentials at:
- Sentinel Hub: https://apps.sentinel-hub.com/dashboard/
- OpenTopography: https://opentopography.org/

## Quick Start

```python
from SAR_GSD import *

# Define area of interest [lon_min, lat_min, lon_max, lat_max]
AOI_BBOX = [-42.85, -19.65, -42.48, -19.40]

# Build datacube from Sentinel-1 time series
datacube, dates = build_datacube(
    bbox_coords=AOI_BBOX,
    start_date="2020-01-01",
    resolution=20,
    max_dates=30,
    polarization="VV"
)

# Process time series and detect changes
results = process_sar_timeseries(
    datacube=datacube['backscatter'],
    threshold=0.07,
    verbose=True
)

# Export results
save_all_outputs(
    trend=results['trend'],
    pvalues=results['pvalues'],
    positive_mask=results['positive_mask'],
    negative_mask=results['negative_mask'],
    bbox=AOI_BBOX
)
```

## Package Structure

```
SAR_GSD/
    __init__.py          # Package exports
    config.py            # Configuration management
    download.py          # Sentinel-1 & DEM data acquisition
    processing.py        # Time series analysis & change detection
    visualization.py     # Plotting & Folium maps
    export.py            # KML & GeoTIFF export
    vector_ops.py        # Zonal statistics & vector operations
    utils.py             # Utility functions
```

## Main Functions

### Data Acquisition

| Function | Description |
|----------|-------------|
| `build_datacube()` | Build georeferenced xarray datacube from Sentinel-1 time series |
| `get_available_s1_dates()` | Query available acquisition dates |
| `request_s1_image()` | Download single Sentinel-1 image |
| `get_dem()` | Download DEM from OpenTopography |

### Processing

| Function | Description |
|----------|-------------|
| `process_sar_timeseries()` | Complete pipeline: trend, smoothing, detection |
| `calculate_slope()` | Compute slope from DEM data |
| `get_device()` | Detect GPU/CPU for PyTorch acceleration |

### Visualization

| Function | Description |
|----------|-------------|
| `create_all_figures()` | Generate all standard matplotlib figures |
| `display_Folium_map()` | Create interactive web map from GeoDataFrames |
| `add_raster_to_folium()` | Add raster overlay to Folium map |
| `plot_trend_map()` | Visualize temporal trend |
| `plot_change_overlay()` | SAR intensity with change overlay |
| `plot_sar_intensity()` | Single SAR intensity image |
| `plot_time_series()` | Time series for specific pixel |
| `plot_pvalues_map()` | P-values significance map |

### Export

| Function | Description |
|----------|-------------|
| `save_all_outputs()` | Save all GeoTIFF outputs |
| `gdf_to_kml()` | Export GeoDataFrame to KML for Google Earth |
| `launch_google_earth()` | Open one or more KML files in Google Earth Pro |

### Vector Operations

| Function | Description |
|----------|-------------|
| `zonal_statistics()` | Compute zonal stats for polygons against raster |
| `vectorize_and_analyze_changes()` | Vectorize masks with slope/trend statistics |
| `calculate_change_areas()` | Compute change areas within zones |
| `buff_and_clip_geopackage()` | Buffer and clip vector data |
| `vectorize_mask()` | Convert raster mask to vector polygons |

### Utilities

| Function | Description |
|----------|-------------|
| `launch_google_earth()` | Launch Google Earth with KML files |
| `create_summary_report()` | Generate text summary of analysis |

## Configuration

All parameters are managed through the `Config` class:

```python
from SAR_GSD import Config

# View current settings
print(Config.CHANGE_THRESHOLD)   # 0.07
print(Config.DEFAULT_RESOLUTION) # 20m
print(Config.OUTPUT_DIR)

# Get output path
path = Config.get_output_path("results.tif")

# Validate credentials
if Config.validate_sentinel_hub_credentials():
    print("Credentials OK")
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CHANGE_THRESHOLD` | 0.07 | Trend threshold (log-units/year) |
| `P_VALUE_THRESHOLD` | 0.05 | Statistical significance level |
| `GAUSSIAN_SIGMA` | 1.5 | Spatial smoothing sigma (pixels) |
| `MEAN_FILTER_SIZE` | 3 | Mean filter kernel size |
| `DEFAULT_RESOLUTION` | 20 | Spatial resolution (meters) |
| `MAX_DATES` | 40 | Maximum acquisitions to process |
| `POLARIZATION` | VV | SAR polarization mode |

## Output Files

| File | Format | Description |
|------|--------|-------------|
| `sar_datacube.nc` | NetCDF | Time series datacube |
| `sar_trend.tif` | GeoTIFF | Temporal trend map |
| `sar_pvalues.tif` | GeoTIFF | P-values from regression |
| `positive_change_mask.tif` | GeoTIFF | Positive change binary mask |
| `negative_change_mask.tif` | GeoTIFF | Negative change binary mask |
| `dem.tif` | GeoTIFF | Digital elevation model |
| `slope.tif` | GeoTIFF | Slope map (degrees) |
| `sar_change_detection.kml` | KML | Google Earth visualization |
| `summary.txt` | Text | Analysis summary report |

## Methodology

The pipeline implements a log-ratio change detection approach:

1. **Data Acquisition**: Downloads Sentinel-1 GRD VV polarization data
2. **Normalization**: Each image is normalized by its median value
3. **Log Transformation**: Applies log10 to reduce multiplicative noise
4. **Trend Analysis**: Fits linear regression to time series at each pixel
5. **Spatial Smoothing**: Applies Gaussian + mean filters to reduce speckle
6. **Thresholding**: Identifies pixels with |trend| > threshold AND p-value < 0.05

### Interpretation

- **Positive trend (blue)**: Increasing backscatter - construction, vegetation growth, surface roughening
- **Negative trend (red)**: Decreasing backscatter - subsidence, deforestation, surface smoothing

## Examples

### Complete Workflow with DEM Analysis

```python
from SAR_GSD import *
import xarray as xr

# Build datacube
datacube, dates = build_datacube(
    bbox_coords=AOI_BBOX,
    start_date="2020-01-01",
    resolution=20,
    max_dates=40
)

# Process SAR time series
results = process_sar_timeseries(datacube['backscatter'], threshold=0.07)

# Add results to datacube
datacube["trend"] = (["y", "x"], results["trend"])
datacube["positive_mask"] = (["y", "x"], results["positive_mask"])
datacube["negative_mask"] = (["y", "x"], results["negative_mask"])

# Download and add DEM
dem = get_dem(AOI_BBOX)
datacube["dem"] = (["y", "x"], dem.values)

# Calculate slope
slope = calculate_slope(datacube["dem"])
datacube["slope"] = (["y", "x"], slope.values)

# Create slope classes
slope_classes = {1: "0-5", 2: "5-10", 3: "10-20", 4: "20-45", 5: "45-75", 6: "75-90"}
datacube["slope_class"] = (["y", "x"], np.digitize(slope.values, [0, 5, 10, 20, 45, 75]))
datacube["slope_class"].attrs['Slope classes'] = slope_classes

# Vectorize changes with slope statistics
gdf_changes = vectorize_and_analyze_changes(datacube, verbose=True)
```

### Zonal Statistics with Vector Data

```python
import geopandas as gpd

# Load vector data
gdf_zones = gpd.read_file("zones.gpkg")

# Compute change areas within zones
gdf_result = calculate_change_areas(
    gdf_zones,
    datacube["positive_mask"],
    datacube["negative_mask"],
    resolution=20,
    output_path="outputs/zones_with_changes.gpkg"
)
```

### Interactive Folium Map

```python
from SAR_GSD import display_Folium_map, add_raster_to_folium

# Create map with vector layers
geojson_display_dicts = [
    {
        'data': gdf_changes,
        'name': 'SAR Changes',
        'attribute_map': {
            "change_type": "Change Type",
            "mean_trend": "Mean Trend",
            "total_area_ha": "Area (ha)"
        },
        'feature_settings': {
            "fillColor": "blue",
            "color": "black",
            "weight": 1,
            "fillOpacity": 0.5
        }
    }
]

fmap = display_Folium_map(geojson_display_dicts, zoom_start=12)

# Add raster layer
fmap = add_raster_to_folium(
    fmap,
    datacube["backscatter"][-1],
    name="SAR Intensity",
    cmap='gray',
    log_transform=True
)

# Save map
fmap.save("outputs/interactive_map.html")
```

### Export to KML

```python
from SAR_GSD import gdf_to_kml, launch_google_earth

# Export GeoDataFrame to KML
gdf_to_kml(
    gdf_changes,
    "outputs/changes.kml",
    name_column="change_type",
    description_columns=["mean_trend", "total_area_ha"],
    color="ffff0000"  # Blue in KML format (AABBGGRR)
)

# Open multiple KMLs in Google Earth
launch_google_earth([
    "outputs/sar_change_detection.kml",
    "outputs/changes.kml",
    "outputs/zones.kml"
])
```

## Requirements

Core dependencies:
- `numpy`, `pandas`, `scipy` - Scientific computing
- `xarray`, `rioxarray` - Multi-dimensional geospatial arrays
- `geopandas`, `fiona`, `shapely` - Vector data handling
- `rasterio`, `rasterstats` - Raster operations
- `sentinelhub` - Sentinel Hub API
- `matplotlib`, `folium`, `branca` - Visualization
- `torch` - GPU-accelerated convolutions
- `simplekml` - KML generation
- `Pillow` - Image processing
- `python-dotenv` - Environment variables

## License

MIT License

## References

- [Sentinel-1 Product Specification](https://sentinel.esa.int/web/sentinel/missions/sentinel-1)
- [Sentinel Hub Documentation](https://docs.sentinel-hub.com/)
- [rioxarray Documentation](https://corteva.github.io/rioxarray/)
- [OpenTopography](https://opentopography.org/)
