# SAR Ground Deformation Detection

A Python-based system for detecting and analyzing ground deformation (landslides, erosion, sedimentation) using Sentinel-1 SAR time series analysis with GRD (Ground Range Detected) products.

## Overview

This project implements a **SAR-based workflow** for monitoring ground deformation using Sentinel-1 GRD imagery. It provides tools for:

- **Data Acquisition**: Download Sentinel-1 GRD products from Copernicus Data Space Ecosystem
- **Time Series Analysis**: Create and analyze spatio-temporal data cubes
- **Deformation Detection**: Identify linear trends, change points, and deformation hotspots
- **Visualization**: Generate publication-quality maps, plots, and animations
- **Landslide Monitoring**: Assess landslide susceptibility and track deformation patterns

### Scientific Background

The methodology is inspired by established InSAR techniques but adapted for GRD intensity analysis:
- **Ferretti et al. (2001)** - Persistent Scatterer Interferometry (PSI) concepts
- **Berardino et al. (2002)** - Small Baseline Subset (SBAS) approach
- **Handwerger et al. (2019)** - Sentinel-1 landslide monitoring

This implementation uses **GRD products** for rapid prototyping and change detection. It can be extended to **SLC-based interferometric processing** (true InSAR) for scientifically rigorous phase-based displacement analysis.

## Installation

### Prerequisites
- Python 3.9 or higher
- Git

### Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd SAR_GSD
   ```

2. **Create virtual environment** (recommended):
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Install package in development mode**:
   ```bash
   pip install -e .
   ```

5. **Configure credentials**:
   - Copy `.env.example` to `.env`
   - Register at [Copernicus Data Space](https://dataspace.copernicus.eu/)
   - Add your credentials to `.env`:
     ```
     COPERNICUS_USERNAME=your_username
     COPERNICUS_PASSWORD=your_password
     ```

## Quick Start

### 1. Download Sentinel-1 SAR Data

```python
from insar.download import SentinelDownloader

# Initialize SAR data downloader
downloader = SentinelDownloader()

# Define area of interest (bounding box)
bbox = [-8.7, 40.5, -8.3, 40.8]  # [min_lon, min_lat, max_lon, max_lat]

# Query and download products
products_df, files = downloader.download_by_area_and_dates(
    bbox=bbox,
    start_date="2023-01-01",
    end_date="2023-12-31",
    max_products=10
)
```

### 2. Create Data Cube

```python
from insar.cubes import create_datacube_from_files, save_datacube

# Create time series data cube
datacube = create_datacube_from_files(
    file_paths=files,
    band_name="intensity"
)

# Save for later use
save_datacube(datacube, "data/processed/datacube.nc")
```

### 3. Analyze Deformation

```python
from insar.timeseries import detect_linear_trends
from insar.deformation import calculate_velocity_map, identify_deformation_hotspots

# Detect linear deformation trends
velocity, pvalue, rsquared = detect_linear_trends(datacube)

# Identify deformation hotspots
hotspots, stats = identify_deformation_hotspots(
    velocity,
    threshold=20.0,  # mm/year
    min_size=10  # pixels
)

print(f"Found {len(stats)} deformation hotspots")
```

### 4. Visualize Results

```python
from insar.visualization import plot_velocity_map, plot_timeseries

# Plot velocity map
fig = plot_velocity_map(
    velocity,
    title="Deformation Velocity Map",
    save_path="outputs/velocity_map.png"
)

# Plot time series for specific pixel
fig = plot_timeseries(
    datacube,
    pixel_coords=(100, 150),
    save_path="outputs/timeseries.png"
)
```

## Project Structure

```
InSAR_GroundSD/
├── data/
│   ├── raw/              # Downloaded Sentinel-1 SAR products
│   ├── processed/        # Processed data cubes
│   └── external/         # DEM, auxiliary data
├── src/insar/            # Main SAR analysis package
│   ├── download.py       # SAR data acquisition
│   ├── cubes.py          # Data cube management
│   ├── timeseries.py     # Time series analysis
│   ├── deformation.py    # Deformation algorithms
│   ├── arrays.py         # Array utilities
│   ├── preprocessing.py  # SAR preprocessing functions
│   ├── visualization.py  # Plotting and visualization
│   └── config.py         # Configuration management
├── notebooks/            # Jupyter notebooks
│   ├── 01_exploration.ipynb
│   ├── 02_timeseries_analysis.ipynb
│   └── 03_results_visualization.ipynb
├── tests/                # Unit tests
├── outputs/              # Generated figures and results
├── docs/                 # Documentation
├── requirements.txt      # Dependencies
└── README.md             # This file
```

## Methodology

### SAR-Based Processing Workflow

1. **Data Acquisition**: Download Sentinel-1 GRD products from Copernicus
2. **Preprocessing**: Co-registration, radiometric calibration, speckle filtering
3. **Data Cube Creation**: Stack images into 3D (x, y, time) arrays
4. **Time Series Analysis**: Compute temporal statistics and intensity change series
5. **Trend Detection**: Linear regression to estimate deformation velocity
6. **Classification**: Identify stable vs. deforming areas
7. **Visualization**: Generate maps, plots, and animations

### Key Algorithms

- **Linear Trend Detection**: RANSAC-based robust regression for velocity estimation
- **Change Point Detection**: Threshold-based anomaly detection in temporal derivatives
- **Temporal Stability Analysis**: Coefficient of variation as proxy for coherence
- **Multi-Temporal Analysis**: Simplified small baseline approach for cumulative changes
- **Hotspot Identification**: Connected component analysis for deformation zones

> **Note:** This implementation uses **SAR intensity** changes, not interferometric phase. For true InSAR displacement measurements, extend to SLC processing.

## Data Sources

- **SAR Data**: [Copernicus Data Space Ecosystem](https://dataspace.copernicus.eu/) - Sentinel-1 GRD/SLC
- **DEM**: [Copernicus DEM](https://spacedata.copernicus.eu/collections/copernicus-digital-elevation-model) or SRTM
- **Validation**: [European Ground Motion Service (EGMS)](https://land.copernicus.eu/pan-european/european-ground-motion-service)

## Future Extensions

This **SAR GRD-based** prototype can be extended to:

- **InSAR Processing**: Transition to SLC products for phase-based interferometry with MintPy or ISCE2
- **Atmospheric Correction**: Integration with weather models for InSAR phase corrections
- **Machine Learning**: Deep learning for deformation pattern classification
- **Real-time Monitoring**: Automated processing pipeline for operational use
- **Multi-Sensor Fusion**: Combine with optical imagery or other SAR sensors

## References

1. Ferretti, A., Prati, C., & Rocca, F. (2001). Permanent scatterers in SAR interferometry. *IEEE TGRS*, 39(1), 8-20.
2. Berardino, P., et al. (2002). A new algorithm for surface deformation monitoring based on small baseline differential SAR interferograms. *IEEE TGRS*, 40(11), 2375-2383.
3. Colesanti, C., & Wasowski, J. (2006). Investigating landslides with space-borne Synthetic Aperture Radar (SAR) interferometry. *Engineering Geology*, 88(3-4), 173-199.
4. Handwerger, A. L., et al. (2019). A shift from drought to extreme rainfall drives a stable landslide to catastrophic failure. *Scientific Reports*, 9(1), 1569.

## License

MIT License - See LICENSE file for details

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## Contact

For questions or collaboration: [Your Contact Information]

#   S A R _ G S D  
 