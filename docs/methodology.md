# SAR Ground Deformation Detection - Methodology

## Overview

This document describes the **SAR-based methodology** for detecting ground deformation using Sentinel-1 GRD time series analysis. This approach uses **intensity changes** rather than interferometric phase.

## Scientific Background

### SAR and InSAR Principles

**Synthetic Aperture Radar (SAR)** is an active remote sensing technique that provides all-weather, day-night imaging capability. SAR measures the backscattered intensity of radar signals.

**Interferometric SAR (InSAR)** extends SAR by comparing the phase difference between images acquired at different times, enabling millimeter-scale displacement measurements. InSAR is widely used for:

- Landslide monitoring
- Subsidence detection
- Volcanic deformation
- Earthquake displacement mapping
- Infrastructure monitoring

### Time Series Approaches

Multi-temporal SAR/InSAR analysis uses stacks of images to improve detection:

1. **Persistent Scatterer Interferometry (PSI)** (Ferretti et al., 2001)
   - Identifies stable point targets
   - Suitable for urban areas
   - Requires high coherence

2. **Small Baseline Subset (SBAS)** (Berardino et al., 2002)
   - Uses short temporal baselines
   - Better for vegetated areas
   - More spatial coverage

## Implementation Approach

### SAR GRD-Based Processing

This implementation uses **Ground Range Detected (GRD)** products for **SAR intensity analysis**:

**GRD Approach:**
- Uses **backscatter intensity** changes over time
- Detects surface changes through temporal analysis
- No interferometric phase processing
- Suitable for rapid change detection and prototyping

**GRD Advantages:**
- Faster processing (no interferogram generation)
- Smaller file sizes (~1GB vs 5GB for SLC)
- Suitable for intensity-based change detection
- Good for educational/prototyping purposes
- Identifies areas of interest for detailed InSAR analysis

**GRD Limitations:**
- No phase information (not true InSAR)
- Cannot measure precise mm-scale displacement
- Intensity changes are qualitative indicators
- Less accurate than phase-based InSAR methods

**Future InSAR Extension:**
For scientifically rigorous phase-based displacement analysis, extend to **Single Look Complex (SLC)** products with tools like MintPy or ISCE2.

## Processing Workflow

### 1. Data Acquisition

**Source:** Copernicus Data Space Ecosystem  
**Product:** Sentinel-1 GRD IW (Interferometric Wide swath)  
**Polarization:** VV (vertical transmit, vertical receive)

```python
from insar.download import SentinelDownloader

downloader = SentinelDownloader()
products_df, files = downloader.download_by_area_and_dates(
    bbox=[-8.7, 40.5, -8.3, 40.8],
    start_date="2023-01-01",
    end_date="2023-12-31",
    product_type="GRD"
)
```

### 2. Preprocessing

**Steps:**
- Co-registration to common grid
- Radiometric calibration (DN to sigma0)
- Speckle filtering (Lee filter)
- Terrain correction (if DEM available)

### 3. Data Cube Creation

Stack 2D images into 3D (x, y, time) data cube using xarray:

```python
from insar.cubes import create_datacube_from_files

datacube = create_datacube_from_files(
    file_paths=files,
    band_name="intensity"
)
```

### 4. Time Series Analysis

#### 4.1 Intensity Change Calculation

Compute pixel-wise **intensity changes** relative to reference (proxy for surface changes):

```
change(t) = intensity(t) - reference
```

Where reference is typically the temporal median.

> **Note:** This measures backscatter intensity changes, not true displacement. For phase-based displacement, use SLC products with InSAR processing.

#### 4.2 Linear Trend Detection

Use robust regression (RANSAC) to estimate deformation velocity:

```
velocity = slope × 365.25  # Convert to mm/year
```

**Algorithm:**
1. Convert time to numeric (days since first acquisition)
2. For each pixel, fit linear model: `y = mx + b`
3. Use RANSAC to reject outliers
4. Calculate velocity, p-value, and R²

```python
from insar.timeseries import detect_linear_trends

velocity, pvalue, rsquared = detect_linear_trends(
    datacube,
    confidence_level=0.95,
    robust=True
)
```

#### 4.3 Change Point Detection

Identify abrupt deformation events using temporal derivatives:

```
anomaly = |d/dt - mean(d/dt)| > threshold × std(d/dt)
```

#### 4.4 Temporal Stability Analysis

For SAR GRD data, use coefficient of variation as proxy for temporal stability (analogous to InSAR coherence):

```
stability = 1 / (1 + CV)
where CV = std / mean
```

> **Note:** This is a **pseudo-coherence** metric. True coherence requires SLC phase data.

### 5. Deformation Classification

Classify pixels into categories based on velocity:

| Class | Velocity Range (mm/year) |
|-------|---------------------------|
| Stable | -5 to +5 |
| Slow subsidence | -10 to -5 |
| Moderate subsidence | -30 to -10 |
| Fast subsidence | < -30 |
| Uplift | > +5 |

### 6. Hotspot Identification

Use connected component analysis to identify contiguous deformation zones:

1. Create binary mask: `|velocity| > threshold`
2. Label connected components
3. Filter by minimum size
4. Calculate statistics for each hotspot

```python
from insar.deformation import identify_deformation_hotspots

hotspots, stats = identify_deformation_hotspots(
    velocity,
    threshold=20.0,  # mm/year
    min_size=10      # pixels
)
```

### 7. Landslide Susceptibility

Combine multiple factors:

```
susceptibility = w1 × velocity_norm + w2 × (1 - coherence) + w3 × slope_norm
```

Where:
- `velocity_norm`: Normalized deformation velocity
- `coherence`: Temporal stability
- `slope_norm`: Terrain slope (if DEM available)

## Validation

### Quality Metrics

1. **Temporal Coherence**: Measure of pixel stability (0-1, higher is better)
2. **R² Value**: Goodness of fit for linear trends
3. **P-value**: Statistical significance of trends

### External Validation

Compare results with:
- European Ground Motion Service (EGMS) products
- Field observations
- Landslide inventories
- Other InSAR studies

## Limitations

### Current SAR GRD Implementation

1. **No Phase Information**: Uses intensity changes, not interferometric phase
2. **Qualitative Measurements**: Changes are relative indicators, not absolute displacement
3. **No Atmospheric Correction**: Atmospheric effects not separated (less critical for intensity)
4. **Geometric Distortions**: Layover and shadow in mountainous terrain
5. **Temporal Sampling**: Limited by Sentinel-1 revisit time (6-12 days)
6. **Sensitivity**: Less sensitive than InSAR phase measurements

### Mitigation Strategies

1. Use robust statistics (RANSAC, median) to reduce noise
2. Filter temporally unstable pixels
3. Combine ascending and descending orbits for better coverage
4. Integrate with auxiliary data (DEM, geology, optical imagery)
5. Use results to identify areas for detailed InSAR analysis

## Future Enhancements

### Transition to InSAR (SLC-Based Processing)

For rigorous **phase-based displacement** analysis, extend to InSAR:

1. **Download SLC Products**: Switch from GRD to Single Look Complex data
2. **Interferogram Generation**: Compute phase differences between image pairs
3. **Phase Unwrapping**: Convert wrapped phase to displacement
4. **Atmospheric Correction**: Remove tropospheric and ionospheric delays
5. **Geocoding**: Project to geographic coordinates
6. **Time Series Inversion**: Apply PSI or SBAS algorithms

**Recommended InSAR Tools:**
- **MintPy**: Time series InSAR analysis
- **ISCE2**: Complete SAR/InSAR processing
- **SNAP**: ESA's processing platform with interferometry module

### Machine Learning

Apply deep learning for:
- SAR/InSAR deformation pattern classification
- Anomaly detection in time series
- Predictive modeling for landslide forecasting

## References

1. Ferretti, A., Prati, C., & Rocca, F. (2001). Permanent scatterers in SAR interferometry. *IEEE Transactions on Geoscience and Remote Sensing*, 39(1), 8-20.

2. Berardino, P., Fornaro, G., Lanari, R., & Sansosti, E. (2002). A new algorithm for surface deformation monitoring based on small baseline differential SAR interferograms. *IEEE Transactions on Geoscience and Remote Sensing*, 40(11), 2375-2383.

3. Colesanti, C., & Wasowski, J. (2006). Investigating landslides with space-borne Synthetic Aperture Radar (SAR) interferometry. *Engineering Geology*, 88(3-4), 173-199.

4. Handwerger, A. L., Huang, M. H., Fielding, E. J., Booth, A. M., & Bürgmann, R. (2019). A shift from drought to extreme rainfall drives a stable landslide to catastrophic failure. *Scientific Reports*, 9(1), 1569.

5. Crosetto, M., Monserrat, O., Cuevas-González, M., Devanthéry, N., & Crippa, B. (2016). Persistent scatterer interferometry: A review. *ISPRS Journal of Photogrammetry and Remote Sensing*, 115, 78-89.
