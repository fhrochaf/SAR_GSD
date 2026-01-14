# Data Sources for InSAR Ground Deformation Detection

## Primary SAR Data

### Copernicus Data Space Ecosystem (Recommended)

**URL:** https://dataspace.copernicus.eu/

**Satellite:** Sentinel-1 (A & B constellation)  
**Products:** GRD (Ground Range Detected) and SLC (Single Look Complex)  
**Coverage:** Global  
**Revisit Time:** 6-12 days  
**Cost:** Free

**Registration:**
1. Go to https://dataspace.copernicus.eu/
2. Create account
3. Verify email
4. Use credentials in `.env` file

**Product Types:**
- **GRD (IW)**: Interferometric Wide swath, 10m resolution, multi-looked
- **SLC (IW)**: Single Look Complex, phase information, for interferometry

**API Access:**
- Catalog API: https://catalogue.dataspace.copernicus.eu/resto
- Python library: `sentinelsat`

## Pre-Processed Deformation Products

### European Ground Motion Service (EGMS)

**URL:** https://land.copernicus.eu/pan-european/european-ground-motion-service

**Coverage:** Europe  
**Product:** PSI-derived deformation velocities  
**Resolution:** Point measurements  
**Time Period:** 2015-2021 (Baseline), updated annually

**Use Cases:**
- Validation of your results
- Comparison with operational products
- Training data for machine learning

**Access:**
- Web viewer: https://egms.land.copernicus.eu/
- Download: Vector files (GeoPackage, Shapefile)

## Auxiliary Datasets

### Digital Elevation Models (DEM)

#### Copernicus DEM

**URL:** https://spacedata.copernicus.eu/collections/copernicus-digital-elevation-model

**Resolution:** 30m (GLO-30) and 90m (GLO-90)  
**Coverage:** Global (between 90°N and 90°S)  
**Vertical Accuracy:** < 4m (relative)

**Download:**
- Direct: https://portal.opentopography.org/
- Python: `elevation` library

### Vector Data

#### Landslide Inventories

**Global:**
- NASA Global Landslide Catalog: https://data.nasa.gov/Earth-Science/Global-Landslide-Catalog/h9d8-neg4

## Data Download Examples

### Sentinel-1 with sentinelsat

```python
from sentinelsat import SentinelAPI

api = SentinelAPI('username', 'password', 
                  'https://catalogue.dataspace.copernicus.eu/resto')

# Query products
products = api.query(
    area='POLYGON((-8.7 40.5, -8.3 40.5, -8.3 40.8, -8.7 40.8, -8.7 40.5))',
    date=('20230101', '20231231'),
    platformname='Sentinel-1',
    producttype='GRD'
)

# Download
api.download_all(products)
```

### DEM with elevation

```python
import elevation

# Download SRTM DEM
elevation.clip(
    bounds=(-8.7, 40.5, -8.3, 40.8),
    output='dem.tif',
    product='SRTM3'
)
```

## Citation

When using these data sources, please cite appropriately:

**Sentinel-1:**
> Copernicus Sentinel-1 data [Year], processed by ESA.

**EGMS:**
> European Ground Motion Service (2021). European Environment Agency.

**Copernicus DEM:**
> Copernicus DEM (2021). European Space Agency.

## Support

- **Copernicus Support:** https://forum.step.esa.int/
- **Sentinel-1 Documentation:** https://sentinels.copernicus.eu/web/sentinel/user-guides/sentinel-1-sar
- **EGMS User Manual:** https://land.copernicus.eu/user-corner/technical-library/egms-specification-and-user-manual
