"""
Xarray data cubes module for managing SAR time series.

This module provides functionality to create and manipulate 3D (x, y, time) 
data cubes from Sentinel-1 imagery using xarray.
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
from datetime import datetime

import numpy as np
import xarray as xr
import rioxarray as rxr
import rasterio
from rasterio.enums import Resampling
import pandas as pd

from .config import Config

logger = logging.getLogger(__name__)


def create_datacube_from_files(
    file_paths: List[Path],
    dates: Optional[List[datetime]] = None,
    band_name: str = "intensity",
    chunks: Optional[Dict] = None
) -> xr.DataArray:
    """
    Create a 3D data cube from multiple raster files.
    
    Args:
        file_paths: List of paths to raster files (GeoTIFF, etc.)
        dates: List of acquisition dates (if None, extracted from filenames)
        band_name: Name for the data variable
        chunks: Dask chunking specification for lazy loading
        
    Returns:
        xarray DataArray with dimensions (time, y, x)
    """
    if not file_paths:
        raise ValueError("No files provided")
    
    logger.info(f"Creating datacube from {len(file_paths)} files")
    
    # Extract dates if not provided
    if dates is None:
        dates = [_extract_date_from_filename(f) for f in file_paths]
    
    if len(dates) != len(file_paths):
        raise ValueError("Number of dates must match number of files")
    
    # Load first file to get spatial dimensions and CRS
    first_da = rxr.open_rasterio(file_paths[0], chunks=chunks)
    
    # Create list to hold all arrays
    arrays = []
    
    for file_path in file_paths:
        da = rxr.open_rasterio(file_path, chunks=chunks)
        
        # Take first band if multi-band
        if 'band' in da.dims:
            da = da.isel(band=0)
        
        arrays.append(da)
    
    # Stack along time dimension
    datacube = xr.concat(arrays, dim='time')
    
    # Assign time coordinates
    datacube = datacube.assign_coords(time=pd.DatetimeIndex(dates))
    
    # Rename to meaningful variable name
    datacube.name = band_name
    
    # Add metadata
    datacube.attrs['description'] = f'Sentinel-1 {band_name} time series'
    datacube.attrs['n_scenes'] = len(file_paths)
    datacube.attrs['start_date'] = str(dates[0])
    datacube.attrs['end_date'] = str(dates[-1])
    
    logger.info(f"Created datacube with shape: {datacube.shape}")
    logger.info(f"Time range: {dates[0]} to {dates[-1]}")
    
    return datacube


def create_datacube_from_arrays(
    arrays: List[np.ndarray],
    dates: List[datetime],
    transform: rasterio.Affine,
    crs: str = "EPSG:4326",
    band_name: str = "intensity"
) -> xr.DataArray:
    """
    Create datacube from numpy arrays with geospatial metadata.
    
    Args:
        arrays: List of 2D numpy arrays
        dates: List of acquisition dates
        transform: Affine transform for georeferencing
        crs: Coordinate reference system
        band_name: Variable name
        
    Returns:
        xarray DataArray with spatial reference
    """
    if len(arrays) != len(dates):
        raise ValueError("Number of arrays must match number of dates")
    
    # Stack arrays
    stacked = np.stack(arrays, axis=0)
    
    # Get spatial dimensions
    height, width = arrays[0].shape
    
    # Create coordinates
    x_coords = np.arange(width) * transform.a + transform.c
    y_coords = np.arange(height) * transform.e + transform.f
    
    # Create DataArray
    datacube = xr.DataArray(
        stacked,
        dims=['time', 'y', 'x'],
        coords={
            'time': pd.DatetimeIndex(dates),
            'y': y_coords,
            'x': x_coords
        },
        name=band_name
    )
    
    # Add spatial reference
    datacube = datacube.rio.write_crs(crs)
    datacube = datacube.rio.write_transform(transform)
    
    return datacube


def spatial_subset(
    datacube: xr.DataArray,
    bbox: Tuple[float, float, float, float]
) -> xr.DataArray:
    """
    Extract spatial subset from datacube.
    
    Args:
        datacube: Input datacube
        bbox: Bounding box (min_x, min_y, max_x, max_y)
        
    Returns:
        Subset datacube
    """
    min_x, min_y, max_x, max_y = bbox
    
    subset = datacube.rio.clip_box(
        minx=min_x,
        miny=min_y,
        maxx=max_x,
        maxy=max_y
    )
    
    logger.info(f"Spatial subset: {subset.shape}")
    
    return subset


def temporal_subset(
    datacube: xr.DataArray,
    start_date: Union[str, datetime],
    end_date: Union[str, datetime]
) -> xr.DataArray:
    """
    Extract temporal subset from datacube.
    
    Args:
        datacube: Input datacube
        start_date: Start date
        end_date: End date
        
    Returns:
        Subset datacube
    """
    subset = datacube.sel(time=slice(start_date, end_date))
    
    logger.info(f"Temporal subset: {len(subset.time)} time steps")
    
    return subset


def save_datacube(
    datacube: xr.DataArray,
    output_path: Path,
    compression: str = "zlib",
    complevel: int = 4
) -> None:
    """
    Save datacube to NetCDF file.
    
    Args:
        datacube: Datacube to save
        output_path: Output file path (.nc)
        compression: Compression algorithm
        complevel: Compression level (1-9)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Encoding for compression
    encoding = {
        datacube.name: {
            'zlib': compression == 'zlib',
            'complevel': complevel,
            'dtype': 'float32'
        }
    }
    
    # Convert to Dataset if DataArray
    if isinstance(datacube, xr.DataArray):
        ds = datacube.to_dataset()
    else:
        ds = datacube
    
    ds.to_netcdf(output_path, encoding=encoding)
    
    logger.info(f"Saved datacube to {output_path}")
    logger.info(f"File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")


def load_datacube(file_path: Path, chunks: Optional[Dict] = None) -> xr.DataArray:
    """
    Load datacube from NetCDF file.
    
    Args:
        file_path: Path to NetCDF file
        chunks: Dask chunking for lazy loading
        
    Returns:
        Loaded datacube
    """
    ds = xr.open_dataset(file_path, chunks=chunks)
    
    # Get first data variable
    var_name = list(ds.data_vars)[0]
    datacube = ds[var_name]
    
    logger.info(f"Loaded datacube from {file_path}")
    logger.info(f"Shape: {datacube.shape}, Dims: {datacube.dims}")
    
    return datacube


def resample_datacube(
    datacube: xr.DataArray,
    target_resolution: float,
    resampling_method: Resampling = Resampling.bilinear
) -> xr.DataArray:
    """
    Resample datacube to different spatial resolution.
    
    Args:
        datacube: Input datacube
        target_resolution: Target resolution in units of CRS
        resampling_method: Resampling algorithm
        
    Returns:
        Resampled datacube
    """
    resampled = datacube.rio.reproject(
        datacube.rio.crs,
        resolution=target_resolution,
        resampling=resampling_method
    )
    
    logger.info(f"Resampled from {datacube.shape} to {resampled.shape}")
    
    return resampled


def compute_temporal_statistics(datacube: xr.DataArray) -> xr.Dataset:
    """
    Compute temporal statistics across datacube.
    
    Args:
        datacube: Input datacube
        
    Returns:
        Dataset with mean, std, min, max, median
    """
    stats = xr.Dataset({
        'mean': datacube.mean(dim='time'),
        'std': datacube.std(dim='time'),
        'min': datacube.min(dim='time'),
        'max': datacube.max(dim='time'),
        'median': datacube.median(dim='time'),
    })
    
    # Preserve spatial reference
    for var in stats.data_vars:
        if hasattr(datacube, 'rio'):
            stats[var] = stats[var].rio.write_crs(datacube.rio.crs)
    
    return stats


def _extract_date_from_filename(file_path: Path) -> datetime:
    """
    Extract acquisition date from Sentinel-1 filename.
    
    Sentinel-1 naming convention:
    S1A_IW_GRDH_1SDV_20230115T061234_...
                     ^^^^^^^^
    
    Args:
        file_path: Path to Sentinel-1 file
        
    Returns:
        Acquisition datetime
    """
    filename = file_path.stem
    
    # Try to extract date from Sentinel-1 naming convention
    parts = filename.split('_')
    for part in parts:
        if len(part) == 15 and part[0] == '2':  # Starts with year 20XX
            try:
                date_str = part[:8]  # YYYYMMDD
                return datetime.strptime(date_str, '%Y%m%d')
            except ValueError:
                continue
    
    # Fallback: try to find any 8-digit date
    import re
    date_match = re.search(r'(\d{8})', filename)
    if date_match:
        try:
            return datetime.strptime(date_match.group(1), '%Y%m%d')
        except ValueError:
            pass
    
    logger.warning(f"Could not extract date from {filename}, using current date")
    return datetime.now()


def create_cube():
    """Legacy function for backward compatibility."""
    logger.warning("create_cube() is deprecated, use create_datacube_from_files()")
    return None

