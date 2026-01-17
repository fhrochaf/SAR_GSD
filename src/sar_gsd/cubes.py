"""
Xarray data cubes module for managing SAR time series.

This module provides functionality to create and manipulate 3D (x, y, time) 
data cubes from Sentinel-1 imagery using xarray.
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
from datetime import datetime
import zipfile
import tempfile
import shutil

import numpy as np
import xarray as xr
import rioxarray as rxr
import rasterio
from rasterio.enums import Resampling
import pandas as pd

from .config import Config

logger = logging.getLogger(__name__)


def _extract_safe_zip(zip_path: Path, extract_dir: Optional[Path] = None) -> Path:
    """
    Extract a Sentinel-1 SAFE zip file and return path to the .SAFE directory.

    Args:
        zip_path: Path to .SAFE.zip file
        extract_dir: Directory to extract to (if None, uses temp directory)

    Returns:
        Path to extracted .SAFE directory
    """
    if not zip_path.suffix == '.zip':
        raise ValueError(f"File {zip_path} is not a zip file")

    # Use provided directory or create temp directory
    if extract_dir is None:
        extract_dir = Path(tempfile.mkdtemp())
    else:
        extract_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Extracting {zip_path.name} to {extract_dir}")

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    # Find the .SAFE directory
    safe_dirs = list(extract_dir.glob('*.SAFE'))
    if not safe_dirs:
        raise ValueError(f"No .SAFE directory found in {zip_path}")

    return safe_dirs[0]


def _find_tiff_in_safe(safe_dir: Path, polarization: str = 'vv') -> Optional[Path]:
    """
    Find the measurement TIFF file within a SAFE directory.

    Args:
        safe_dir: Path to .SAFE directory
        polarization: Polarization to use ('vv' or 'vh')

    Returns:
        Path to TIFF file or None if not found
    """
    # Sentinel-1 GRD structure: <SAFE>/measurement/*-<pol>-*.tiff
    measurement_dir = safe_dir / 'measurement'

    if not measurement_dir.exists():
        logger.warning(f"No measurement directory found in {safe_dir}")
        return None

    # Look for polarization-specific TIFF
    pattern = f'*-{polarization.lower()}-*.tiff'
    tiff_files = list(measurement_dir.glob(pattern))

    if not tiff_files:
        # Try alternative pattern
        pattern = f'*{polarization.upper()}*.tiff'
        tiff_files = list(measurement_dir.glob(pattern))

    if not tiff_files:
        logger.warning(f"No TIFF file with polarization {polarization} found in {measurement_dir}")
        return None

    return tiff_files[0]


def create_datacube_from_files(
    file_paths: List[Path],
    dates: Optional[List[datetime]] = None,
    band_name: str = "intensity",
    chunks: Optional[Dict] = None,
    unzip: bool = False,
    polarization: str = 'vv',
    extract_dir: Optional[Path] = None,
    cleanup: bool = True
) -> xr.DataArray:
    """
    Create a 3D data cube from multiple raster files.

    Args:
        file_paths: List of paths to raster files (GeoTIFF, etc.) or .SAFE.zip files
        dates: List of acquisition dates (if None, extracted from filenames)
        band_name: Name for the data variable
        chunks: Dask chunking specification for lazy loading
        unzip: If True, automatically extract .SAFE.zip files and read TIFFs from them
        polarization: Polarization to use when reading from SAFE files ('vv' or 'vh')
        extract_dir: Directory to extract zip files to (if None, uses temp directory)
        cleanup: If True, delete extracted files after creating datacube (only for temp dirs)

    Returns:
        xarray DataArray with dimensions (time, y, x)
    """
    if not file_paths:
        raise ValueError("No files provided")

    logger.info(f"Creating datacube from {len(file_paths)} files")

    # Track extracted directories for cleanup
    extracted_dirs = []
    temp_dir_used = extract_dir is None

    try:
        # Process file paths - extract zips if needed
        processed_paths = []

        for file_path in file_paths:
            file_path = Path(file_path)

            # Check if file is a zip and unzip is enabled
            if unzip and file_path.suffix == '.zip':
                logger.info(f"Processing zip file: {file_path.name}")

                # Extract the SAFE directory
                safe_dir = _extract_safe_zip(file_path, extract_dir)
                extracted_dirs.append(safe_dir.parent if temp_dir_used else safe_dir)

                # Find the TIFF file
                tiff_path = _find_tiff_in_safe(safe_dir, polarization)

                if tiff_path is None:
                    logger.warning(f"Could not find TIFF in {safe_dir.name}, skipping")
                    continue

                processed_paths.append(tiff_path)
            else:
                # Use file path as-is (assumes it's already a raster)
                processed_paths.append(file_path)

        if not processed_paths:
            raise ValueError("No valid raster files found after processing")

        logger.info(f"Processing {len(processed_paths)} raster files")

        # Extract dates if not provided
        if dates is None:
            dates = [_extract_date_from_filename(f) for f in file_paths]

        if len(dates) != len(file_paths):
            # Adjust dates list if some files were skipped
            dates = dates[:len(processed_paths)]

        # Create list to hold all arrays
        arrays = []

        for file_path in processed_paths:
            da = rxr.open_rasterio(file_path, chunks=chunks)

            # Take first band if multi-band
            if 'band' in da.dims:
                da = da.isel(band=0)

            arrays.append(da)

        # Check if arrays have matching spatial dimensions
        shapes = [arr.shape for arr in arrays]
        if len(set(shapes)) > 1:
            logger.info(f"Arrays have different shapes: {shapes}")
            logger.info("Reprojecting all arrays to match the first array's grid")

            # Use first array as reference grid
            reference = arrays[0]
            reprojected_arrays = [reference]

            # Reproject remaining arrays to match reference
            for i, arr in enumerate(arrays[1:], 1):
                logger.info(f"Reprojecting array {i+1}/{len(arrays)}")
                reprojected = arr.rio.reproject_match(reference)
                reprojected_arrays.append(reprojected)

            arrays = reprojected_arrays

        # Stack along time dimension
        datacube = xr.concat(arrays, dim='time')

        # Assign time coordinates
        datacube = datacube.assign_coords(time=pd.DatetimeIndex(dates))

        # Rename to meaningful variable name
        datacube.name = band_name

        # Add metadata
        datacube.attrs['description'] = f'Sentinel-1 {band_name} time series'
        datacube.attrs['n_scenes'] = len(processed_paths)
        datacube.attrs['start_date'] = str(dates[0])
        datacube.attrs['end_date'] = str(dates[-1])
        datacube.attrs['polarization'] = polarization if unzip else 'N/A'

        logger.info(f"Created datacube with shape: {datacube.shape}")
        logger.info(f"Time range: {dates[0]} to {dates[-1]}")

        return datacube

    finally:
        # Cleanup extracted files if requested and temp directory was used
        if cleanup and temp_dir_used and extracted_dirs:
            for extract_path in extracted_dirs:
                if extract_path.exists():
                    logger.info(f"Cleaning up extracted files: {extract_path}")
                    shutil.rmtree(extract_path, ignore_errors=True)


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
    complevel: int = 4,
) -> None:
    """
    Save datacube to NetCDF file.

    Args:
        datacube: Datacube to save
        output_path: Output file path (.nc)
        compression: Compression algorithm
        complevel: Compression level (1-9)
    """
    import time

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

    # Try to save
    try:
        # Use a temporary file first, then rename (atomic operation)
        temp_path = output_path.with_suffix('.nc.tmp')

        # Save to temporary file with engine specification for better compatibility
        datacube.to_netcdf(temp_path, encoding=encoding, engine='netcdf4')

        logger.debug(f"Saved temporary file {temp_path}.")

        # Remove target file if it exists
        if output_path.exists():
            output_path.unlink()

        logger.debug(f"Removed target file {output_path}.")

        # Rename temp file to final name
        temp_path.rename(output_path)

        logger.info(f"Saved datacube to {output_path}")
        logger.info(f"File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
        
    except Exception as e:
        logger.error(f"Failed to save datacube to {output_path}: {e}")
        
        raise e
        


def load_datacube(file_path: Path, chunks: Optional[Dict] = None) -> xr.DataArray:
    """
    Load datacube from NetCDF file.

    Args:
        file_path: Path to NetCDF file
        chunks: Dask chunking for lazy loading

    Returns:
        Loaded datacube
    """
    ds = rxr.open_rasterio(file_path)

    logger.info(f"Loaded datacube from {file_path}")
    logger.info(f"Dataset sizes: {ds.sizes}")

    return ds


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

