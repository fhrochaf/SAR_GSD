"""
Data acquisition module for downloading Sentinel-1 SAR imagery.

This module provides functions to:
- Query available Sentinel-1 acquisition dates
- Download GRD images from Sentinel Hub
- Build georeferenced xarray datacubes
"""

import requests
import numpy as np
import xarray as xr
import rioxarray
from datetime import datetime, date
from typing import List, Tuple, Optional
import os
from rasterio.io import MemoryFile

from sentinelhub import (
    SHConfig,
    SentinelHubRequest,
    SentinelHubCatalog,
    DataCollection,
    BBox,
    CRS,
    bbox_to_dimensions,
    MimeType,
)

from .config import Config


def get_sentinel_hub_config() -> SHConfig:
    """
    Create and configure Sentinel Hub API client.

    Returns:
        Configured SHConfig object

    Raises:
        RuntimeError: If credentials are not properly configured
    """
    if not Config.validate_sentinel_hub_credentials():
        raise RuntimeError(
            "Sentinel Hub credentials not configured. "
            "Please set SENTINEL_HUB_CLIENT_ID and SENTINEL_HUB_CLIENT_SECRET in .env file."
        )

    config = SHConfig()
    config.sh_client_id = Config.SENTINEL_HUB_CLIENT_ID
    config.sh_client_secret = Config.SENTINEL_HUB_CLIENT_SECRET

    return config


def get_available_s1_dates(
    bbox: BBox, start_date: str, end_date: Optional[str] = None, config: Optional[SHConfig] = None
) -> List[str]:
    """
    Query Sentinel Hub catalog for available Sentinel-1 acquisition dates.

    Args:
        bbox: BBox object defining the area of interest
        start_date: Starting date for the search (ISO format string, e.g., '2020-01-01')
        end_date: Ending date for the search (ISO format string). If None, uses today's date.
        config: Sentinel Hub configuration. If None, uses default from Config.

    Returns:
        Sorted list of unique acquisition dates in YYYY-MM-DD format

    Example:
        >>> bbox = BBox([-42.85, -19.65, -42.48, -19.40], CRS.WGS84)
        >>> dates = get_available_s1_dates(bbox, "2020-01-01", "2020-12-31")
        >>> print(f"Found {len(dates)} acquisitions")
    """
    if config is None:
        config = get_sentinel_hub_config()

    if end_date is None:
        end_date = date.today().isoformat()

    # Initialize catalog
    catalog = SentinelHubCatalog(config=config)

    # Search the Sentinel-1 IW (Interferometric Wide swath) collection
    search = catalog.search(
        DataCollection.SENTINEL1_IW,
        bbox=bbox,
        time=(start_date, end_date),
        fields={"include": ["properties.datetime"], "exclude": []},
    )

    # Extract and deduplicate dates (truncate to day level)
    dates = sorted({item["properties"]["datetime"][:10] for item in search})

    return dates


def request_s1_image(
    date_str: str,
    bbox: BBox,
    size: Tuple[int, int],
    config: Optional[SHConfig] = None,
    polarization: str = "VV",
) -> np.ndarray:
    """
    Request a single Sentinel-1 GRD image for a specific date.

    Args:
        date_str: Acquisition date in YYYY-MM-DD format
        bbox: BBox object defining the spatial extent
        size: Tuple (width, height) in pixels
        config: Sentinel Hub configuration. If None, uses default from Config.
        polarization: SAR polarization ('VV', 'VH', 'HH', or 'HV'). Default is 'VV'.

    Returns:
        2D numpy array with backscatter values (linear scale, float32)

    Example:
        >>> bbox = BBox([-42.85, -19.65, -42.48, -19.40], CRS.WGS84)
        >>> size = (512, 512)
        >>> img = request_s1_image("2020-06-15", bbox, size)
        >>> print(f"Image shape: {img.shape}, dtype: {img.dtype}")
    """
    if config is None:
        config = get_sentinel_hub_config()

    # Validate polarization
    valid_pols = ["VV", "VH", "HH", "HV"]
    if polarization not in valid_pols:
        raise ValueError(f"Invalid polarization '{polarization}'. Must be one of {valid_pols}")

    # Evalscript defines what data to retrieve and how to process it
    evalscript = f"""
    //VERSION=3
    function setup() {{
        return {{ input: ["{polarization}"], output: {{ bands: 1, sampleType: "FLOAT32" }}}};
    }}
    function evaluatePixel(s) {{ return [s.{polarization}]; }}
    """

    # Build the request
    req = SentinelHubRequest(
        evalscript=evalscript,
        input_data=[
            SentinelHubRequest.input_data(
                DataCollection.SENTINEL1_IW,
                time_interval=(date_str, date_str),
                mosaicking_order="mostRecent",
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=bbox,
        size=size,
        config=config,
        )   



    # Execute request and return first (and only) image
    return req.get_data()[0]


def build_datacube(
    bbox_coords: List[float],
    start_date: str,
    end_date: Optional[str] = None,
    resolution: int = 20,
    max_dates: Optional[int] = None,
    polarization: str = "VV",
    normalize: bool = True,
    config: Optional[SHConfig] = None,
    verbose: bool = True,
) -> Tuple[xr.DataArray, List[str]]:
    """
    Build a georeferenced xarray datacube from Sentinel-1 time series.

    This function:
    1. Queries available Sentinel-1 dates in the specified time range
    2. Downloads images for each date
    3. Optionally normalizes each image by its median
    4. Stacks images into a 3D datacube with time, y, x dimensions
    5. Adds geospatial coordinates and CRS information

    Args:
        bbox_coords: Bounding box as [lon_min, lat_min, lon_max, lat_max]
        start_date: Starting date (ISO format, e.g., '2020-01-01')
        end_date: Ending date (ISO format). If None, uses today's date.
        resolution: Spatial resolution in meters (default: 20)
        max_dates: Maximum number of dates to process. If None or greater than available,
                   uses all dates. If less, subsamples evenly across time range.
        polarization: SAR polarization ('VV', 'VH', 'HH', or 'HV'). Default is 'VV'.
        normalize: If True, normalizes each image by its median (default: True)
        config: Sentinel Hub configuration. If None, uses default from Config.
        verbose: Print progress messages (default: True)

    Returns:
        Tuple containing:
        - datacube: xarray.DataArray with dimensions (time, y, x)
        - valid_dates: List of date strings that were successfully processed

    Raises:
        RuntimeError: If fewer than Config.MIN_ACQUISITIONS valid images are available

    Example:
        >>> bbox = [-42.85, -19.65, -42.48, -19.40]
        >>> datacube, dates = build_datacube(
        ...     bbox, "2020-01-01", "2020-12-31", resolution=20, max_dates=20
        ... )
        >>> print(f"Datacube shape: {datacube.shape}")
        >>> print(f"Date range: {dates[0]} to {dates[-1]}")
    """
    if config is None:
        config = get_sentinel_hub_config()

    # Create BBox object from coordinates
    bbox = BBox(bbox_coords, CRS.WGS84)

    # Calculate pixel dimensions based on resolution
    size = bbox_to_dimensions(bbox, resolution=resolution)

    if verbose:
        print(f"AOI Bounding Box: {bbox_coords}")
        print(f"Image size: {size[0]} x {size[1]} pixels")
        print(f"Resolution: {resolution} m")

    # Query available dates
    dates = get_available_s1_dates(bbox, start_date, end_date, config)

    if verbose:
        print(f"Total available dates: {len(dates)}")

    if len(dates) == 0:
        raise RuntimeError("No Sentinel-1 acquisitions found for the specified area and time range")

    # Subsample dates if max_dates is specified
    if max_dates is not None and max_dates < len(dates):
        step = max(1, len(dates) // max_dates)
        dates = dates[::step]
        if verbose:
            print(f"Subsampling to {len(dates)} dates")

    # Initialize lists for building the datacube
    cube_data = []  # Will store image arrays
    valid_dates = []  # Will store corresponding dates

    # Download and process each date
    for i, d in enumerate(dates):
        if verbose:
            print(f"Downloading {i+1}/{len(dates)}: {d}...", end=" ")

        try:
            # Request the image
            img = request_s1_image(d, bbox, size, config, polarization).astype(np.float32)

            # Remove invalid values (zero or negative backscatter)
            img[img <= 0] = np.nan

            # Skip images with no valid data
            if np.all(np.isnan(img)):
                if verbose:
                    print("No valid data, skipping.")
                continue

            # Optionally normalize by median
            if normalize:
                img = img / np.nanmedian(img)

            cube_data.append(img)
            valid_dates.append(d)

            if verbose:
                print("OK")

        except Exception as e:
            if verbose:
                print(f"Error: {e}")
            continue

    # Verify we have enough data
    if len(cube_data) < Config.MIN_ACQUISITIONS:
        raise RuntimeError(
            f"Not enough valid data. Only {len(cube_data)} images available "
            f"(minimum required: {Config.MIN_ACQUISITIONS})"
        )

    if verbose:
        print(f"\nSuccessfully downloaded {len(cube_data)} images")

    # Stack all images into a 3D numpy array (time, y, x)
    cube_array = np.stack(cube_data)

    # Convert date strings to datetime objects
    time_coords = [datetime.fromisoformat(d) for d in valid_dates]

    # Create spatial coordinates
    lon_min, lat_min, lon_max, lat_max = bbox_coords
    ny, nx = cube_array.shape[1:]

    # Generate x (longitude) and y (latitude) coordinate arrays
    x_coords = np.linspace(lon_min, lon_max, nx)
    y_coords = np.linspace(lat_max, lat_min, ny)  # Note: lat_max to lat_min (top to bottom)

    # Create xarray DataArray with dimensions and coordinates
    datacube = xr.DataArray(
        cube_array,
        dims=["time", "y", "x"],
        coords={
            "time": time_coords,
            "y": y_coords,
            "x": x_coords,
        },
        attrs={
            "description": f"Sentinel-1 {polarization} backscatter datacube"
            + (" (normalized)" if normalize else ""),
            "units": "dimensionless (normalized to median)" if normalize else "linear backscatter",
            "processing_date": datetime.now().isoformat(),
            "source": "Sentinel Hub",
            "resolution_m": resolution,
            "polarization": polarization,
            "start_date": valid_dates[0],
            "end_date": valid_dates[-1],
            "n_acquisitions": len(valid_dates),
        },
    )

    # Add CRS (Coordinate Reference System) information using rioxarray
    datacube = datacube.rio.write_crs("EPSG:4326")

    # Set spatial dimensions for rioxarray
    datacube = datacube.rio.set_spatial_dims(x_dim="x", y_dim="y")

    if verbose:
        print(f"Datacube shape: {datacube.shape}")
        print(f"Date range: {valid_dates[0]} to {valid_dates[-1]}")

    return datacube, valid_dates


def get_dem(bounds: List[float],
            key_opentopo: Optional[str] = Config.KEY_OPEN_TOPOGRAPHY,
            save_dem_raster: bool = True,
            output_dir: Optional[str] = None):
    """
    Get DEM data from OpenTopography.

    bounds: [xmin, ymin, xmax, ymax] bounding box
    key_opentopo: API key for OpenTopography
    save_dem_raster: Whether to save the DEM as a GeoTIFF
    output_dir: Output directory (defaults to ./outputs)
        
    Returns:
        tuple: (dem_data, dem_meta) or None if failed
    """
    
    # Setup output directory
    if output_dir is None:
        output_dir = str(Config.OUTPUT_DIR)
    
    # Attempt to retrieve data from OpenTopography 
    dem_data = _load_from_opentopo(bounds, key_opentopo)
    
    if dem_data is None:
        print("Failed to retrieve DEM from all sources.")
        return None
    
    # Print metadata
    crs = dem_data.rio.crs
    data_min, data_max = dem_data.min(), dem_data.max()
    print(f"CRS: {crs}")
    print(f"Data range: {data_min:.2f}m to {data_max:.2f}m")
    
    dem_data = dem_data.squeeze() #Squeeze the raster into shape (x,y)
    # Save if requested
    if save_dem_raster:
        file_path = os.path.join(output_dir, Config.DEM_FILENAME)
        """Save raster data to file."""
        print(f"Saving DEM to {file_path}")
        try:
            dem_data.rio.to_raster(file_path)
        except Exception as e:
            print(f"Failed to save file: {e}")
 
    return dem_data


def _load_from_opentopo(bounds, api_key):
    """Load DEM from OpenTopography API and return as rioxarray with target CRS and resolution."""
    print('Attempting to load DEM from OpenTopography.')
    try:
        west, south, east, north = map(str, bounds)
        params = {
            "demtype": "SRTMGL3",
            "south": south,
            "north": north,
            "west": west,
            "east": east,
            "outputFormat": "GTiff",
            "API_Key": api_key
        }
        
        url = 'https://portal.opentopography.org/API/globaldem'
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        with MemoryFile(response.content) as memfile:
            with memfile.open() as src:
                # Read directly as rioxarray
                dem_raster = rioxarray.open_rasterio(src, masked=True)
                # Load into memory to avoid issues when MemoryFile closes
                dem_raster = dem_raster.load()
               
        return dem_raster
    
    except Exception as e:
        print(f'Failed to load from OpenTopography: {e}')
        return None