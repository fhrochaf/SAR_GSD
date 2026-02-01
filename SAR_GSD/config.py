"""
Configuration management for SAR Ground Deformation Detection.

This module handles all configuration parameters, file paths, and credentials
for the SAR processing pipeline.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Central configuration class for SAR processing."""

    # ============================================================
    # PROJECT PATHS
    # ============================================================

    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    OUTPUT_DIR = PROJECT_ROOT / "outputs"
    CACHE_DIR = PROJECT_ROOT / ".cache"

    # ============================================================
    # API CREDENTIALS
    # ============================================================

    # Sentinel Hub API credentials
    SENTINEL_HUB_CLIENT_ID = os.getenv("SENTINEL_HUB_CLIENT_ID")
    SENTINEL_HUB_CLIENT_SECRET = os.getenv("SENTINEL_HUB_CLIENT_SECRET")
    SENTINEL_HUB_CONFIGURATION_ID = os.getenv("SENTINEL_HUB_CONFIGURATION_ID")


    # Open Topography Key
    KEY_OPEN_TOPOGRAPHY = os.getenv("KEY_OPEN_TOPOGRAPHY")

    # ============================================================
    # SENTINEL-1 DATA ACQUISITION PARAMETERS
    # ============================================================

    # Satellite and product configuration
    SENTINEL_PLATFORM = "Sentinel-1"
    PRODUCT_TYPE = "GRD"  # Ground Range Detected
    SENSOR_MODE = "IW"    # Interferometric Wide swath
    POLARIZATION = "VV"   # Vertical-Vertical polarization

    # Spatial resolution in meters
    DEFAULT_RESOLUTION = 20

    # Temporal parameters
    DEFAULT_START_DATE = "2017-01-01"
    MAX_DATES = 40  # Maximum number of acquisitions to process
    TEMPORAL_BASELINE_DAYS = 12  # Sentinel-1 revisit time

    # ============================================================
    # CHANGE DETECTION PARAMETERS
    # ============================================================

    # Change detection threshold (log-intensity change per year)
    CHANGE_THRESHOLD = 0.05
    P_VALUE_THRESHOLD = 0.05

    # Spatial filtering parameters
    GAUSSIAN_SIGMAS = [1.5, 2.5]  # Gaussian filter (in pixels) list for multi-scale smoothing
    GAUSSIAN_WEIGHTS = [0.8, 0.2]  # Weights for the Gaussian filters
    MEAN_FILTER_SIZE = 3  # Mean filter kernel size (3x3)

    # Minimum number of valid acquisitions required
    MIN_ACQUISITIONS = 3

    # ============================================================
    # AREA OF INTEREST (AOI) DEFINITIONS
    # ============================================================

    # Default AOI bounding box [lon_min, lat_min, lon_max, lat_max]
    DEFAULT_AOI_BBOX = [-42.85, -19.65, -42.48, -19.40]

    # ============================================================
    # OUTPUT AND EXPORT PARAMETERS
    # ============================================================

    # Output file names
    DATACUBE_FILENAME = "sar_datacube.nc"
    TREND_MAP_FILENAME = "sar_trend.tif"
    POSITIVE_MASK_FILENAME = "positive_change_mask.tif"
    NEGATIVE_MASK_FILENAME = "negative_change_mask.tif"
    INTENSITY_FIGURE_FILENAME = "sar_intensity.png"
    OVERLAY_FIGURE_FILENAME = "sar_change_overlay.png"
    GPKG_FILENAME = "sar_change_detection.gpkg"
    DEM_FILENAME = "dem.tif"
    SLOPE_FILENAME = "slope.tif"

    # Visualization parameters
    DPI = 150  # Resolution for saved figures
    FIGURE_SIZE = (10, 8)  # Default figure size in inches
    COLORMAP_INTENSITY = "gray"
    COLORMAP_DEFORMATION = "RdYlBu_r"  # Red (negative) to Blue (positive)

    # ============================================================
    # METHODS
    # ============================================================

    @classmethod
    def ensure_directories(cls) -> None:
        """Create necessary directories if they don't exist."""
        directories = [
            cls.DATA_DIR,
            cls.OUTPUT_DIR,
            cls.CACHE_DIR,
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    @classmethod
    def validate_sentinel_hub_credentials(cls) -> bool:
        """
        Check if Sentinel Hub credentials are configured.

        Returns:
            True if both CLIENT_ID and CLIENT_SECRET are set, False otherwise
        """
        return bool(cls.SENTINEL_HUB_CLIENT_ID and cls.SENTINEL_HUB_CLIENT_SECRET)

    @classmethod
    def get_output_path(cls, filename: str) -> Path:
        """
        Get full path for an output file.

        Args:
            filename: Name of the output file

        Returns:
            Full path to the output file
        """
        return cls.OUTPUT_DIR / filename

    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """
        Export configuration as dictionary.

        Returns:
            Dictionary containing key configuration parameters
        """
        return {
            "paths": {
                "project_root": str(cls.PROJECT_ROOT),
                "output": str(cls.OUTPUT_DIR),
                "cache": str(cls.CACHE_DIR),
            },
            "sentinel1": {
                "platform": cls.SENTINEL_PLATFORM,
                "product_type": cls.PRODUCT_TYPE,
                "sensor_mode": cls.SENSOR_MODE,
                "polarization": cls.POLARIZATION,
                "resolution": cls.DEFAULT_RESOLUTION,
            },
            "change_detection": {
                "threshold": cls.CHANGE_THRESHOLD,
                "gaussian_sigma": cls.GAUSSIAN_SIGMA,
                "mean_filter_size": cls.MEAN_FILTER_SIZE,
                "min_acquisitions": cls.MIN_ACQUISITIONS,
            },
            "credentials": {
                "sentinel_hub_configured": cls.validate_sentinel_hub_credentials(),
            },
        }


# Initialize directories on import
Config.ensure_directories()
