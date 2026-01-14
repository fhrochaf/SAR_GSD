"""
Configuration management for SAR Ground Deformation Detection.

This module handles all configuration parameters, file paths, and credentials
for the SAR processing pipeline.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Central configuration class for SAR processing."""
    
    # Project paths
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    EXTERNAL_DATA_DIR = DATA_DIR / "external"
    OUTPUT_DIR = PROJECT_ROOT / "outputs"
    CACHE_DIR = PROJECT_ROOT / ".cache"
    
    # Copernicus Data Space credentials
    COPERNICUS_USERNAME = os.getenv("COPERNICUS_USERNAME")
    COPERNICUS_PASSWORD = os.getenv("COPERNICUS_PASSWORD")
    
    # Data acquisition parameters
    SENTINEL_PLATFORM = "Sentinel-1"
    PRODUCT_TYPE = "GRD"  # Ground Range Detected (can be changed to SLC later)
    SENSOR_MODE = "IW"  # Interferometric Wide swath
    POLARIZATION = "VV"  # Vertical-Vertical (or VH for cross-pol)
    
    # Processing parameters
    SPATIAL_RESOLUTION = 10  # meters (for GRD IW mode)
    TEMPORAL_BASELINE_DAYS = 12  # Sentinel-1 revisit time
    MIN_COHERENCE_THRESHOLD = 0.3  # Minimum coherence for valid pixels
    
    # Time series analysis
    REFERENCE_DATE_METHOD = "median"  # or "first", "mean"
    DEFORMATION_THRESHOLD_MM = 5.0  # Minimum detectable deformation in mm/year
    TREND_CONFIDENCE_LEVEL = 0.95  # For linear regression
    
    # Spatial filtering
    SPECKLE_FILTER_SIZE = 5  # pixels (Lee filter window)
    GAUSSIAN_SIGMA = 1.0  # For Gaussian smoothing
    
    # Visualization
    DPI = 300  # For saved figures
    COLORMAP_DEFORMATION = "RdYlBu_r"  # Red (subsidence) to Blue (uplift)
    COLORMAP_VELOCITY = "seismic"
    
    # Machine learning (optional)
    ML_BATCH_SIZE = 32
    ML_LEARNING_RATE = 0.001
    ML_EPOCHS = 50
    
    @classmethod
    def ensure_directories(cls) -> None:
        """Create necessary directories if they don't exist."""
        directories = [
            cls.DATA_DIR,
            cls.RAW_DATA_DIR,
            cls.PROCESSED_DATA_DIR,
            cls.EXTERNAL_DATA_DIR,
            cls.OUTPUT_DIR,
            cls.CACHE_DIR,
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def validate_credentials(cls) -> bool:
        """Check if Copernicus credentials are configured."""
        return bool(cls.COPERNICUS_USERNAME and cls.COPERNICUS_PASSWORD)
    
    @classmethod
    def get_study_area_config(cls, name: str) -> Optional[Dict[str, Any]]:
        """
        Get predefined study area configurations.
        
        Args:
            name: Study area name (e.g., 'test_area', 'portugal_north')
            
        Returns:
            Dictionary with bbox, dates, etc., or None if not found
        """
        study_areas = {
            "test_area": {
                "bbox": [-8.7, 40.5, -8.3, 40.8],  # [min_lon, min_lat, max_lon, max_lat]
                "name": "Test Area - Northern Portugal",
                "start_date": "2023-01-01",
                "end_date": "2023-12-31",
            },
            "portugal_north": {
                "bbox": [-8.8, 41.0, -8.0, 41.5],
                "name": "Northern Portugal - Douro Valley",
                "start_date": "2022-01-01",
                "end_date": "2023-12-31",
            },
        }
        return study_areas.get(name)
    
    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """Export configuration as dictionary."""
        return {
            "data_dirs": {
                "raw": str(cls.RAW_DATA_DIR),
                "processed": str(cls.PROCESSED_DATA_DIR),
                "external": str(cls.EXTERNAL_DATA_DIR),
                "output": str(cls.OUTPUT_DIR),
            },
            "sentinel": {
                "platform": cls.SENTINEL_PLATFORM,
                "product_type": cls.PRODUCT_TYPE,
                "sensor_mode": cls.SENSOR_MODE,
                "polarization": cls.POLARIZATION,
            },
            "processing": {
                "spatial_resolution": cls.SPATIAL_RESOLUTION,
                "temporal_baseline": cls.TEMPORAL_BASELINE_DAYS,
                "coherence_threshold": cls.MIN_COHERENCE_THRESHOLD,
            },
        }


# Initialize directories on import
Config.ensure_directories()
