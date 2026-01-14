"""
Array manipulation and statistical utilities for SAR data.
"""

import logging
from typing import Tuple, Optional

import numpy as np
from scipy import stats
from scipy.ndimage import uniform_filter, gaussian_filter, median_filter

logger = logging.getLogger(__name__)


def normalize_array(
    array: np.ndarray,
    method: str = "minmax",
    clip_percentiles: Optional[Tuple[float, float]] = None
) -> np.ndarray:
    """
    Normalize array values.
    
    Args:
        array: Input array
        method: 'minmax', 'zscore', or 'robust'
        clip_percentiles: Percentiles for clipping (e.g., (2, 98))
        
    Returns:
        Normalized array
    """
    arr = array.copy()
    
    # Clip outliers if requested
    if clip_percentiles:
        lower, upper = np.nanpercentile(arr, clip_percentiles)
        arr = np.clip(arr, lower, upper)
    
    if method == "minmax":
        min_val = np.nanmin(arr)
        max_val = np.nanmax(arr)
        normalized = (arr - min_val) / (max_val - min_val + 1e-10)
    
    elif method == "zscore":
        mean_val = np.nanmean(arr)
        std_val = np.nanstd(arr)
        normalized = (arr - mean_val) / (std_val + 1e-10)
    
    elif method == "robust":
        median_val = np.nanmedian(arr)
        mad = np.nanmedian(np.abs(arr - median_val))
        normalized = (arr - median_val) / (mad + 1e-10)
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized


def compute_statistics(array: np.ndarray) -> dict:
    """
    Calculate comprehensive array statistics.
    
    Args:
        array: Input array
        
    Returns:
        Dictionary of statistics
    """
    return {
        'mean': np.nanmean(array),
        'median': np.nanmedian(array),
        'std': np.nanstd(array),
        'min': np.nanmin(array),
        'max': np.nanmax(array),
        'p25': np.nanpercentile(array, 25),
        'p75': np.nanpercentile(array, 75),
        'p95': np.nanpercentile(array, 95),
        'p99': np.nanpercentile(array, 99),
        'n_valid': np.sum(~np.isnan(array)),
        'n_total': array.size,
    }


def detect_outliers(
    array: np.ndarray,
    method: str = "iqr",
    threshold: float = 3.0
) -> np.ndarray:
    """
    Detect outliers in array.
    
    Args:
        array: Input array
        method: 'iqr', 'zscore', or 'mad'
        threshold: Threshold for outlier detection
        
    Returns:
        Boolean mask (True = outlier)
    """
    if method == "iqr":
        q25, q75 = np.nanpercentile(array, [25, 75])
        iqr = q75 - q25
        lower = q25 - threshold * iqr
        upper = q75 + threshold * iqr
        outliers = (array < lower) | (array > upper)
    
    elif method == "zscore":
        z_scores = np.abs(stats.zscore(array, nan_policy='omit'))
        outliers = z_scores > threshold
    
    elif method == "mad":
        median = np.nanmedian(array)
        mad = np.nanmedian(np.abs(array - median))
        modified_z = 0.6745 * (array - median) / (mad + 1e-10)
        outliers = np.abs(modified_z) > threshold
    
    else:
        raise ValueError(f"Unknown outlier detection method: {method}")
    
    return outliers


def apply_spatial_filter(
    array: np.ndarray,
    filter_type: str = "gaussian",
    size: int = 3,
    **kwargs
) -> np.ndarray:
    """
    Apply spatial filter to array.
    
    Args:
        array: Input 2D array
        filter_type: 'gaussian', 'median', or 'uniform'
        size: Filter window size
        **kwargs: Additional filter parameters
        
    Returns:
        Filtered array
    """
    if filter_type == "gaussian":
        sigma = kwargs.get('sigma', size / 3)
        filtered = gaussian_filter(array, sigma=sigma)
    
    elif filter_type == "median":
        filtered = median_filter(array, size=size)
    
    elif filter_type == "uniform":
        filtered = uniform_filter(array, size=size)
    
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")
    
    logger.info(f"Applied {filter_type} filter (size={size})")
    
    return filtered


def calculate_gradient(array: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate spatial gradient.
    
    Args:
        array: Input 2D array
        
    Returns:
        Tuple of (gradient_y, gradient_x)
    """
    grad_y, grad_x = np.gradient(array)
    return grad_y, grad_x


def calculate_slope_aspect(
    dem: np.ndarray,
    pixel_size: float = 10.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate slope and aspect from DEM.
    
    Args:
        dem: Digital elevation model
        pixel_size: Pixel size in meters
        
    Returns:
        Tuple of (slope in degrees, aspect in degrees)
    """
    # Calculate gradients
    dz_dy, dz_dx = np.gradient(dem, pixel_size)
    
    # Slope in radians then degrees
    slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
    slope_deg = np.degrees(slope_rad)
    
    # Aspect in radians then degrees
    aspect_rad = np.arctan2(-dz_dy, dz_dx)
    aspect_deg = np.degrees(aspect_rad)
    aspect_deg = (aspect_deg + 360) % 360  # Convert to 0-360
    
    return slope_deg, aspect_deg
