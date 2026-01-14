"""
Deformation-specific algorithms for landslide and erosion detection.

This module provides higher-level functions for classifying and analyzing
ground deformation patterns.
"""

import logging
from typing import Tuple, Optional, Dict

import numpy as np
import xarray as xr
from scipy.ndimage import label, binary_dilation
from sklearn.cluster import DBSCAN
import pandas as pd

from .config import Config

logger = logging.getLogger(__name__)


def calculate_velocity_map(
    datacube: xr.DataArray,
    method: str = "linear",
    **kwargs
) -> xr.DataArray:
    """
    Generate deformation velocity map.
    
    Args:
        datacube: Input time series
        method: 'linear' or 'polynomial'
        **kwargs: Additional parameters for trend detection
        
    Returns:
        Velocity map in mm/year
    """
    from .timeseries import detect_linear_trends
    
    velocity, pvalue, rsquared = detect_linear_trends(datacube, **kwargs)
    
    # Mask insignificant pixels
    significance_threshold = kwargs.get('significance_threshold', 0.05)
    velocity_masked = velocity.where(pvalue < significance_threshold)
    
    logger.info(f"Velocity map: {np.nanmean(velocity_masked):.2f} ± {np.nanstd(velocity_masked):.2f} mm/year")
    
    return velocity_masked


def classify_deformation_zones(
    velocity_map: xr.DataArray,
    thresholds: Optional[Dict[str, float]] = None
) -> xr.DataArray:
    """
    Classify pixels into deformation categories.
    
    Args:
        velocity_map: Deformation velocity in mm/year
        thresholds: Dictionary of thresholds for classification
        
    Returns:
        Classification map (0=stable, 1=slow, 2=moderate, 3=fast, -1=uplift)
    """
    if thresholds is None:
        thresholds = {
            'stable': Config.DEFORMATION_THRESHOLD_MM,
            'slow': 10.0,
            'moderate': 30.0,
            'fast': 50.0
        }
    
    classification = xr.zeros_like(velocity_map, dtype=int)
    
    # Subsidence (negative velocity)
    classification = xr.where(
        velocity_map < -thresholds['fast'],
        3,  # Fast subsidence
        classification
    )
    classification = xr.where(
        (velocity_map < -thresholds['moderate']) & (velocity_map >= -thresholds['fast']),
        2,  # Moderate subsidence
        classification
    )
    classification = xr.where(
        (velocity_map < -thresholds['slow']) & (velocity_map >= -thresholds['moderate']),
        1,  # Slow subsidence
        classification
    )
    
    # Uplift (positive velocity)
    classification = xr.where(
        velocity_map > thresholds['stable'],
        -1,  # Uplift
        classification
    )
    
    classification.name = "deformation_class"
    classification.attrs['classes'] = {
        0: 'stable',
        1: 'slow_subsidence',
        2: 'moderate_subsidence',
        3: 'fast_subsidence',
        -1: 'uplift'
    }
    classification.attrs['thresholds_mm_year'] = thresholds
    
    # Copy spatial reference
    if hasattr(velocity_map, 'rio'):
        classification = classification.rio.write_crs(velocity_map.rio.crs)
    
    logger.info("Deformation classification complete")
    
    return classification


def landslide_susceptibility(
    velocity_map: xr.DataArray,
    coherence_map: Optional[xr.DataArray] = None,
    slope_map: Optional[xr.DataArray] = None
) -> xr.DataArray:
    """
    Calculate landslide susceptibility based on deformation patterns.
    
    Args:
        velocity_map: Deformation velocity
        coherence_map: Temporal coherence (optional)
        slope_map: Terrain slope in degrees (optional)
        
    Returns:
        Susceptibility score (0-1, higher is more susceptible)
    """
    logger.info("Calculating landslide susceptibility...")
    
    # Normalize velocity to 0-1 (higher subsidence = higher risk)
    velocity_norm = np.abs(velocity_map) / (np.nanmax(np.abs(velocity_map)) + 1e-10)
    velocity_norm = xr.where(velocity_norm > 1, 1, velocity_norm)
    
    susceptibility = velocity_norm
    
    # Incorporate coherence (low coherence = higher uncertainty = higher risk)
    if coherence_map is not None:
        coherence_risk = 1 - coherence_map
        susceptibility = (susceptibility + coherence_risk) / 2
    
    # Incorporate slope (steeper slopes = higher risk)
    if slope_map is not None:
        slope_norm = slope_map / 90.0  # Normalize to 0-1
        susceptibility = (susceptibility * 0.6 + slope_norm * 0.4)
    
    susceptibility.name = "landslide_susceptibility"
    susceptibility.attrs['range'] = '0-1 (higher is more susceptible)'
    
    # Copy spatial reference
    if hasattr(velocity_map, 'rio'):
        susceptibility = susceptibility.rio.write_crs(velocity_map.rio.crs)
    
    logger.info(f"Mean susceptibility: {susceptibility.mean().values:.3f}")
    
    return susceptibility


def temporal_clustering(
    datacube: xr.DataArray,
    n_clusters: int = 5,
    sample_fraction: float = 0.1
) -> Tuple[xr.DataArray, np.ndarray]:
    """
    Cluster pixels by temporal deformation behavior.
    
    Args:
        datacube: Input time series
        n_clusters: Number of clusters (if using K-means)
        sample_fraction: Fraction of pixels to sample for clustering
        
    Returns:
        Tuple of (cluster map, cluster centers)
    """
    from sklearn.cluster import KMeans
    
    logger.info(f"Clustering temporal patterns (n_clusters={n_clusters})...")
    
    # Reshape datacube to (n_pixels, n_times)
    n_times = datacube.shape[0]
    ny, nx = datacube.shape[1:]
    
    data_2d = datacube.values.reshape(n_times, -1).T
    
    # Remove pixels with too many NaNs
    valid_mask = np.isnan(data_2d).sum(axis=1) < (n_times * 0.3)
    valid_data = data_2d[valid_mask]
    
    # Fill remaining NaNs with temporal mean
    col_mean = np.nanmean(valid_data, axis=0)
    inds = np.where(np.isnan(valid_data))
    valid_data[inds] = np.take(col_mean, inds[1])
    
    # Sample for efficiency
    n_samples = int(len(valid_data) * sample_fraction)
    sample_idx = np.random.choice(len(valid_data), n_samples, replace=False)
    sample_data = valid_data[sample_idx]
    
    # Cluster
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(sample_data)
    
    # Predict all pixels
    labels = kmeans.predict(valid_data)
    
    # Reshape back to spatial dimensions
    cluster_map = np.full(ny * nx, -1, dtype=int)
    cluster_map[valid_mask] = labels
    cluster_map = cluster_map.reshape(ny, nx)
    
    # Create DataArray
    cluster_da = xr.DataArray(
        cluster_map,
        coords={'y': datacube.y, 'x': datacube.x},
        dims=['y', 'x'],
        name='clusters'
    )
    
    cluster_da.attrs['n_clusters'] = n_clusters
    cluster_da.attrs['method'] = 'K-means'
    
    # Copy spatial reference
    if hasattr(datacube, 'rio'):
        cluster_da = cluster_da.rio.write_crs(datacube.rio.crs)
    
    logger.info(f"Clustering complete. Found {n_clusters} temporal patterns")
    
    return cluster_da, kmeans.cluster_centers_


def identify_deformation_hotspots(
    velocity_map: xr.DataArray,
    threshold: float = 20.0,
    min_size: int = 10
) -> Tuple[xr.DataArray, pd.DataFrame]:
    """
    Identify contiguous deformation hotspots.
    
    Args:
        velocity_map: Deformation velocity map
        threshold: Minimum velocity threshold (mm/year)
        min_size: Minimum hotspot size in pixels
        
    Returns:
        Tuple of (labeled hotspot map, hotspot statistics DataFrame)
    """
    logger.info(f"Identifying deformation hotspots (threshold={threshold} mm/year)...")
    
    # Create binary mask
    hotspot_mask = np.abs(velocity_map.values) > threshold
    
    # Label connected components
    labeled_array, num_features = label(hotspot_mask)
    
    # Filter by size
    hotspot_stats = []
    filtered_labels = np.zeros_like(labeled_array)
    label_counter = 1
    
    for i in range(1, num_features + 1):
        component_mask = labeled_array == i
        size = component_mask.sum()
        
        if size >= min_size:
            filtered_labels[component_mask] = label_counter
            
            # Calculate statistics
            mean_velocity = velocity_map.values[component_mask].mean()
            max_velocity = velocity_map.values[component_mask].min()  # Most negative
            
            hotspot_stats.append({
                'hotspot_id': label_counter,
                'size_pixels': size,
                'mean_velocity_mm_year': mean_velocity,
                'max_velocity_mm_year': max_velocity,
            })
            
            label_counter += 1
    
    # Create DataArray
    hotspots_da = xr.DataArray(
        filtered_labels,
        coords={'y': velocity_map.y, 'x': velocity_map.x},
        dims=['y', 'x'],
        name='hotspots'
    )
    
    hotspots_da.attrs['threshold_mm_year'] = threshold
    hotspots_da.attrs['min_size_pixels'] = min_size
    hotspots_da.attrs['n_hotspots'] = label_counter - 1
    
    # Copy spatial reference
    if hasattr(velocity_map, 'rio'):
        hotspots_da = hotspots_da.rio.write_crs(velocity_map.rio.crs)
    
    # Create DataFrame
    stats_df = pd.DataFrame(hotspot_stats)
    
    logger.info(f"Found {len(stats_df)} deformation hotspots")
    
    return hotspots_da, stats_df
