"""
Deformation time series analysis module.

This module provides functions for analyzing SAR time series to detect
ground deformation, including landslides, subsidence, and erosion.
"""

import logging
from typing import Optional, Tuple, Dict, List
from datetime import datetime

import numpy as np
import xarray as xr
from scipy import stats, signal
from scipy.ndimage import gaussian_filter
from sklearn.linear_model import LinearRegression, RANSACRegressor
import pandas as pd

from .config import Config

logger = logging.getLogger(__name__)


def calculate_temporal_mean(
    datacube: xr.DataArray,
    method: str = "median"
) -> xr.DataArray:
    """
    Calculate temporal reference baseline.
    
    Args:
        datacube: Input time series datacube
        method: Aggregation method ('mean', 'median', 'first')
        
    Returns:
        2D reference image
    """
    if method == "mean":
        reference = datacube.mean(dim='time')
    elif method == "median":
        reference = datacube.median(dim='time')
    elif method == "first":
        reference = datacube.isel(time=0)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    logger.info(f"Calculated temporal {method} as reference")
    
    return reference


def compute_displacement_series(
    datacube: xr.DataArray,
    reference: Optional[xr.DataArray] = None,
    method: str = "difference"
) -> xr.DataArray:
    """
    Compute pixel-wise displacement time series.
    
    For GRD data, this computes backscatter intensity changes rather than
    true phase-based displacement (which requires SLC data).
    
    Args:
        datacube: Input time series
        reference: Reference image (if None, uses temporal median)
        method: 'difference' or 'ratio'
        
    Returns:
        Displacement time series
    """
    if reference is None:
        reference = calculate_temporal_mean(datacube, method="median")
    
    if method == "difference":
        displacement = datacube - reference
    elif method == "ratio":
        # Avoid division by zero
        displacement = (datacube - reference) / (reference + 1e-10)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    displacement.name = "displacement"
    displacement.attrs['reference_method'] = method
    
    logger.info(f"Computed displacement series using {method}")
    
    return displacement


def detect_linear_trends(
    datacube: xr.DataArray,
    confidence_level: float = 0.95,
    robust: bool = True
) -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """
    Detect linear deformation trends using regression.
    
    Args:
        datacube: Input time series
        confidence_level: Confidence level for significance testing
        robust: Use RANSAC for robust regression
        
    Returns:
        Tuple of (velocity map, p-value map, r-squared map)
    """
    logger.info("Detecting linear trends...")
    
    # Get time coordinates as numeric (days since first acquisition)
    times = datacube.time.values
    time_numeric = (times - times[0]) / np.timedelta64(1, 'D')
    time_numeric = time_numeric.astype(float)
    
    # Prepare output arrays
    velocity = np.zeros(datacube.shape[1:])
    pvalue = np.ones(datacube.shape[1:])
    rsquared = np.zeros(datacube.shape[1:])
    
    # Iterate over spatial pixels
    ny, nx = datacube.shape[1:]
    
    for i in range(ny):
        for j in range(nx):
            pixel_series = datacube[:, i, j].values
            
            # Skip if too many NaN values
            if np.isnan(pixel_series).sum() > len(pixel_series) * 0.3:
                velocity[i, j] = np.nan
                pvalue[i, j] = np.nan
                rsquared[i, j] = np.nan
                continue
            
            # Remove NaN values
            valid_mask = ~np.isnan(pixel_series)
            X = time_numeric[valid_mask].reshape(-1, 1)
            y = pixel_series[valid_mask]
            
            if len(y) < 3:  # Need at least 3 points
                continue
            
            try:
                if robust:
                    # RANSAC for outlier rejection
                    model = RANSACRegressor(random_state=42)
                    model.fit(X, y)
                    slope = model.estimator_.coef_[0]
                    
                    # Calculate R-squared manually
                    y_pred = model.predict(X)
                    ss_res = np.sum((y - y_pred) ** 2)
                    ss_tot = np.sum((y - np.mean(y)) ** 2)
                    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                    
                    # Approximate p-value using scipy
                    _, p = stats.pearsonr(X.flatten(), y)
                else:
                    # Standard linear regression
                    slope, intercept, r, p, std_err = stats.linregress(X.flatten(), y)
                    r2 = r ** 2
                
                # Convert slope from units/day to mm/year (assuming input is in mm)
                velocity[i, j] = slope * 365.25
                pvalue[i, j] = p
                rsquared[i, j] = r2
                
            except Exception as e:
                logger.debug(f"Regression failed at pixel ({i}, {j}): {e}")
                continue
    
    # Create DataArrays with spatial coordinates
    coords = {'y': datacube.y, 'x': datacube.x}
    
    velocity_da = xr.DataArray(velocity, coords=coords, dims=['y', 'x'], name='velocity')
    pvalue_da = xr.DataArray(pvalue, coords=coords, dims=['y', 'x'], name='pvalue')
    rsquared_da = xr.DataArray(rsquared, coords=coords, dims=['y', 'x'], name='rsquared')
    
    # Add metadata
    velocity_da.attrs['units'] = 'mm/year'
    velocity_da.attrs['description'] = 'Linear deformation velocity'
    velocity_da.attrs['confidence_level'] = confidence_level
    
    # Copy spatial reference if available
    if hasattr(datacube, 'rio'):
        velocity_da = velocity_da.rio.write_crs(datacube.rio.crs)
        pvalue_da = pvalue_da.rio.write_crs(datacube.rio.crs)
        rsquared_da = rsquared_da.rio.write_crs(datacube.rio.crs)
    
    logger.info(f"Detected trends. Mean velocity: {np.nanmean(velocity):.2f} mm/year")
    
    return velocity_da, pvalue_da, rsquared_da


def identify_deformation_events(
    datacube: xr.DataArray,
    threshold: float = 2.0,
    min_duration: int = 2
) -> xr.DataArray:
    """
    Identify change points / deformation events in time series.
    
    Uses a simple threshold-based approach on temporal derivatives.
    
    Args:
        datacube: Input time series
        threshold: Threshold in standard deviations
        min_duration: Minimum duration of event in time steps
        
    Returns:
        Binary mask of deformation events (time, y, x)
    """
    logger.info("Identifying deformation events...")
    
    # Compute temporal derivative
    derivative = datacube.diff(dim='time')
    
    # Compute statistics
    mean_deriv = derivative.mean(dim='time')
    std_deriv = derivative.std(dim='time')
    
    # Identify anomalies (values beyond threshold * std)
    anomalies = np.abs(derivative - mean_deriv) > (threshold * std_deriv)
    
    # Apply minimum duration filter
    if min_duration > 1:
        # Simple rolling window approach
        anomalies_filtered = anomalies.rolling(time=min_duration, center=True).sum() >= min_duration
    else:
        anomalies_filtered = anomalies
    
    anomalies_filtered.name = "deformation_events"
    anomalies_filtered.attrs['threshold_std'] = threshold
    anomalies_filtered.attrs['min_duration'] = min_duration
    
    n_events = anomalies_filtered.sum().values
    logger.info(f"Identified {n_events} deformation event pixels")
    
    return anomalies_filtered


def coherence_analysis(
    datacube: xr.DataArray,
    window_size: int = 3
) -> xr.DataArray:
    """
    Assess temporal coherence/stability of pixels.
    
    For GRD data, this uses coefficient of variation as a proxy for coherence.
    True coherence requires SLC data.
    
    Args:
        datacube: Input time series
        window_size: Spatial window size for smoothing
        
    Returns:
        Coherence map (0-1, higher is more stable)
    """
    logger.info("Computing temporal coherence...")
    
    # Compute coefficient of variation (inverse of coherence)
    mean_val = datacube.mean(dim='time')
    std_val = datacube.std(dim='time')
    
    # Avoid division by zero
    cv = std_val / (mean_val + 1e-10)
    
    # Convert to coherence-like metric (0-1, higher is better)
    # Normalize and invert
    coherence = 1 / (1 + cv)
    
    # Spatial smoothing
    if window_size > 1:
        coherence_smooth = xr.apply_ufunc(
            gaussian_filter,
            coherence,
            kwargs={'sigma': window_size / 3},
            dask='parallelized'
        )
    else:
        coherence_smooth = coherence
    
    coherence_smooth.name = "coherence"
    coherence_smooth.attrs['description'] = 'Temporal stability (pseudo-coherence)'
    coherence_smooth.attrs['range'] = '0-1 (higher is more stable)'
    
    # Copy spatial reference
    if hasattr(datacube, 'rio'):
        coherence_smooth = coherence_smooth.rio.write_crs(datacube.rio.crs)
    
    logger.info(f"Mean coherence: {coherence_smooth.mean().values:.3f}")
    
    return coherence_smooth


def sbas_inspired_analysis(
    datacube: xr.DataArray,
    max_temporal_baseline: int = 48
) -> xr.DataArray:
    """
    Simplified Small Baseline Subset (SBAS) inspired analysis.
    
    This is a simplified version that computes short-baseline differences
    rather than full SBAS inversion (which requires SLC data).
    
    Args:
        datacube: Input time series
        max_temporal_baseline: Maximum temporal baseline in days
        
    Returns:
        Cumulative displacement estimate
    """
    logger.info(f"Running SBAS-inspired analysis (max baseline: {max_temporal_baseline} days)...")
    
    times = pd.DatetimeIndex(datacube.time.values)
    n_times = len(times)
    
    # Find pairs within temporal baseline
    pairs = []
    for i in range(n_times):
        for j in range(i + 1, n_times):
            delta_days = (times[j] - times[i]).days
            if delta_days <= max_temporal_baseline:
                pairs.append((i, j))
    
    logger.info(f"Found {len(pairs)} valid pairs")
    
    # Compute differences for each pair
    differences = []
    for i, j in pairs:
        diff = datacube.isel(time=j) - datacube.isel(time=i)
        differences.append(diff)
    
    # Average differences (simplified inversion)
    if differences:
        mean_diff = sum(differences) / len(differences)
    else:
        mean_diff = xr.zeros_like(datacube.isel(time=0))
    
    # Compute cumulative displacement
    cumulative = xr.concat(
        [datacube.isel(time=i) - datacube.isel(time=0) for i in range(n_times)],
        dim='time'
    )
    
    cumulative.name = "cumulative_displacement"
    cumulative.attrs['method'] = 'SBAS-inspired'
    cumulative.attrs['max_baseline_days'] = max_temporal_baseline
    cumulative.attrs['n_pairs'] = len(pairs)
    
    return cumulative


def calculate_timeseries():
    """Legacy function for backward compatibility."""
    logger.warning("calculate_timeseries() is deprecated, use compute_displacement_series()")
    return None

