"""
Processing mod
ule for SAR time series analysis and change detection.

This module provides functions to:
- Compute temporal trends using linear regression
- Apply spatial smoothing filters
- Generate change detection masks
- Create georeferenced output products
"""

import numpy as np
import xarray as xr
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from datetime import datetime
from typing import Tuple, Optional
from scipy import stats
import time

from .config import Config

def get_device(verbose: bool = True) -> torch.device:
    """
    Automatically detect and return the best available device.

    Returns:
        torch.device: 'cuda' if GPU is available, otherwise 'cpu'
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')

        if verbose:
            print(f"PyTorch version: {torch.__version__}")
            print(f"CUDA available: {torch.cuda.is_available()}")
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU count: {torch.cuda.device_count()}")
            print(f"GPU detected: {torch.cuda.get_device_name(0)}")
            print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            
    else:
        device = torch.device('cpu')
        if verbose:
            print(f"No GPU detected, using CPU")

    return device

def compute_temporal_trend(
    datacube: xr.DataArray,
    log_transform: bool = True,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute temporal trend (rate of change) in SAR backscatter using linear regression.

    The trend is computed by fitting a linear regression model to the time series
    at each pixel. The slope (trend coefficient) represents the rate of change
    in backscatter intensity over time.

    Args:
        datacube: xarray DataArray with dimensions (time, y, x)
        log_transform: If True, applies log10 transformation before computing trend.
                       Recommended for SAR data to reduce multiplicative noise.
        verbose: Print progress messages (default: True)

    Returns:
        Tuple containing:
        - trend: 2D array (y, x) with trend coefficients (log-intensity/year or intensity/year)
        - p_values: 2D array (y, x) with p-values for the trend coefficients
        - time_array: 1D array with time coordinates in years from first acquisition

    Example:
        >>> trend, p_vals, time = compute_temporal_trend(datacube, log_transform=True)
        >>> significant = p_vals < 0.05
        >>> print(f"Significant pixels: {np.sum(significant)} / {significant.size}")
    """
    # Get time coordinates and convert to years from first acquisition
    time_coords = datacube.time.values
    time_objects = [datetime.fromisoformat(str(t)[:10]) for t in time_coords]
    time_array = np.array([(t - time_objects[0]).days / 365.25 for t in time_objects])

    if verbose:
        print(f"Time span: {time_array[0]:.2f} to {time_array[-1]:.2f} years")
        print(f"Number of acquisitions: {len(time_array)}")

    # Get datacube values
    cube_data = datacube.values

    # Apply log transformation if requested
    if log_transform:
        # Ensure no negative or zero values before log
        cube_data = cube_data.copy()
        cube_data[cube_data <= 0] = np.nan
        cube_data = np.log10(cube_data)

    # Reshape for vectorized linear regression
    # Shape: (n_time, n_pixels) where n_pixels = height * width
    n_time, n_y, n_x = cube_data.shape
    data_flat = cube_data.reshape(n_time, -1)

    # Perform linear regression: fit slope (trend) and intercept for each pixel
    # np.polyfit returns [slope, intercept] for degree=1
    coeffs = np.polyfit(time_array, data_flat, 1)
    trend_flat = coeffs[0]  # slopes
    intercept_flat = coeffs[1]  # intercepts

    # Compute p-values (vectorized)
    # Calculate predicted values
    y_pred = trend_flat[np.newaxis, :] * time_array[:, np.newaxis] + intercept_flat[np.newaxis, :]
    
    # Calculate residuals
    residuals = data_flat - y_pred
    
    # Degrees of freedom
    df = n_time - 2
    
    # Residual sum of squares
    rss = np.nansum(residuals**2, axis=0)
    
    # Standard error of the residuals
    mse = rss / df
    
    # Standard error of the slope
    # SE(slope) = sqrt(MSE / sum((x - x_mean)^2))
    x_mean = np.mean(time_array)
    ss_x = np.sum((time_array - x_mean)**2)
    se_slope = np.sqrt(mse / ss_x)
    
    # t-statistic
    t_stat = trend_flat / se_slope
    
    # p-value (two-tailed test)
    p_values_flat = 2 * (1 - stats.t.cdf(np.abs(t_stat), df))
    
    # Reshape back to 2D
    trend = trend_flat.reshape(n_y, n_x)
    p_values = p_values_flat.reshape(n_y, n_x)

    if verbose:
        units = "log-units/year" if log_transform else "intensity-units/year"
        print(f"Trend range: {np.nanmin(trend):.4f} to {np.nanmax(trend):.4f} {units}")
        
        # Report significance statistics
        valid_p = ~np.isnan(p_values)
        if np.any(valid_p):
            sig_05 = np.sum(p_values[valid_p] < 0.05)
            sig_01 = np.sum(p_values[valid_p] < 0.01)
            total = np.sum(valid_p)
            print(f"Significant trends (p < 0.05): {sig_05}/{total} ({100*sig_05/total:.1f}%)")
            print(f"Highly significant (p < 0.01): {sig_01}/{total} ({100*sig_01/total:.1f}%)")

    return trend, p_values, time_array


def apply_spatial_smoothing(
    array: np.ndarray,
    gaussian_sigma: Optional[float] = None,
    mean_filter_size: Optional[int] = None,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> np.ndarray:
    """
    Apply spatial smoothing to reduce noise and enhance coherent patterns.

    Two-step smoothing process:
    1. Gaussian filter - reduces speckle noise
    2. Mean filter (convolution) - further smooths boundaries

    Args:
        array: 2D numpy array to smooth
        gaussian_sigma: Sigma for Gaussian filter in pixels. If None, uses Config.GAUSSIAN_SIGMA.
        mean_filter_size: Size of mean filter kernel. If None, uses Config.MEAN_FILTER_SIZE.
        device: PyTorch device to use for smoothing. If None, uses get_device().
        verbose: Print progress messages (default: True)

    Returns:
        Smoothed 2D array

    Example:
        >>> trend_smooth = apply_spatial_smoothing(trend, gaussian_sigma=1.5, mean_filter_size=3)
    """
    if gaussian_sigma is None:
        gaussian_sigma = Config.GAUSSIAN_SIGMA

    if mean_filter_size is None:
        mean_filter_size = Config.MEAN_FILTER_SIZE

    if verbose:
        print("Applying spatial smoothing...")
        print(f"  Gaussian filter: sigma={gaussian_sigma} pixels")
        print(f"  Mean filter: {mean_filter_size}x{mean_filter_size} kernel")

    # Step 1: Gaussian smoothing
    smooth_array = gaussian_filter(array, sigma=gaussian_sigma)

    # Step 2: Mean filter using PyTorch for efficient convolution
    # Convert to tensor and add batch and channel dimensions
    if device is None:
        device = get_device() #Check if GPU is available
    array_tensor = torch.from_numpy(smooth_array).float()[None, None].to(device)

    # Create averaging kernel
    kernel_size = mean_filter_size
    kernel = torch.ones((1, 1, kernel_size, kernel_size), dtype=torch.float32) / (kernel_size ** 2)
    kernel = kernel.to(device)

    # Apply convolution with padding to maintain size
    padding = kernel_size // 2
    smooth_tensor = F.conv2d(array_tensor, kernel, padding=padding)

    # Convert back to numpy
    final_array = smooth_tensor.squeeze().cpu().numpy()

    if verbose:
        print(f"  Output range: {np.nanmin(final_array):.4f} to {np.nanmax(final_array):.4f}")

    return final_array
   

def detect_changes(
    trend: np.ndarray,
    pvalues: np.ndarray,
    threshold: Optional[float] = None,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate binary masks identifying areas with significant positive or negative trends.

    Args:
        trend: 2D trend array (e.g., from compute_temporal_trend)
        pvalues: 2D p-value array (e.g., from compute_temporal_trend)
        threshold: Change threshold. Pixels with |trend| > threshold are marked as changed.
                   If None, uses Config.CHANGE_THRESHOLD.
        verbose: Print statistics (default: True)

    Returns:
        Tuple containing:
        - positive_mask: Boolean array where trend > threshold
        - negative_mask: Boolean array where trend < -threshold

    Example:
        >>> pos_mask, neg_mask = detect_changes(trend, threshold=0.07)
        >>> print(f"Positive changes: {np.sum(pos_mask)} pixels")
    """
    if threshold is None:
        threshold = Config.CHANGE_THRESHOLD

    # Apply threshold to identify significant changes
    positive_mask = (trend > threshold) & (pvalues < Config.P_VALUE_THRESHOLD)
    negative_mask = (trend < -threshold) & (pvalues < Config.P_VALUE_THRESHOLD)

    if verbose:
        n_positive = np.sum(positive_mask)
        n_negative = np.sum(negative_mask)
        total_pixels = trend.size
        n_stable = total_pixels - n_positive - n_negative

        print(f"\nChange Detection Results:")
        print(f"  Threshold: ±{threshold} log-units/year")
        print(f"  Positive changes: {n_positive} pixels ({100*n_positive/total_pixels:.2f}%)")
        print(f"  Negative changes: {n_negative} pixels ({100*n_negative/total_pixels:.2f}%)")
        print(f"  Stable areas: {n_stable} pixels ({100*n_stable/total_pixels:.2f}%)")

    return positive_mask, negative_mask


def process_sar_timeseries(
    datacube: xr.DataArray,
    threshold: Optional[float] = None,
    gaussian_sigma: Optional[float] = None,
    mean_filter_size: Optional[int] = None,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> dict:
    """
    Complete processing pipeline: compute trends, apply smoothing, detect changes.

    This is a convenience function that chains together the main processing steps.

    Args:
        datacube: xarray DataArray with dimensions (time, y, x)
        threshold: Change detection threshold. If None, uses Config.CHANGE_THRESHOLD.
        gaussian_sigma: Sigma for Gaussian filter. If None, uses Config.GAUSSIAN_SIGMA.
        mean_filter_size: Mean filter size. If None, uses Config.MEAN_FILTER_SIZE.
        device: PyTorch device to use for smoothing. If None, uses get_device().
        verbose: Print progress messages (default: True)

    Returns:
        Dictionary containing:
        - 'trend_da': Trend as xarray DataArray
        - 'pvalues_da': P-values as xarray DataArray
        - 'positive_mask_da': Positive mask as xarray DataArray
        - 'negative_mask_da': Negative mask as xarray DataArray
        - 'time_array': Time coordinates in years

    Example:
        >>> results = process_sar_timeseries(datacube, threshold=0.07)
        >>> results['trend_da'].rio.to_raster("outputs/trend.tif")
        >>> results['positive_mask_da'].rio.to_raster("outputs/positive_mask.tif")
    """
    if threshold is None:
        threshold = Config.CHANGE_THRESHOLD

    if verbose:
        print("=" * 60)
        print("SAR TIME SERIES PROCESSING PIPELINE")
        print("=" * 60)

    # Step 1: Compute temporal trend
    if verbose:
        print("\n[1/3] Computing temporal trend...")
    start_time = time.time()
    trend_raw, p_values, time_array = compute_temporal_trend(datacube, log_transform=True, verbose=verbose)
    end_time = time.time()
    if verbose:
        print(f"Temporal trend computed in {end_time - start_time:.2f} seconds")

    # Step 2: Apply spatial smoothing
    if verbose:
        print("\n[2/3] Applying spatial smoothing...")
    start_time = time.time()
    trend = apply_spatial_smoothing(
        trend_raw, gaussian_sigma=gaussian_sigma, mean_filter_size=mean_filter_size, device=device, verbose=verbose
    )
    end_time = time.time()
    if verbose:
        print(f"Spatial smoothing applied in {end_time - start_time:.2f} seconds")

    # Step 3: Detect changes
    if verbose:
        print("\n[3/3] Detecting changes...")
    start_time = time.time()
    positive_mask, negative_mask = detect_changes(trend, p_values, threshold=threshold, verbose=verbose)
    end_time = time.time()
    if verbose:
        print(f"Change detection completed in {end_time - start_time:.2f} seconds")


    return {
        "trend": trend,
        "pvalues": p_values,
        "positive_mask": positive_mask,
        "negative_mask": negative_mask,
        "time_array": time_array,
    }


def calculate_slope(dem_da: xr.DataArray,
                    path_slope: str = None,
                    mode:str = 'deg'):
    """
    Calculate slope from DEM data.

    dem: Path to the DEM file
    path_slope: Path to the output slope file
    mode: 'perc' or 'deg'
    """

    if path_slope is None:
        path_slope = Config.get_output_path(Config.SLOPE_FILENAME)

    # Get the data array (first band)
    # Corrected: Use dem_da.values directly if dem_da is already 2D
    dem = dem_da.values

    # Get cell size from the transform
    dx = abs(dem_da.rio.resolution()[0])  # pixel width
    dy = abs(dem_da.rio.resolution()[1])  # pixel height

    # Calculate gradients using numpy gradient
    dz_dx, dz_dy = np.gradient(dem, dx, dy)

    if mode == 'perc':
        # Calculate slope in percentage
        slope = np.sqrt(dz_dx**2 + dz_dy**2) * 100
    elif mode == 'deg':
    # Calculate slope in degrees
        slope = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2)) * (180 / np.pi)

    # Create a new DataArray with the slope data
    slope_da = dem_da.copy()
    slope_da.values = slope # Assign the 2D slope array
    slope_da.name = 'slope'

    # Write output
    slope_da.squeeze().rio.to_raster(path_slope, dtype='float32')

    return slope_da