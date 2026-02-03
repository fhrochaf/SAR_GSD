"""
Tensor-accelerated processing module for SAR time series analysis.

Performs linear regression, then applies multi-scale Gaussian smoothing
to the trend results - all operations done on GPU/CPU tensors via PyTorch.

Computes slope from DEM data.
"""

import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr

from .config import Config

def _build_gaussian_kernel(
    sigma: float,
    device: torch.device,
) -> torch.Tensor:
    """Build a 2-D Gaussian convolution kernel for the given sigma."""
    kernel_size = int(2 * np.ceil(3 * sigma) + 1)
    x = torch.arange(
        -kernel_size // 2 + 1,
        kernel_size // 2 + 1,
        dtype=torch.float32,
        device=device,
    )
    xg, yg = torch.meshgrid(x, x, indexing="ij")
    kernel = torch.exp(-(xg ** 2 + yg ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    return kernel.view(1, 1, kernel_size, kernel_size)


def tensor_temporal_trend(
    ts_tensor: torch.Tensor,
    time_values: np.ndarray,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    
    if device is None:
        device = get_device(verbose=False)

    ts_tensor = ts_tensor.to(device)
    n_times = ts_tensor.shape[0]

    # Convert time_values to years from first acquisition
    from datetime import datetime
    time_objects = [datetime.fromisoformat(str(t)[:10]) for t in time_values]
    time_years = np.array([(t - time_objects[0]).days / 365.25 for t in time_objects])
    
    # Use actual time in years instead of indices
    t = torch.from_numpy(time_years).float().to(device).view(-1, 1, 1)
    t_mean = t.mean()
    y_mean = ts_tensor.mean(dim=0)

    numerator = ((ts_tensor - y_mean) * (t - t_mean)).sum(dim=0)
    denominator = ((t - t_mean) ** 2).sum()

    slope = numerator / denominator  # This is already in units/year
    intercept = y_mean - slope * t_mean

    y_pred = slope * t + intercept
    ss_res = ((ts_tensor - y_pred) ** 2).sum(dim=0)
    ss_tot = ((ts_tensor - y_mean) ** 2).sum(dim=0)

    r_squared = 1 - ss_res / (ss_tot + 1e-8)

    # No need to annualize - slope is already per year
    trend_annual = slope

    if verbose:
        print(f"  Trend range: {trend_annual.min().item():.6f} to {trend_annual.max().item():.6f} /year")
        print(f"  Mean R²: {r_squared.mean().item():.4f}")

    return trend_annual, r_squared


def tensor_smooth_spatial(
    spatial_tensor: torch.Tensor,
    scales: Optional[List[float]] = None,
    weights: Optional[List[float]] = None,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> torch.Tensor:
    """
    Apply multi-scale Gaussian smoothing to a 2-D spatial tensor.

    Each scale produces a spatially smoothed copy; the final result is
    a weighted combination of all scales.

    Args:
        spatial_tensor: Tensor of shape (height, width).
        scales: List of Gaussian sigma values (default [1.5, 2.5]).
        weights: Per-scale weights for the combination (must sum to 1).
                 Default [0.4, 0.6].
        device: Torch device. If None, uses get_device().
        verbose: Print progress messages.

    Returns:
        Smoothed tensor of the same shape as *spatial_tensor*.
    """
    if scales is None:
        scales = Config.GAUSSIAN_SIGMAS
    if weights is None:
        weights = Config.GAUSSIAN_WEIGHTS

    if len(scales) != len(weights):
        raise ValueError("scales and weights must have the same length")

    if device is None:
        device = get_device(verbose=verbose)

    spatial_tensor = spatial_tensor.to(device)
    w = torch.tensor(weights, device=device, dtype=torch.float32)

    # Add batch and channel dimensions for conv2d: (1, 1, H, W)
    input_tensor = spatial_tensor.unsqueeze(0).unsqueeze(0)

    smoothed_scales: list[torch.Tensor] = []

    for sigma in scales:
        kernel = _build_gaussian_kernel(sigma, device)
        pad = kernel.shape[-1] // 2
        smoothed = F.conv2d(input_tensor, kernel, padding=pad)
        # Remove batch and channel dims: (H, W)
        smoothed_scales.append(smoothed.squeeze(0).squeeze(0))

    combined = sum(w[i] * smoothed_scales[i] for i in range(len(scales)))

    if verbose:
        print(f"  Multi-scale Gaussian smoothing applied (scales={scales}, weights={weights})")

    return combined


def process_sar_timeseries_tensor(
    sar_da: xr.DataArray,
    threshold: Optional[float] = None,
    scales: Optional[List[float]] = None,
    weights: Optional[List[float]] = None,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Tensor-accelerated SAR processing pipeline with REORDERED steps.

    Steps:
      1. Extract backscatter, replace NaNs, move to device.
      2. Pixel-wise linear regression (trend + R²) on RAW data.
      3. Multi-scale Gaussian smoothing on TREND and R² maps.
      4. Change detection masks (positive / negative).

    Args:
        sar_da: xarray DataArray with dims (time, y, x).
        threshold: Change detection threshold. If None, uses
                   ``Config.CHANGE_THRESHOLD``.
        weights: Combination weights for each scale.
        scales: Gaussian sigma values for multi-scale smoothing.
        device: Torch device. If None, auto-detected.
        verbose: Print progress messages.

    Returns:
        Dictionary with keys:
        - ``trend``: 2-D smoothed annualised trend (y, x).
        - ``r_squared``: 2-D smoothed R² map (y, x).
        - ``trend_raw``: 2-D raw (unsmoothed) trend (y, x).
        - ``r_squared_raw``: 2-D raw (unsmoothed) R² map (y, x).
        - ``positive_mask``: Boolean 2-D array of positive changes.
        - ``negative_mask``: Boolean 2-D array of negative changes.
        - ``processing_time``: Wall-clock seconds for the full pipeline.
    """
    if threshold is None:
        threshold = Config.CHANGE_THRESHOLD

    if device is None:
        device = get_device(verbose=verbose)

    if verbose:
        print("=" * 60)
        print("TENSOR-ACCELERATED SAR PROCESSING PIPELINE")
        print("=" * 60)

    start_total = time.time()

    # Get datacube values
    ts_array = sar_da.values

    # --- 1. Prepare tensor ---------------------------------------------------
    if verbose:
        print("\n[1/3] Preparing tensor...")

    ts_clean = np.nan_to_num(ts_array, nan=0.0).astype(np.float32)
    ts_tensor = torch.from_numpy(ts_clean).to(device)

    if verbose:
        print(f"  Tensor shape: {tuple(ts_tensor.shape)}  device: {device}")

    # --- 2. Linear regression on RAW data ------------------------------------
    if verbose:
        print("\n[2/3] Computing tensor linear regression on raw data...")

    t0 = time.time()
    trend_annual, r_squared = tensor_temporal_trend(
        ts_tensor,
        time_values=sar_da.time.values,
        device=device,
        verbose=verbose,
    )
    if verbose:
        print(f"  Regression completed in {time.time() - t0:.2f} s")

    # Keep raw versions
    trend_raw = trend_annual.clone()
    r_squared_raw = r_squared.clone()

    # --- 3. Multi-scale Gaussian smoothing on TREND and R² -------------------
    if verbose:
        print("\n[3/3] Applying multi-scale Gaussian smoothing to trend and R²...")

    t0 = time.time()
    trend_smoothed = tensor_smooth_spatial(
        trend_annual,
        scales=scales,
        weights=weights,
        device=device,
        verbose=verbose,
    )
    r_squared_smoothed = tensor_smooth_spatial(
        r_squared,
        scales=scales,
        weights=weights,
        device=device,
        verbose=False,  # Don't print twice
    )
    if verbose:
        print(f"  Smoothing completed in {time.time() - t0:.2f} s")

    # --- 4. Convert to numpy -------------------------------------------------
    trend_numpy = trend_smoothed.cpu().numpy()
    r_squared_numpy = r_squared_smoothed.cpu().numpy()
    trend_raw_numpy = trend_raw.cpu().numpy()
    r_squared_raw_numpy = r_squared_raw.cpu().numpy()

    # --- 5. Change detection --------------------------------------------------
    # R² is not a p-value; pass a dummy all-zero array so that every pixel
    # passes the p < 0.05 gate inside detect_changes.  Users who need
    # proper p-values should fall back to processing.py.
    dummy_pvalues = np.zeros_like(trend_numpy)
    positive_mask, negative_mask = detect_changes(
        trend_numpy, dummy_pvalues, threshold=threshold, verbose=verbose,
    )

    total_time = time.time() - start_total
    if verbose:
        print(f"\nTotal tensor processing time: {total_time:.2f} s")

    return {
        "trend": trend_numpy,
        "r_squared": r_squared_numpy,
        "trend_raw": trend_raw_numpy,
        "r_squared_raw": r_squared_raw_numpy,
        "positive_mask": positive_mask,
        "negative_mask": negative_mask,
        "processing_time": total_time,
    }


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


def calculate_slope(dem_da: xr.DataArray,
                    path_slope: str = None,
                    mode: str = 'deg'):

    if dem_da.rio.crs.is_geographic:
        raise ValueError("DEM must be projected (units in meters).")

    if path_slope is None:
        path_slope = Config.get_output_path(Config.SLOPE_FILENAME)

    dem = dem_da.values.astype(np.float32)

    res = dem_da.rio.resolution()[0]

    nodata = dem_da.rio.nodata
    if nodata is not None:
        dem = np.where(dem == nodata, np.nan, dem)

    dz_dy, dz_dx = np.gradient(dem, res)

    grad = np.sqrt(dz_dx**2 + dz_dy**2)

    if mode == 'perc':
        slope = grad * 100
    elif mode == 'deg':
        slope = np.degrees(np.arctan(grad))
    else:
        raise ValueError("mode must be 'deg' or 'perc'")

    slope_da = dem_da.copy(data=slope)
    slope_da.name = 'slope'

    slope_da.rio.to_raster(path_slope, dtype='float32')

    return slope_da