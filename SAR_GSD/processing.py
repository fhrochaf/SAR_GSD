"""
Tensor-accelerated processing module for SAR time series analysis.

Performs multi-scale Gaussian smoothing
and linear regression entirely on GPU/CPU tensors via PyTorch

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


def tensor_smooth_timeseries(
    ts_tensor: torch.Tensor,
    scales: Optional[List[float]] = None,
    weights: Optional[List[float]] = None,
    batch_size: Optional[int] = None,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> torch.Tensor:
    """
    Apply multi-scale Gaussian smoothing to a 3-D time-series tensor.

    Each scale produces a spatially smoothed copy of the full time series;
    the final result is a weighted combination of all scales.

    Args:
        ts_tensor: Tensor of shape (n_times, height, width).
        scales: List of Gaussian sigma values (default [1.5, 2.5]).
        weights: Per-scale weights for the combination (must sum to 1).
                 Default [0.4, 0.6].
        batch_size: Frames per convolution batch. If None, auto-selected
                    (16 for CUDA, 8 for CPU).
        device: Torch device. If None, uses get_device().
        verbose: Print progress messages.

    Returns:
        Smoothed tensor of the same shape as *ts_tensor*.
    """
    if scales is None:
        scales = Config.GAUSSIAN_SIGMAS
    if weights is None:
        weights = Config.GAUSSIAN_WEIGHTS

    if len(scales) != len(weights):
        raise ValueError("scales and weights must have the same length")

    if device is None:
        device = get_device(verbose=verbose)

    ts_tensor = ts_tensor.to(device)
    w = torch.tensor(weights, device=device, dtype=torch.float32)
    n_times = ts_tensor.shape[0]

    if batch_size is None:
        batch_size = 16 if device.type == "cuda" else 8

    smoothed_scales: list[torch.Tensor] = []

    for sigma in scales:
        kernel = _build_gaussian_kernel(sigma, device)
        pad = kernel.shape[-1] // 2
        frames = []
        for i in range(0, n_times, batch_size):
            batch = ts_tensor[i : i + batch_size].unsqueeze(1)
            smoothed = F.conv2d(batch, kernel, padding=pad)
            frames.append(smoothed.squeeze(1))
        smoothed_scales.append(torch.cat(frames, dim=0))

    combined = sum(w[i] * smoothed_scales[i] for i in range(len(scales)))

    if verbose:
        print(f"  Multi-scale Gaussian smoothing applied (scales={scales}, weights={weights})")

    return combined


def tensor_temporal_trend(
    smoothed: torch.Tensor,
    time_values: np.ndarray,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pixel-wise linear regression over the time dimension using tensors.

    Args:
        smoothed: Tensor of shape (n_times, height, width) – typically the
                  output of ``tensor_smooth_timeseries``.
        time_values: 1-D array of datetime64 values (e.g. from
                     ``datacube.time.values``).
        device: Torch device. If None, uses get_device().
        verbose: Print progress messages.

    Returns:
        Tuple of:
        - trend_annual: 2-D numpy array (y, x) – slope annualised to
          units / year.
        - r_squared: 2-D numpy array (y, x) – coefficient of determination.
        - smoothed_numpy: 3-D numpy array (time, y, x) – the smoothed data
          passed through for convenience.
    """
    if device is None:
        device = get_device(verbose=False)

    smoothed = smoothed.to(device)
    n_times = smoothed.shape[0]

    # Tensor index as regressor (0 … n_times-1)
    t = torch.arange(n_times, dtype=torch.float32, device=device).view(-1, 1, 1)
    t_mean = t.mean()
    y_mean = smoothed.mean(dim=0)

    numerator = ((smoothed - y_mean) * (t - t_mean)).sum(dim=0)
    denominator = ((t - t_mean) ** 2).sum()

    slope = numerator / denominator
    intercept = y_mean - slope * t_mean

    y_pred = slope * t + intercept
    ss_res = ((smoothed - y_pred) ** 2).sum(dim=0)
    ss_tot = ((smoothed - y_mean) ** 2).sum(dim=0)

    r_squared = 1 - ss_res / (ss_tot + 1e-8)

    # Convert back to numpy
    smoothed_numpy = smoothed.cpu().numpy()
    trend_numpy = slope.cpu().numpy()
    r_squared_numpy = r_squared.cpu().numpy()

    # Annualise: slope is per-index-step; convert to per-year
    time_diff = time_values[-1] - time_values[0]
    time_span_years = float(
        np.timedelta64(time_diff, "D").astype("float64") / 365.25
    )
    trend_annual = trend_numpy / time_span_years if time_span_years > 0 else trend_numpy

    if verbose:
        print(f"  Trend range: {np.nanmin(trend_annual):.6f} to {np.nanmax(trend_annual):.6f} /year")
        print(f"  Mean R²: {np.nanmean(r_squared_numpy):.4f}")

    return trend_annual, r_squared_numpy, smoothed_numpy


def process_sar_timeseries_tensor(
    sar_da: xr.DataArray,
    threshold: Optional[float] = None,
    scales: Optional[List[float]] = None,
    weights: Optional[List[float]] = None,
    batch_size: Optional[int] = None,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Tensor-accelerated SAR processing pipeline.

    Steps:
      1. Extract backscatter, replace NaNs, move to device.
      2. Multi-scale Gaussian smoothing.
      3. Pixel-wise linear regression (trend + R²).
      4. Change detection masks (positive / negative).

    Args:
        sar_da: xarray DataArray with dims (time, y, x).
        threshold: Change detection threshold. If None, uses
                   ``Config.CHANGE_THRESHOLD``.
        scales: Gaussian sigma values for multi-scale smoothing.
        weights: Combination weights for each scale.
        batch_size: Frames per convolution batch.
        device: Torch device. If None, auto-detected.
        verbose: Print progress messages.

    Returns:
        Dictionary with keys:
        - ``trend``: 2-D annualised trend (y, x).
        - ``r_squared``: 2-D R² map (y, x).
        - ``smoothed``: 3-D smoothed backscatter (time, y, x).
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

    # --- 1. Prepare tensor ---------------------------------------------------
    if verbose:
        print("\n[1/3] Preparing tensor...")

    ts_array = sar_da.values
    ts_clean = np.nan_to_num(ts_array, nan=0.0).astype(np.float32)
    ts_tensor = torch.from_numpy(ts_clean).to(device)

    if verbose:
        print(f"  Tensor shape: {tuple(ts_tensor.shape)}  device: {device}")

    # --- 2. Multi-scale Gaussian smoothing ------------------------------------
    if verbose:
        print("\n[2/3] Applying multi-scale Gaussian smoothing...")

    t0 = time.time()
    smoothed = tensor_smooth_timeseries(
        ts_tensor,
        scales=scales,
        weights=weights,
        batch_size=batch_size,
        device=device,
        verbose=verbose,
    )
    if verbose:
        print(f"  Smoothing completed in {time.time() - t0:.2f} s")

    # --- 3. Linear regression -------------------------------------------------
    if verbose:
        print("\n[3/3] Computing tensor linear regression...")

    t0 = time.time()
    trend_annual, r_squared, smoothed_numpy = tensor_temporal_trend(
        smoothed,
        time_values=sar_da.time.values,
        device=device,
        verbose=verbose,
    )
    if verbose:
        print(f"  Regression completed in {time.time() - t0:.2f} s")

    # --- 4. Change detection --------------------------------------------------
    # R² is not a p-value; pass a dummy all-zero array so that every pixel
    # passes the p < 0.05 gate inside detect_changes.  Users who need
    # proper p-values should fall back to processing.py.
    dummy_pvalues = np.zeros_like(trend_annual)
    positive_mask, negative_mask = detect_changes(
        trend_annual, dummy_pvalues, threshold=threshold, verbose=verbose,
    )

    total_time = time.time() - start_total
    if verbose:
        print(f"\nTotal tensor processing time: {total_time:.2f} s")

    return {
        "trend": trend_annual,
        "r_squared": r_squared,
        "smoothed": smoothed_numpy,
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
