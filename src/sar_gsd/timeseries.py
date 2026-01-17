"""
Deformation time series analysis module.

This module provides functions for analyzing SAR time series to detect
ground deformation velocity using linear regression.
"""

import logging
from typing import Tuple

import numpy as np
import xarray as xr
from scipy import stats
from sklearn.linear_model import RANSACRegressor

from .config import Config

logger = logging.getLogger(__name__)


def detect_linear_trends(
    datacube: xr.DataArray,
    confidence_level: float = 0.95,
    robust: bool = True
) -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """
    Detect linear deformation trends using regression.

    Args:
        datacube: Input time series (time, y, x)
        confidence_level: Confidence level for significance testing
        robust: Use RANSAC for robust regression (recommended)

    Returns:
        Tuple of (velocity map, p-value map, r-squared map)
        - velocity: Linear deformation velocity in mm/year
        - pvalue: Statistical significance of the trend
        - rsquared: Goodness of fit (0-1)
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
