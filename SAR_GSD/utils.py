import folium
import geopandas as gpd
from rasterio.enums import Resampling
import matplotlib as mpl
from matplotlib.colors import rgb2hex
import matplotlib.colors as mcolors
import numpy as np
import branca.colormap as cm
import pandas as pd
import xarray as xr
import os
import subprocess
from typing import List, Tuple, Optional, Union
from .config import Config


def launch_google_earth(
    kml_paths: Union[str, List[str]],
    verbose: bool = True
) -> bool:
    """
    Attempt to launch Google Earth Pro with one or more KML files.

    Parameters:
    -----------
    kml_paths : str or list of str
        Path to a single KML file, or a list of paths to multiple KML files.
    verbose : bool
        Print status messages. Default is True.

    Returns:
    --------
    bool
        True if Google Earth was launched successfully, False otherwise.

    Example:
    --------
    >>> # Single KML file
    >>> success = launch_google_earth("outputs/changes.kml")

    >>> # Multiple KML files
    >>> success = launch_google_earth([
    ...     "outputs/changes.kml",
    ...     "outputs/transmission_lines.kml",
    ...     "outputs/sar_changes.kml"
    ... ])
    """
    # Normalize input to list
    if isinstance(kml_paths, str):
        kml_paths = [kml_paths]

    # Validate all paths exist
    for kml_path in kml_paths:
        if not os.path.exists(kml_path):
            if verbose:
                print(f"Warning: KML file not found: {kml_path}")

    # List of common Google Earth Pro installation paths
    google_earth_paths = [
        r"C:\Program Files\Google\Google Earth Pro\client\googleearth.exe",
        r"C:\Program Files (x86)\Google\Google Earth Pro\client\googleearth.exe",
        "/Applications/Google Earth Pro.app/Contents/MacOS/Google Earth Pro",  # macOS
        "/usr/bin/google-earth-pro",  # Linux
    ]

    # Try to launch Google Earth with the KML files
    for ge_path in google_earth_paths:
        if os.path.exists(ge_path):
            try:
                # Get absolute paths to all KML files
                kml_abs_paths = [os.path.abspath(p) for p in kml_paths if os.path.exists(p)]

                if not kml_abs_paths:
                    if verbose:
                        print("No valid KML files to open.")
                    return False

                # Launch Google Earth with all KML files
                subprocess.Popen([ge_path] + kml_abs_paths)

                if verbose:
                    print(f"\nLaunched Google Earth from: {ge_path}")
                    print(f"  Loading {len(kml_abs_paths)} KML file(s):")
                    for kml_abs_path in kml_abs_paths:
                        print(f"    - {kml_abs_path}")

                return True

            except Exception as e:
                if verbose:
                    print(f"Failed to launch Google Earth: {e}")
                return False

    # Google Earth not found
    if verbose:
        print("\nGoogle Earth Pro not found in standard locations.")
        print("To view the results, manually open these files:")
        for kml_path in kml_paths:
            print(f"  - {kml_path}")
        print("\nAlternatively, you can:")
        print("  1. Upload the KML files to Google Earth Web (https://earth.google.com/web/)")
        print("  2. View the GeoTIFF files in QGIS or other GIS software")

    return False


def create_summary_report(
    outputs: dict,
    datacube: xr.DataArray,
    trend_da: xr.DataArray,
    positive_mask: np.ndarray,
    negative_mask: np.ndarray,
    output_path: Optional[str] = None,
    ) -> str:
    """
    Generate a text summary report of the analysis.

    Args:
        outputs: Dictionary of output file paths
        datacube: xarray DataArray with SAR time series
        trend_da: xarray DataArray with trend map
        positive_mask: Numpy array with positive change mask
        negative_mask: Numpy array with negative change mask
        output_path: Path for summary report. If None, uses "outputs/summary.txt".

    Returns:
        Path to created summary report

    Example:
        >>> report_path = create_summary_report(
        ...     outputs, datacube, trend_da, pos_mask, neg_mask
        ... )
    """
    if output_path is None:
        output_path = os.path.join(str(Config.OUTPUT_DIR), "summary.txt")

    # Compute statistics
    n_acquisitions = len(datacube.time)
    start_date = str(datacube.time.values[0])[:10]
    end_date = str(datacube.time.values[-1])[:10]

    trend_min = np.nanmin(trend_da.values)
    trend_max = np.nanmax(trend_da.values)
    trend_mean = np.nanmean(trend_da.values)

    n_positive = np.sum(positive_mask)
    n_negative = np.sum(negative_mask)
    total_pixels = positive_mask.size
    pct_positive = 100 * n_positive / total_pixels
    pct_negative = 100 * n_negative / total_pixels

    # Generate report
    report = f"""
    SAR GROUND SURFACE CHANGE DETECTION - SUMMARY REPORT
    {'=' * 70}

    DATA ACQUISITION
    {'-' * 70}
    Source: Sentinel-1 {datacube.attrs.get('polarization', 'VV')} polarization
    Date range: {start_date} to {end_date}
    Number of acquisitions: {n_acquisitions}
    Spatial resolution: {datacube.attrs.get('resolution_m', 'N/A')} meters

    TREND ANALYSIS
    {'-' * 70}
    Trend range: {trend_min:.4f} to {trend_max:.4f} log-units/year
    Mean trend: {trend_mean:.4f} log-units/year
    Method: {trend_da.attrs.get('method', 'Linear regression')}

    CHANGE DETECTION
    {'-' * 70}
    Threshold: ±{Config.CHANGE_THRESHOLD} log-units/year
    Total pixels: {total_pixels:,}
    Positive changes: {n_positive:,} pixels ({pct_positive:.2f}%)
    Negative changes: {n_negative:,} pixels ({pct_negative:.2f}%)
    Stable areas: {total_pixels - n_positive - n_negative:,} pixels

    OUTPUT FILES
    {'-' * 70}
    """

    for key, path in outputs.items():
        report += f"  {key}: {path}\n"

    report += f"""
    {'=' * 70}
    Report generated: {datacube.attrs.get('processing_date', 'N/A')}
    """

    # Write report to file
    with open(output_path, "w") as f:
        f.write(report)

    print(f"\nSummary report saved: {output_path}")

    return output_path