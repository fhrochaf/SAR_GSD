"""
Export module for generating KML files and other output formats.

This module provides functions to:
- Convert raster masks to vector polygons
- Export results as KML for Google Earth
- Save georeferenced GeoTIFF files
- Launch Google Earth automatically
"""

import numpy as np
import xarray as xr
import rioxarray  # IMPORTANT: This registers the .rio accessor for xarray
import rasterio
from rasterio.features import shapes
from rasterio.transform import from_bounds
from shapely.geometry import shape
import simplekml
import subprocess
import os
from typing import List, Optional, Tuple
from pathlib import Path

from .config import Config


def vectorize_mask(
    mask: np.ndarray,
    bbox: List[float],
    verbose: bool = True,
) -> List[Tuple[dict, int]]:
    """
    Convert binary raster mask to vector polygons.

    Args:
        mask: 2D boolean or uint8 array
        bbox: Bounding box as [lon_min, lat_min, lon_max, lat_max]
        verbose: Print progress messages (default: True)

    Returns:
        List of (geometry, value) tuples from rasterio.features.shapes

    Example:
        >>> polygons = vectorize_mask(positive_mask, [-42.85, -19.65, -42.48, -19.40])
        >>> print(f"Found {len(polygons)} polygons")
    """
    ny, nx = mask.shape
    lon_min, lat_min, lon_max, lat_max = bbox

    # Create affine transform for pixel-to-coordinates conversion
    transform = from_bounds(lon_min, lat_min, lon_max, lat_max, nx, ny)

    # Convert mask to uint8 if boolean
    if mask.dtype == bool:
        mask = mask.astype(np.uint8)

    # Extract polygon geometries
    polygons = list(shapes(mask, transform=transform))

    if verbose:
        # Count polygons with value=1
        n_polygons = sum(1 for geom, val in polygons if val == 1)
        print(f"  Found {n_polygons} polygons")

    return polygons


def create_kml_file(
    positive_mask: np.ndarray,
    negative_mask: np.ndarray,
    bbox: List[float],
    output_path: str,
    positive_color: simplekml.Color = simplekml.Color.blue,
    negative_color: simplekml.Color = simplekml.Color.red,
    verbose: bool = True,
) -> str:
    """
    Create KML file with change detection polygons for Google Earth.

    Args:
        positive_mask: Boolean array indicating positive changes
        negative_mask: Boolean array indicating negative changes
        bbox: Bounding box as [lon_min, lat_min, lon_max, lat_max]
        output_path: Path for output KML file
        positive_color: Color for positive change polygons (default: blue)
        negative_color: Color for negative change polygons (default: red)
        verbose: Print progress messages (default: True)

    Returns:
        Path to created KML file

    Example:
        >>> kml_path = create_kml_file(
        ...     pos_mask, neg_mask,
        ...     bbox=[-42.85, -19.65, -42.48, -19.40],
        ...     output_path="outputs/changes.kml"
        ... )
    """
    if verbose:
        print("Creating KML file...")

    # Initialize KML object
    kml = simplekml.Kml()

    # Define style for positive changes
    style_pos = simplekml.Style()
    style_pos.polystyle.color = positive_color
    style_pos.polystyle.fill = 1  # Fill the polygon
    style_pos.polystyle.outline = 1  # Draw outline

    # Define style for negative changes
    style_neg = simplekml.Style()
    style_neg.polystyle.color = negative_color
    style_neg.polystyle.fill = 1
    style_neg.polystyle.outline = 1

    # Vectorize positive change mask
    if verbose:
        print("  Vectorizing positive change polygons...")
    pos_polygons = vectorize_mask(positive_mask, bbox, verbose=verbose)

    polygon_count_pos = 0
    for geom, val in pos_polygons:
        if val == 1:  # Only process pixels marked as changed
            # Convert rasterio geometry to shapely geometry
            poly_shape = shape(geom)

            # Create KML polygon with exterior coordinates
            poly = kml.newpolygon(
                name=f"Positive Change {polygon_count_pos + 1}",
                outerboundaryis=list(poly_shape.exterior.coords),
            )
            poly.style = style_pos
            polygon_count_pos += 1

    # Vectorize negative change mask
    if verbose:
        print("  Vectorizing negative change polygons...")
    neg_polygons = vectorize_mask(negative_mask, bbox, verbose=verbose)

    polygon_count_neg = 0
    for geom, val in neg_polygons:
        if val == 1:
            poly_shape = shape(geom)

            poly = kml.newpolygon(
                name=f"Negative Change {polygon_count_neg + 1}",
                outerboundaryis=list(poly_shape.exterior.coords),
            )
            poly.style = style_neg
            polygon_count_neg += 1

    # Save KML file
    kml.save(output_path)

    if verbose:
        print(f"\n✓ KML file exported: {output_path}")
        print(f"  Total polygons: {polygon_count_pos + polygon_count_neg}")
        print(f"    Blue (positive): {polygon_count_pos}")
        print(f"    Red (negative): {polygon_count_neg}")

    return output_path


def save_geotiff(
    data: xr.DataArray,
    output_path: str,
    verbose: bool = True,
) -> str:
    """
    Save xarray DataArray as GeoTIFF file.

    Args:
        data: xarray DataArray with rioxarray CRS information
        output_path: Path for output GeoTIFF file
        verbose: Print progress messages (default: True)

    Returns:
        Path to created GeoTIFF file

    Example:
        >>> save_geotiff(trend_da, "outputs/trend.tif")
    """
    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Save as GeoTIFF
    data.rio.to_raster(output_path)

    if verbose:
        print(f"✓ GeoTIFF saved: {output_path}")

    return output_path


def save_all_outputs(
    datacube: xr.DataArray,
    trend_da: xr.DataArray,
    positive_mask_da: xr.DataArray,
    negative_mask_da: xr.DataArray,
    positive_mask: np.ndarray,
    negative_mask: np.ndarray,
    bbox: List[float],
    output_dir: Optional[str] = None,
    verbose: bool = True,
) -> dict:
    """
    Save all output files (datacube, trend, masks, KML).

    Args:
        datacube: xarray DataArray with SAR time series
        trend_da: xarray DataArray with trend map
        positive_mask_da: xarray DataArray with positive change mask
        negative_mask_da: xarray DataArray with negative change mask
        positive_mask: Numpy array with positive change mask
        negative_mask: Numpy array with negative change mask
        bbox: Bounding box as [lon_min, lat_min, lon_max, lat_max]
        output_dir: Output directory. If None, uses Config.OUTPUT_DIR.
        verbose: Print progress messages (default: True)

    Returns:
        Dictionary mapping output names to file paths

    Example:
        >>> outputs = save_all_outputs(
        ...     datacube, trend_da, pos_mask_da, neg_mask_da,
        ...     pos_mask, neg_mask, bbox
        ... )
        >>> print(outputs['kml'])
    """
    if output_dir is None:
        output_dir = str(Config.OUTPUT_DIR)

    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    outputs = {}

    if verbose:
        print("\n" + "=" * 60)
        print("SAVING OUTPUT FILES")
        print("=" * 60)

    # Save datacube as NetCDF
    if verbose:
        print("\n[1/6] Saving datacube (NetCDF)...")
    datacube_path = os.path.join(output_dir, Config.DATACUBE_FILENAME)
    datacube.to_netcdf(datacube_path)
    outputs["datacube"] = datacube_path
    if verbose:
        print(f"  ✓ {datacube_path}")

    # Save latest image as GeoTIFF
    if verbose:
        print("\n[2/6] Saving latest image (GeoTIFF)...")
    latest_path = os.path.join(output_dir, Config.LATEST_IMAGE_FILENAME)
    outputs["latest"] = save_geotiff(datacube.isel(time=-1), latest_path, verbose=False)
    if verbose:
        print(f"  ✓ {latest_path}")

    # Save trend map as GeoTIFF
    if verbose:
        print("\n[3/6] Saving trend map (GeoTIFF)...")
    trend_path = os.path.join(output_dir, Config.TREND_MAP_FILENAME)
    outputs["trend"] = save_geotiff(trend_da, trend_path, verbose=False)
    if verbose:
        print(f"  ✓ {trend_path}")

    # Save positive mask as GeoTIFF
    if verbose:
        print("\n[4/6] Saving positive change mask (GeoTIFF)...")
    pos_mask_path = os.path.join(output_dir, Config.POSITIVE_MASK_FILENAME)
    outputs["positive_mask"] = save_geotiff(positive_mask_da, pos_mask_path, verbose=False)
    if verbose:
        print(f"  ✓ {pos_mask_path}")

    # Save negative mask as GeoTIFF
    if verbose:
        print("\n[5/6] Saving negative change mask (GeoTIFF)...")
    neg_mask_path = os.path.join(output_dir, Config.NEGATIVE_MASK_FILENAME)
    outputs["negative_mask"] = save_geotiff(negative_mask_da, neg_mask_path, verbose=False)
    if verbose:
        print(f"  ✓ {neg_mask_path}")

    # Save KML file
    if verbose:
        print("\n[6/6] Creating KML file...")
    kml_path = os.path.join(output_dir, Config.KML_FILENAME)
    outputs["kml"] = create_kml_file(
        positive_mask, negative_mask, bbox, kml_path, verbose=verbose
    )

    if verbose:
        print("\n" + "=" * 60)
        print("✓ ALL OUTPUT FILES SAVED")
        print("=" * 60)

    return outputs


def launch_google_earth(kml_path: str, verbose: bool = True) -> bool:
    """
    Attempt to launch Google Earth Pro with the specified KML file.

    Args:
        kml_path: Path to KML file
        verbose: Print status messages (default: True)

    Returns:
        True if Google Earth was launched successfully, False otherwise

    Example:
        >>> success = launch_google_earth("outputs/changes.kml")
    """
    # List of common Google Earth Pro installation paths
    google_earth_paths = [
        r"C:\Program Files\Google\Google Earth Pro\client\googleearth.exe",
        r"C:\Program Files (x86)\Google\Google Earth Pro\client\googleearth.exe",
        "/Applications/Google Earth Pro.app/Contents/MacOS/Google Earth Pro",  # macOS
        "/usr/bin/google-earth-pro",  # Linux
    ]

    # Try to launch Google Earth with the KML file
    for path in google_earth_paths:
        if os.path.exists(path):
            try:
                # Get absolute path to KML file
                kml_abs_path = os.path.abspath(kml_path)

                # Launch Google Earth as a subprocess
                subprocess.Popen([path, kml_abs_path])

                if verbose:
                    print(f"\n✓ Launched Google Earth from: {path}")
                    print(f"  Loading KML: {kml_abs_path}")

                return True

            except Exception as e:
                if verbose:
                    print(f"✗ Failed to launch Google Earth: {e}")
                return False

    # Google Earth not found
    if verbose:
        print("\nGoogle Earth Pro not found in standard locations.")
        print(f"To view the results, manually open: {kml_path}")
        print("\nAlternatively, you can:")
        print("  1. Upload the KML to Google Earth Web (https://earth.google.com/web/)")
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

    print(f"\n✓ Summary report saved: {output_path}")

    return output_path
