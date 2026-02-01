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
from rasterio.features import shapes
from rasterio.transform import from_bounds
from shapely.geometry import shape
import simplekml
import os
from typing import List, Optional, Tuple
from pathlib import Path

from .config import Config
from .vector_ops import vectorize_mask


def gdf_to_kml(
    gdf,
    output_path: str,
    name_column: Optional[str] = None,
    description_columns: Optional[List[str]] = None,
    color: str = "ff0000ff",  # AABBGGRR format (default: red)
    fill_color: Optional[str] = None,
    line_width: float = 1.0,
    fill: bool = True,
    verbose: bool = True,
) -> str:
    """
    Export a GeoDataFrame to KML format for Google Earth.

    Parameters:
    -----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame with geometries to export
    output_path : str
        Path for output KML file
    name_column : str, optional
        Column to use for feature names. If None, uses index.
    description_columns : list of str, optional
        Columns to include in feature description popup.
        If None, includes all non-geometry columns.
    color : str
        Line/outline color in KML format (AABBGGRR hex). Default is red.
        Common colors: 'ff0000ff' (red), 'ffff0000' (blue), 'ff00ff00' (green)
    fill_color : str, optional
        Fill color for polygons in KML format (AABBGGRR hex).
        If None, uses the same as color with 50% transparency.
    line_width : float
        Width of lines/outlines in pixels. Default is 1.0.
    fill : bool
        Whether to fill polygons. Default is True.
    verbose : bool
        Print progress messages. Default is True.

    Returns:
    --------
    str
        Path to created KML file

    Example:
    --------
    >>> gdf_to_kml(
    ...     gdf_changes,
    ...     "outputs/changes.kml",
    ...     name_column="change_type",
    ...     description_columns=["total_area", "mean_trend"],
    ...     color="ffff0000",  # Blue
    ... )

    Notes:
    ------
    - GeoDataFrame will be automatically reprojected to WGS84 (EPSG:4326) if needed
    - Supports Point, LineString, and Polygon geometries (including Multi* variants)
    - KML color format is AABBGGRR (Alpha, Blue, Green, Red)
    """
    import geopandas as gpd

    if verbose:
        print(f"Exporting GeoDataFrame to KML: {output_path}")

    # Create a copy and reproject to WGS84 if needed
    gdf_export = gdf.copy()
    if gdf_export.crs is not None and not gdf_export.crs.equals("EPSG:4326"):
        if verbose:
            print(f"  Reprojecting from {gdf_export.crs} to EPSG:4326...")
        gdf_export = gdf_export.to_crs("EPSG:4326")

    # Initialize KML
    kml = simplekml.Kml()

    # Create shared style
    shared_style = simplekml.Style()
    shared_style.linestyle.color = color
    shared_style.linestyle.width = line_width
    if fill_color is None:
        # Use same color with 50% transparency for fill
        fill_color = "7f" + color[2:]
    shared_style.polystyle.color = fill_color
    shared_style.polystyle.fill = 1 if fill else 0
    shared_style.polystyle.outline = 1
    shared_style.iconstyle.color = color

    # Determine description columns
    if description_columns is None:
        description_columns = [col for col in gdf_export.columns if col != 'geometry']

    # Process each feature
    feature_count = 0
    for idx, row in gdf_export.iterrows():
        geom = row.geometry

        if geom is None or geom.is_empty:
            continue

        # Determine feature name
        if name_column and name_column in row:
            name = str(row[name_column])
        else:
            name = f"Feature {idx}"

        # Build description from columns
        desc_parts = []
        for col in description_columns:
            if col in row and col != name_column:
                value = row[col]
                # Format numeric values
                if isinstance(value, float):
                    desc_parts.append(f"<b>{col}:</b> {value:.4f}")
                else:
                    desc_parts.append(f"<b>{col}:</b> {value}")
        description = "<br>".join(desc_parts) if desc_parts else None

        # Handle different geometry types
        geom_type = geom.geom_type

        if geom_type == 'Point':
            kml_geom = kml.newpoint(name=name, coords=[(geom.x, geom.y)])
            kml_geom.style = shared_style

        elif geom_type == 'MultiPoint':
            for i, point in enumerate(geom.geoms):
                kml_geom = kml.newpoint(
                    name=f"{name} ({i+1})",
                    coords=[(point.x, point.y)]
                )
                kml_geom.style = shared_style

        elif geom_type == 'LineString':
            kml_geom = kml.newlinestring(
                name=name,
                coords=list(geom.coords)
            )
            kml_geom.style = shared_style

        elif geom_type == 'MultiLineString':
            for i, line in enumerate(geom.geoms):
                kml_geom = kml.newlinestring(
                    name=f"{name} ({i+1})",
                    coords=list(line.coords)
                )
                kml_geom.style = shared_style

        elif geom_type == 'Polygon':
            kml_geom = kml.newpolygon(
                name=name,
                outerboundaryis=list(geom.exterior.coords)
            )
            # Add inner rings (holes) if present
            if geom.interiors:
                kml_geom.innerboundaryis = [list(ring.coords) for ring in geom.interiors]
            kml_geom.style = shared_style

        elif geom_type == 'MultiPolygon':
            for i, poly in enumerate(geom.geoms):
                kml_geom = kml.newpolygon(
                    name=f"{name} ({i+1})",
                    outerboundaryis=list(poly.exterior.coords)
                )
                if poly.interiors:
                    kml_geom.innerboundaryis = [list(ring.coords) for ring in poly.interiors]
                kml_geom.style = shared_style

        else:
            if verbose:
                print(f"  Warning: Unsupported geometry type '{geom_type}' at index {idx}")
            continue

        # Add description if we created a geometry
        if description and 'kml_geom' in locals():
            kml_geom.description = description

        feature_count += 1

    # Save KML
    kml.save(output_path)

    if verbose:
        print(f"Exported {feature_count} features to {output_path}")

    return output_path


def save_all_outputs(
    trend: np.ndarray,
    positive_mask: np.ndarray,
    negative_mask: np.ndarray,
    bbox: List[float],
    output_dir: Optional[str] = None,
    verbose: bool = True,
) -> dict:
    """
    Save all output files (datacube, trend, masks, KML).

    Args:
        trend: Numpy array with trend values
        positive_mask: Numpy array with positive change mask
        negative_mask: Numpy array with negative change mask
        bbox: Bounding box as [lon_min, lat_min, lon_max, lat_max]
        output_dir: Output directory. If None, uses Config.OUTPUT_DIR.
        verbose: Print progress messages (default: True)

    Returns:
        Dictionary mapping output names to file paths

    Example:
        >>> outputs = save_all_outputs(
        ...     trend, pvalues, pos_mask, neg_mask, bbox
        ... )
    """
    if output_dir is None:
        output_dir = str(Config.OUTPUT_DIR)

    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    outputs = {}

    # Save trend map as GeoTIFF
    if verbose:
        print("\nSaving trend map (GeoTIFF)...")
    trend_path = os.path.join(output_dir, Config.TREND_MAP_FILENAME)
    trend.rio.to_raster(trend_path)
    outputs["trend"] = trend_path
    if verbose:
        print(f"  ✓ {trend_path}")

    # Save positive mask as GeoTIFF
    if verbose:
        print("\nSaving positive mask (GeoTIFF)...")
    pos_mask_path = os.path.join(output_dir, Config.POSITIVE_MASK_FILENAME)
    positive_mask.rio.to_raster(pos_mask_path)
    outputs["positive_mask"] = pos_mask_path
    if verbose:
        print(f"  ✓ {pos_mask_path}")

    # Save negative mask as GeoTIFF
    if verbose:
        print("\nSaving negative mask (GeoTIFF)...")
    neg_mask_path = os.path.join(output_dir, Config.NEGATIVE_MASK_FILENAME)
    negative_mask.rio.to_raster(neg_mask_path)
    outputs["negative_mask"] = neg_mask_path
    if verbose:
        print(f"  ✓ {neg_mask_path}")

    return outputs