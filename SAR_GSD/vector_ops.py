"""
Vector operations module for raster-vector analysis.

This module provides functions to:
- Calculate zonal statistics for raster masks within vector geometries
- Compute area of change masks within polygons
- Sample raster values within vector features
"""

import numpy as np
import xarray as xr
import geopandas as gpd
import fiona
from shapely.geometry import shape, mapping
from rasterstats import zonal_stats
from rasterio.features import shapes
from typing import List, Tuple, Optional


def zonal_statistics(gdf, data_array, stats=['mean', 'min', 'max', 'std', 'count']):
    """
    Perform zonal statistics on a GeoDataFrame using a data array.
    
    Parameters:
    -----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame with polygon geometries
    data_array : xarray.DataArray
        DataArray with 'x' and 'y' coordinates and spatial reference
    stats : list
        Statistics to calculate: 'mean', 'min', 'max', 'std', 'count', 'sum', 'median'
    
    Returns:
    --------
    geopandas.GeoDataFrame
        GeoDataFrame with original data and zonal statistics as new columns
    """
    
    # Create a copy to avoid modifying the original
    result_gdf = gdf.copy()

    # Prepare data for zonal statistics
    # Extract array and transform from xarray
    array_data = data_array.values

    # Create affine transform
    transform = data_array.rio.transform()

    # Determine appropriate nodata value based on dtype
    # NaN only works for float arrays, not integers
    if hasattr(data_array, 'rio') and data_array.rio.nodata is not None:
        nodata_val = data_array.rio.nodata
    elif np.issubdtype(array_data.dtype, np.floating):
        nodata_val = np.nan
    else:
        # For integer arrays, use None to let rasterstats handle it
        nodata_val = None

    # Perform zonal statistics
    zs_results = zonal_stats(
        result_gdf.geometry,
        array_data,
        affine=transform,
        stats=stats,
        nodata=nodata_val
    )
    
    # Add zonal statistics to GeoDataFrame
    for stat in stats:
        result_gdf[stat] = [result[stat] for result in zs_results]
    
    return result_gdf


def buff_and_clip_geopackage(input_path, output_path, buffer_distance, clip_geom=None, 
                              layer=None, cap_style=1, join_style=1, resolution=16):
    """
    Open a geopackage with Fiona, calculate buffer around geometries, clip to boundary, and save.
    
    Parameters:
    -----------
    input_path : str
        Path to input geopackage
    output_path : str
        Path to output geopackage
    buffer_distance : float
        Buffer distance in the units of the CRS (meters if projected, degrees if geographic)
    clip_geom : shapely.geometry or dict, optional
        Geometry to clip the buffered results to. Can be a Shapely geometry or GeoJSON-like dict.
        If None, no clipping is performed.
    layer : str, optional
        Layer name if geopackage has multiple layers
    cap_style : int
        1=round (default), 2=flat, 3=square
    join_style : int
        1=round (default), 2=mitre, 3=bevel
    resolution : int
        Number of segments per quarter circle (default: 16)
    
    Returns:
    --------
    str
        Path to output file
    """
    
    # Open input geopackage with Fiona
    if layer:
        src = fiona.open(input_path, layer=layer)
    else:
        src = fiona.open(input_path)
    
    # Get metadata from source
    input_crs = src.crs
    input_schema = src.schema.copy()
    
    # Update schema - buffer will always create Polygon/MultiPolygon
    input_schema['geometry'] = 'Polygon'
    
    # Convert clip_geom to Shapely if it's a dict
    if clip_geom is not None:
        if isinstance(clip_geom, dict):
            clip_geom = shape(clip_geom)
    
    # Create output geopackage
    with fiona.open(
        output_path,
        'w',
        driver='GPKG',
        crs=input_crs,
        schema=input_schema
    ) as dst:
        
        # Process each feature
        for feature in src:
            # Convert to Shapely geometry
            geom = shape(feature['geometry'])

            # Clip to boundary if provided
            if clip_geom is not None:
                geom = geom.intersection(clip_geom)
            
            # Calculate buffer
            buffered_geom = geom.buffer(
                buffer_distance,
                cap_style=cap_style,
                join_style=join_style,
                resolution=resolution
            )

            # Skip empty geometries
            if buffered_geom.is_empty:
                continue              
            
            # Explode MultiPolygons into individual Polygons
            if buffered_geom.geom_type == 'MultiPolygon':
                polygons = list(buffered_geom.geoms)
            elif buffered_geom.geom_type == 'GeometryCollection':
                # Clipping can also produce GeometryCollections with mixed types
                polygons = [g for g in buffered_geom.geoms if g.geom_type == 'Polygon']
            else:
                polygons = [buffered_geom]

            for poly in polygons:
                new_feature = {
                    'geometry': mapping(poly),
                    'properties': feature['properties']
                }
                
            dst.write(new_feature)
    
    src.close()
    
    print(f"Buffered geopackage saved to: {output_path}")
    return output_path


def calculate_change_areas(gdf, positive_mask, negative_mask, resolution, output_path=None):
    """
    Calculate positive and negative change areas from zonal statistics.

    Performs zonal statistics on positive and negative change masks,
    calculates areas and ratios relative to total polygon area.

    Parameters:
    -----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame with polygon geometries (e.g., buffered zones)
    positive_mask : xarray.DataArray
        DataArray containing positive change mask
    negative_mask : xarray.DataArray
        DataArray containing negative change mask
    resolution : float
        Pixel resolution in the same units as the CRS (typically meters)
    output_path : str, optional
        Path to save the resulting GeoDataFrame. If None, file is not saved.

    Returns:
    --------
    geopandas.GeoDataFrame
        GeoDataFrame with the following additional columns:
        - count_pos: pixel count of positive changes (sum of binary mask)
        - pos_area: area of positive changes (resolution² * count_pos)
        - count_neg: pixel count of negative changes (sum of binary mask)
        - neg_area: area of negative changes (resolution² * count_neg)
        - total_area: total geometry area
        - pos_area_percent: positive change area / total area
        - neg_area_percent: negative change area / total area

    Example:
    --------
    >>> gdf_result = calculate_change_areas(gdf_buff, positive_mask, negative_mask, res=10, output_path='output.gpkg')
    """
    # Perform zonal statistics for positive changes
    # Use 'sum' instead of 'count' because for binary masks (0/1),
    # sum gives the count of pixels with value 1 (changes detected)
    gdf_result = zonal_statistics(gdf, positive_mask, stats=['sum'])
    gdf_result = gdf_result.rename(columns={'sum': 'count_pos'})
    gdf_result['pos_area_ha'] = gdf_result['count_pos'] * (resolution ** 2) / 10000

    # Perform zonal statistics for negative changes
    gdf_result = zonal_statistics(gdf_result, negative_mask, stats=['sum'])
    gdf_result = gdf_result.rename(columns={'sum': 'count_neg'})
    gdf_result['neg_area_ha'] = gdf_result['count_neg'] * (resolution ** 2) / 10000

    # Calculate total area
    gdf_result['total_area_ha'] = gdf_result.geometry.area / 10000

    # Calculate percentage of positive change area
    gdf_result['pos_area_perc'] = gdf_result['pos_area_ha'] * 100 / gdf_result['total_area_ha']

    # Calculate percentage of negative change area
    gdf_result['neg_area_perc'] = gdf_result['neg_area_ha'] * 100 / gdf_result['total_area_ha']

    # Save vector data if output path provided
    if output_path is not None:
        gdf_result.to_file(output_path)
        print(f"Change areas saved to: {output_path}")

    return gdf_result


def vectorize_mask(
    mask: xr.DataArray,
    verbose: bool = True,
) -> List[Tuple[dict, int]]:
    """
    Convert binary raster mask to vector polygons.

    Args:
        mask: 2D boolean or uint8 data array
        verbose: Print progress messages (default: True)

    Returns:
        List of (geometry, value) tuples from rasterio.features.shapes

    Example:
        >>> polygons = vectorize_mask(positive_mask, [-42.85, -19.65, -42.48, -19.40])
        >>> print(f"Found {len(polygons)} polygons")
    """

    # Create affine transform for pixel-to-coordinates conversion
    transform = mask.rio.transform()

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


def vectorize_and_analyze_changes(
    datacube: xr.Dataset,
    crs: str = None,
    verbose: bool = True,
) -> gpd.GeoDataFrame:
    """
    Vectorize change masks and compute zonal statistics for trend and slope classes.

    This function:
    1. Vectorizes positive and negative change masks from the datacube
    2. Creates GeoDataFrames with polygon geometries
    3. Calculates total area for each polygon
    4. Computes zonal statistics against the trend raster (mean trend per polygon)
    5. Computes zonal statistics against slope classes (area per slope class)
    6. Concatenates positive and negative change polygons into a single GeoDataFrame

    Parameters:
    -----------
    datacube : xarray.Dataset
        Dataset containing:
        - 'positive_mask': binary mask of positive changes
        - 'negative_mask': binary mask of negative changes
        - 'trend': trend raster for zonal statistics
        - 'slope_class': slope classification raster with 'Slope classes' attribute
    crs : str, optional
        CRS for the output GeoDataFrame. If None, uses datacube's CRS.
    verbose : bool
        Print progress messages (default: True)

    Returns:
    --------
    geopandas.GeoDataFrame
        GeoDataFrame with columns:
        - geometry: polygon geometry
        - change_type: 'positive' or 'negative'
        - total_area: polygon area in CRS units
        - mean: mean trend value within polygon
        - count_{class}: pixel count for each slope class
        - area_{class}: area for each slope class

    Example:
    --------
    >>> gdf_changes = vectorize_and_analyze_changes(datacube)
    >>> gdf_positive = gdf_changes[gdf_changes['change_type'] == 'positive']
    """
    from shapely import Polygon
    import pandas as pd

    if crs is None:
        crs = datacube.rio.crs

    # Get pixel resolution
    res = abs(datacube.rio.resolution()[0])

    def _create_gdf_from_mask(mask_da, change_type):
        """Helper to vectorize mask and create GeoDataFrame."""
        if verbose:
            print(f"Vectorizing {change_type} change mask...")

        geoms = vectorize_mask(mask_da, verbose=verbose)

        # Create polygon geometries (only for value=1, i.e., change pixels)
        geom_list = []
        for i, (geom, val) in enumerate(geoms):
            if val != 1:
                continue
            try:
                geom_list.append(Polygon(geom['coordinates'][0]))
            except Exception as e:
                if verbose:
                    print(f"  Geom {i} ({geom['type']}) failed: {e}")

        if not geom_list:
            if verbose:
                print(f"  No valid polygons found for {change_type} changes")
            return None

        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(geometry=geom_list, crs=crs)
        gdf['change_type'] = change_type
        gdf['total_area_ha'] = gdf.geometry.area / 10000

        return gdf

    # Create GeoDataFrames from masks
    gdf_pos = _create_gdf_from_mask(datacube["positive_mask"], 'positive')
    gdf_neg = _create_gdf_from_mask(datacube["negative_mask"], 'negative')

    # Calculate zonal statistics against trend raster
    if verbose:
        print("Computing zonal statistics against trend raster...")

    if gdf_pos is not None and "trend" in datacube:
        gdf_pos = zonal_statistics(gdf_pos, datacube["trend"], stats=['mean'])
        gdf_pos = gdf_pos.rename(columns={'mean': 'mean_trend'})

    if gdf_neg is not None and "trend" in datacube:
        gdf_neg = zonal_statistics(gdf_neg, datacube["trend"], stats=['mean'])
        gdf_neg = gdf_neg.rename(columns={'mean': 'mean_trend'})

    # Calculate zonal statistics against slope classes
    if "slope_class" in datacube and 'Slope classes' in datacube["slope_class"].attrs:
        slope_classes = datacube["slope_class"].attrs['Slope classes']

        if verbose:
            print(f"Computing zonal statistics for {len(slope_classes)} slope classes...")

        for class_val, class_name in slope_classes.items():
            class_mask = (datacube["slope_class"] == class_val)

            if gdf_pos is not None:
                gdf_pos = zonal_statistics(gdf_pos, class_mask, stats=['sum'])
                gdf_pos = gdf_pos.rename(columns={'sum': f'count_{class_name}'})
                gdf_pos[f'area_{class_name}'] = gdf_pos[f'count_{class_name}'] * (res ** 2) / 10000
                gdf_pos[f'area_{class_name}_perc'] = gdf_pos[f'area_{class_name}'] / gdf_pos['total_area_ha'] *100

            if gdf_neg is not None:
                gdf_neg = zonal_statistics(gdf_neg, class_mask, stats=['sum'])
                gdf_neg = gdf_neg.rename(columns={'sum': f'count_{class_name}'})
                gdf_neg[f'area_{class_name}'] = gdf_neg[f'count_{class_name}'] * (res ** 2) / 10000
                gdf_neg[f'area_{class_name}_perc'] = gdf_neg[f'area_{class_name}'] / gdf_neg['total_area_ha'] *100

    # Concatenate positive and negative GeoDataFrames
    gdfs_to_concat = [gdf for gdf in [gdf_pos, gdf_neg] if gdf is not None]

    if not gdfs_to_concat:
        raise ValueError("No valid polygons found in either positive or negative masks")

    gdf_result = pd.concat(gdfs_to_concat, ignore_index=True)

    if verbose:
        n_pos = len(gdf_pos) if gdf_pos is not None else 0
        n_neg = len(gdf_neg) if gdf_neg is not None else 0
        print(f"Created GeoDataFrame with {n_pos} positive and {n_neg} negative change polygons")

    return gdf_result