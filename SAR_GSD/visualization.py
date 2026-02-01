"""
Visualization module for SAR change detection results.

This module provides functions to:
- Display SAR intensity images
- Create change detection overlays
- Generate publication-quality figures
- Customize colormaps and styling
"""

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import rioxarray  # Registers the .rio accessor for xarray
from typing import Optional, Tuple
from pathlib import Path
import folium
import branca.colormap as cm
from io import BytesIO
from PIL import Image
import base64

from .config import Config

def plot_sar_intensity(
    datacube: xr.DataArray,
    time_index: int = -1,
    log_scale: bool = True,
    title: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
    save_path: Optional[str] = None,
    dpi: Optional[int] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Display a single SAR intensity image.

    Args:
        datacube: xarray DataArray with dimensions (time, y, x)
        time_index: Index of time slice to display. Default is -1 (latest).
        log_scale: If True, displays in log10 scale (default: True)
        title: Custom title. If None, generates automatic title with date.
        figsize: Figure size as (width, height) in inches. If None, uses Config.FIGURE_SIZE.
        save_path: Path to save figure. If None, doesn't save.
        dpi: Resolution for saved figure. If None, uses Config.DPI.
        show: Whether to display the figure (default: True)

    Returns:
        matplotlib Figure object

    Example:
        >>> fig = plot_sar_intensity(datacube, time_index=-1, save_path="outputs/intensity.png")
    """
    if figsize is None:
        figsize = Config.FIGURE_SIZE

    if dpi is None:
        dpi = Config.DPI

    # Extract the specified time slice
    img = datacube.isel(time=time_index).values

    # Apply log transformation if requested
    if log_scale:
        img = np.log10(img)
        colorbar_label = "log₁₀(Backscatter)"
    else:
        colorbar_label = "Backscatter"

    # Get acquisition date for title
    date_str = str(datacube.time.values[time_index])[:10]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot the image
    im = ax.imshow(img, cmap=Config.COLORMAP_INTENSITY, aspect="auto")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label=colorbar_label, shrink=0.8)

    # Set title
    if title is None:
        polarization = datacube.attrs.get("polarization", "VV")
        title = f"SAR Intensity ({polarization} Polarization)\nAcquisition: {date_str}"
    ax.set_title(title, fontsize=14, fontweight="bold")

    # Add axis labels
    ax.set_xlabel("Longitude (pixels)", fontsize=11)
    ax.set_ylabel("Latitude (pixels)", fontsize=11)

    # Tight layout
    plt.tight_layout()

    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    # Show if requested
    if show:
        plt.show()

    return fig


def plot_change_overlay(
    datacube: xr.DataArray,
    positive_mask: np.ndarray,
    negative_mask: np.ndarray,
    threshold: float,
    time_index: int = -1,
    log_scale: bool = True,
    title: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
    save_path: Optional[str] = None,
    dpi: Optional[int] = None,
    show: bool = True,
    positive_color: Tuple[float, float, float, float] = (0, 0, 1, 0.6),
    negative_color: Tuple[float, float, float, float] = (1, 0, 0, 0.6),
) -> plt.Figure:
    """
    Display SAR intensity with color-coded change detection overlay.

    Args:
        datacube: xarray DataArray with dimensions (time, y, x)
        positive_mask: Boolean array indicating positive changes
        negative_mask: Boolean array indicating negative changes
        threshold: Threshold value used for change detection
        time_index: Index of time slice to display. Default is -1 (latest).
        log_scale: If True, displays intensity in log10 scale (default: True)
        title: Custom title. If None, generates automatic title.
        figsize: Figure size as (width, height) in inches. If None, uses Config.FIGURE_SIZE.
        save_path: Path to save figure. If None, doesn't save.
        dpi: Resolution for saved figure. If None, uses Config.DPI.
        show: Whether to display the figure (default: True)
        positive_color: RGBA color for positive changes (default: blue with 60% opacity)
        negative_color: RGBA color for negative changes (default: red with 60% opacity)

    Returns:
        matplotlib Figure object

    Example:
        >>> fig = plot_change_overlay(
        ...     datacube, pos_mask, neg_mask, threshold=0.07,
        ...     save_path="outputs/overlay.png"
        ... )
    """
    if figsize is None:
        figsize = Config.FIGURE_SIZE

    if dpi is None:
        dpi = Config.DPI

    # Extract the specified time slice
    img = datacube.isel(time=time_index).values

    # Apply log transformation if requested
    if log_scale:
        img = np.log10(img)

    # Create RGBA overlay array
    overlay = np.zeros((*img.shape, 4))
    overlay[positive_mask] = positive_color
    overlay[negative_mask] = negative_color

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot base SAR intensity in grayscale
    ax.imshow(img, cmap=Config.COLORMAP_INTENSITY, aspect="auto")

    # Overlay change masks
    ax.imshow(overlay, aspect="auto")

    # Set title
    if title is None:
        title = (
            f"SAR Intensity with Persistent Change Detection\n"
            f"Blue: Positive Trend | Red: Negative Trend | Threshold: ±{threshold}"
        )
    ax.set_title(title, fontsize=13, fontweight="bold")

    # Add axis labels
    ax.set_xlabel("Longitude (pixels)", fontsize=11)
    ax.set_ylabel("Latitude (pixels)", fontsize=11)

    # Tight layout
    plt.tight_layout()

    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    # Show if requested
    if show:
        plt.show()

    return fig


def plot_trend_map(
    trend: np.ndarray,
    threshold: Optional[float] = None,
    title: str = "Temporal Trend in SAR Backscatter",
    figsize: Optional[Tuple[float, float]] = None,
    save_path: Optional[str] = None,
    dpi: Optional[int] = None,
    show: bool = True,
    cmap: str = "RdBu_r",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> plt.Figure:
    """
    Display the temporal trend map with diverging colormap.

    Args:
        trend: 2D trend array
        threshold: If provided, draws threshold lines on colorbar
        title: Figure title
        figsize: Figure size as (width, height) in inches. If None, uses Config.FIGURE_SIZE.
        save_path: Path to save figure. If None, doesn't save.
        dpi: Resolution for saved figure. If None, uses Config.DPI.
        show: Whether to display the figure (default: True)
        cmap: Colormap name. Default is 'RdBu_r' (red=negative, blue=positive)
        vmin: Minimum value for colormap. If None, uses data minimum.
        vmax: Maximum value for colormap. If None, uses data maximum.

    Returns:
        matplotlib Figure object

    Example:
        >>> fig = plot_trend_map(trend, threshold=0.07, save_path="outputs/trend_map.png")
    """
    if figsize is None:
        figsize = Config.FIGURE_SIZE

    if dpi is None:
        dpi = Config.DPI

    # Set symmetric color limits if not provided
    if vmin is None or vmax is None:
        vmax_abs = np.nanmax(np.abs(trend))
        vmin = -vmax_abs
        vmax = vmax_abs

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot trend map
    im = ax.imshow(trend, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label="Trend (log₁₀-units/year)", shrink=0.8)

    # Add threshold lines if provided
    if threshold is not None:
        cbar.ax.axhline(threshold, color="black", linestyle="--", linewidth=1, alpha=0.7)
        cbar.ax.axhline(-threshold, color="black", linestyle="--", linewidth=1, alpha=0.7)

    # Set title
    ax.set_title(title, fontsize=14, fontweight="bold")

    # Add axis labels
    ax.set_xlabel("Longitude (pixels)", fontsize=11)
    ax.set_ylabel("Latitude (pixels)", fontsize=11)

    # Tight layout
    plt.tight_layout()

    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    # Show if requested
    if show:
        plt.show()

    return fig


def plot_time_series(
    datacube: xr.DataArray,
    row: int,
    col: int,
    log_scale: bool = True,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 5),
    save_path: Optional[str] = None,
    dpi: Optional[int] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot time series for a single pixel.

    Args:
        datacube: xarray DataArray with dimensions (time, y, x)
        row: Row index (y coordinate)
        col: Column index (x coordinate)
        log_scale: If True, plots in log10 scale (default: True)
        title: Custom title. If None, generates automatic title.
        figsize: Figure size as (width, height) in inches
        save_path: Path to save figure. If None, doesn't save.
        dpi: Resolution for saved figure. If None, uses Config.DPI.
        show: Whether to display the figure (default: True)

    Returns:
        matplotlib Figure object

    Example:
        >>> fig = plot_time_series(datacube, row=100, col=150, save_path="outputs/timeseries.png")
    """
    if dpi is None:
        dpi = Config.DPI

    # Extract time series for the specified pixel
    ts = datacube.isel(y=row, x=col).values

    # Apply log transformation if requested
    if log_scale:
        ts = np.log10(ts)
        ylabel = "log₁₀(Backscatter)"
    else:
        ylabel = "Backscatter"

    # Get time coordinates
    time_coords = datacube.time.values

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot time series
    ax.plot(time_coords, ts, marker="o", linestyle="-", linewidth=1.5, markersize=4)

    # Set title
    if title is None:
        title = f"SAR Backscatter Time Series\nPixel (row={row}, col={col})"
    ax.set_title(title, fontsize=12, fontweight="bold")

    # Add labels
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)

    # Rotate x-axis labels
    plt.xticks(rotation=45, ha="right")

    # Grid
    ax.grid(True, alpha=0.3)

    # Tight layout
    plt.tight_layout()

    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    # Show if requested
    if show:
        plt.show()

    return fig


def create_all_figures(
    datacube: xr.DataArray,
    trend: np.ndarray,
    positive_mask: np.ndarray,
    negative_mask: np.ndarray,
    threshold: float,
    output_dir: Optional[str] = None,
    dpi: Optional[int] = None,
    show: bool = False,
) -> dict:
    """
    Generate all standard visualization figures.

    This convenience function creates:
    - SAR intensity image
    - Change overlay
    - Trend map

    Args:
        datacube: xarray DataArray with dimensions (time, y, x)
        trend: 2D trend array
        positive_mask: Boolean array of positive changes
        negative_mask: Boolean array of negative changes
        threshold: Change detection threshold
        output_dir: Directory to save figures. If None, uses Config.OUTPUT_DIR.
        dpi: Resolution for saved figures. If None, uses Config.DPI.
        show: Whether to display figures (default: False)

    Returns:
        Dictionary mapping figure names to Figure objects

    Example:
        >>> figures = create_all_figures(
        ...     datacube, trend, pos_mask, neg_mask, threshold=0.07
        ... )
    """
    if output_dir is None:
        output_dir = str(Config.OUTPUT_DIR)

    if dpi is None:
        dpi = Config.DPI

    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    figures = {}

    print("Generating figures...")

    # Figure 1: SAR Intensity
    print("  SAR intensity image...")
    fig1 = plot_sar_intensity(
        datacube,
        save_path=f"{output_dir}/{Config.INTENSITY_FIGURE_FILENAME}",
        dpi=dpi,
        show=show,
    )
    figures["intensity"] = fig1

    # Figure 2: Change Overlay
    print("  Change detection overlay...")
    fig2 = plot_change_overlay(
        datacube,
        positive_mask,
        negative_mask,
        threshold,
        save_path=f"{output_dir}/{Config.OVERLAY_FIGURE_FILENAME}",
        dpi=dpi,
        show=show,
    )
    figures["overlay"] = fig2

    # Figure 3: Trend Map
    print("  Trend map...")
    fig3 = plot_trend_map(
        trend,
        threshold=threshold,
        save_path=f"{output_dir}/sar_trend_map.png",
        dpi=dpi,
        show=show,
    )
    figures["trend"] = fig3

    print("All figures generated!")

    return figures


def display_Folium_map(geojson_display_dicts, zoom_start: int = 6):
    """
    Creates a Folium map from a GeoDataFrame, with hover and popup attributes.
    It will center on the first GeoDataFrame of the list

    geojson_display_dict = dictionary with geodataframes mapped to keyworded arguments to be passed to
    folium.GeoJson(...)

    zoom_start = Initial zoom level for the map.
    """

    print('Creating a Folium Map visualization.')

    # Define CRS projection to WGS 84 (Folium Standard)
    for i, dict in enumerate(geojson_display_dicts):
        if dict['data'].crs != 4326:
            dict['data'] = dict['data'].to_crs(epsg=4326)

    # Calculate map center
    center = [
        geojson_display_dicts[0]['data'].geometry.centroid.y.mean(),
        geojson_display_dicts[0]['data'].geometry.centroid.x.mean()
    ]

    # Initialize map
    fmap = folium.Map(
        location=center,
        zoom_start=zoom_start,
        tiles="OpenStreetMap"
    )

    for i in range(len(geojson_display_dicts)):
        #Check if the layer is of point type
        is_point_layer = geojson_display_dicts[i]['data'].geometry.iloc[0].geom_type == "Point"

        # Tooltip (hover)
        if 'attribute_map' in geojson_display_dicts[i]:
            tooltip = folium.GeoJsonTooltip(
                fields=list(geojson_display_dicts[i]['attribute_map'].keys()),
                aliases=[f"{label}:" for label in geojson_display_dicts[i]['attribute_map'].values()],
                localize=True
            )

            # Popup (click)
            popup = folium.GeoJsonPopup(
                fields=list(geojson_display_dicts[i]['attribute_map'].keys()),
                aliases=[f"{label}:" for label in geojson_display_dicts[i]['attribute_map'].values()],
                localize=True
            )
        else:
            tooltip = None
            popup = None

        # Define style - use static style dict or callable style function
        feature_settings = geojson_display_dicts[i].get('feature_settings', None)

        # Build GeoJson kwargs
        geojson_kwargs = {
            'data': geojson_display_dicts[i]['data'],
            'name': geojson_display_dicts[i].get('name', f'Layer {i}'),
            'zoom_on_click': True,
            'tooltip': tooltip,
            'marker': geojson_display_dicts[i].get('marker', None),
            'popup': popup,
            'highlight_function': geojson_display_dicts[i].get('highlight_function', None),
            'popup_keep_highlighted': True
        }

        # Apply styling for non-point layers
        if not is_point_layer and feature_settings is not None:
            if callable(feature_settings):
                # User provided a style function
                geojson_kwargs['style_function'] = feature_settings
            else:
                # Use 'style' parameter for static styles (Folium >= 0.14)
                geojson_kwargs['style'] = feature_settings

        # Add geospatial data to map
        folium.GeoJson(**geojson_kwargs).add_to(fmap)

    return fmap


def add_raster_to_folium(fmap, raster_data, name="Raster Layer", opacity=0.6,
                         cmap='gray', vmin=None, vmax=None, percentile_clip=(2, 98),
                         log_transform=False, nodata_value=None):
    """
    Add a raster layer to an existing Folium map with automatic reprojection.

    Parameters:
    -----------
    fmap : folium.Map
        Existing Folium map object to add the raster to
    raster_data : xarray.DataArray
        Raster data with rio accessor (rioxarray)
    name : str
        Name of the layer for layer control
    opacity : float
        Opacity of the raster overlay (0-1)
    cmap : str or matplotlib.colors.Colormap
        Matplotlib colormap name or object (e.g., 'gray', 'RdYlBu', 'viridis')
    vmin : float, optional
        Minimum value for colormap normalization. If None, uses percentile_clip
    vmax : float, optional
        Maximum value for colormap normalization. If None, uses percentile_clip
    percentile_clip : tuple
        Percentiles to clip data for display (min, max). Only used if vmin/vmax not provided
    log_transform : bool
        Whether to apply log10 transformation (useful for SAR data)
    nodata_value : float, optional
        Value to treat as nodata (will be transparent). If None, uses NaN

    Returns:
    --------
    folium.Map
        The map object with the raster layer added
    """

    # Reproject to WGS84 if needed (Folium standard)
    if raster_data.rio.crs != "EPSG:4326":
        print(f"Reprojecting from {raster_data.rio.crs} to EPSG:4326...")
        raster_data = raster_data.rio.reproject("EPSG:4326")

    # Get bounds in Folium format [[south, west], [north, east]]
    bounds = raster_data.rio.bounds()  # (minx, miny, maxx, maxy)
    bounds_folium = [[bounds[1], bounds[0]], [bounds[3], bounds[2]]]

    # Get data as numpy array
    data = raster_data.to_numpy()

    # Handle nodata
    if nodata_value is not None:
        data = np.where(data == nodata_value, np.nan, data)

    # Apply log transformation if requested
    if log_transform:
        data = np.log10(data + 1e-10)  # Add small value to avoid log(0)

    # Normalize data
    if vmin is None or vmax is None:
        vmin_calc, vmax_calc = np.nanpercentile(data, percentile_clip)
        vmin = vmin if vmin is not None else vmin_calc
        vmax = vmax if vmax is not None else vmax_calc

    data_norm = np.clip((data - vmin) / (vmax - vmin), 0, 1)

    # Get colormap
    if isinstance(cmap, str):
        cmap_obj = plt.get_cmap(cmap)
    else:
        cmap_obj = cmap

    # Apply colormap
    rgba = cmap_obj(data_norm)

    # Set NaN values to transparent
    rgba[np.isnan(data)] = [0, 0, 0, 0]

    # Convert to uint8 image
    img = Image.fromarray((rgba * 255).astype(np.uint8), mode='RGBA')

    # Convert to base64 for Folium
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()

    # Add to Folium map
    folium.raster_layers.ImageOverlay(
        image=f"data:image/png;base64,{img_str}",
        bounds=bounds_folium,
        opacity=opacity,
        name=name,
        interactive=True,
        cross_origin=False
    ).add_to(fmap)

    print(f"Added raster layer '{name}' to map")
    return fmap