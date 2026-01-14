"""
Visualization module for InSAR deformation analysis.

Provides functions to create publication-quality figures, maps, and animations.
"""

import logging
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation, PillowWriter
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from .config import Config

logger = logging.getLogger(__name__)


def plot_timeseries(
    datacube: xr.DataArray,
    pixel_coords: Tuple[int, int],
    save_path: Optional[Path] = None,
    title: Optional[str] = None
) -> plt.Figure:
    """
    Plot time series for a single pixel.
    
    Args:
        datacube: Input time series
        pixel_coords: (y_index, x_index)
        save_path: Path to save figure
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    y_idx, x_idx = pixel_coords
    
    # Extract pixel time series
    pixel_ts = datacube[:, y_idx, x_idx]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot
    ax.plot(pixel_ts.time, pixel_ts.values, marker='o', linestyle='-', linewidth=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    else:
        ax.set_title(f'Time Series at Pixel ({y_idx}, {x_idx})', fontsize=14)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=Config.DPI, bbox_inches='tight')
        logger.info(f"Saved time series plot to {save_path}")
    
    return fig


def plot_velocity_map(
    velocity_map: xr.DataArray,
    save_path: Optional[Path] = None,
    title: str = "Deformation Velocity",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = None
) -> plt.Figure:
    """
    Plot deformation velocity map.
    
    Args:
        velocity_map: Velocity DataArray
        save_path: Path to save figure
        title: Plot title
        vmin, vmax: Color scale limits
        cmap: Colormap name
        
    Returns:
        Matplotlib figure
    """
    if cmap is None:
        cmap = Config.COLORMAP_DEFORMATION
    
    # Auto-scale if not provided
    if vmin is None or vmax is None:
        vmin_auto = np.nanpercentile(velocity_map, 5)
        vmax_auto = np.nanpercentile(velocity_map, 95)
        vmax_abs = max(abs(vmin_auto), abs(vmax_auto))
        vmin = vmin or -vmax_abs
        vmax = vmax or vmax_abs
    
    # Create figure with map projection if CRS available
    if hasattr(velocity_map, 'rio') and velocity_map.rio.crs:
        fig = plt.figure(figsize=(14, 10))
        ax = plt.axes(projection=ccrs.PlateCarree())
        
        # Add map features
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle=':')
        ax.gridlines(draw_labels=True, alpha=0.3)
        
        # Plot
        im = velocity_map.plot(
            ax=ax,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            cbar_kwargs={'label': 'Velocity (mm/year)', 'shrink': 0.8}
        )
    else:
        fig, ax = plt.subplots(figsize=(12, 10))
        im = velocity_map.plot(
            ax=ax,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            cbar_kwargs={'label': 'Velocity (mm/year)'}
        )
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=Config.DPI, bbox_inches='tight')
        logger.info(f"Saved velocity map to {save_path}")
    
    return fig


def plot_deformation_histogram(
    velocity_map: xr.DataArray,
    save_path: Optional[Path] = None,
    bins: int = 50
) -> plt.Figure:
    """
    Plot histogram of deformation velocities.
    
    Args:
        velocity_map: Velocity DataArray
        save_path: Path to save figure
        bins: Number of histogram bins
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Flatten and remove NaNs
    values = velocity_map.values.flatten()
    values = values[~np.isnan(values)]
    
    # Plot histogram
    ax.hist(values, bins=bins, alpha=0.7, edgecolor='black')
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero deformation')
    ax.axvline(np.median(values), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(values):.2f}')
    
    ax.set_xlabel('Velocity (mm/year)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Deformation Velocities', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=Config.DPI, bbox_inches='tight')
        logger.info(f"Saved histogram to {save_path}")
    
    return fig


def create_animation(
    datacube: xr.DataArray,
    save_path: Path,
    fps: int = 2,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None
) -> None:
    """
    Create animation of time series evolution.
    
    Args:
        datacube: Input time series
        save_path: Path to save animation (GIF)
        fps: Frames per second
        vmin, vmax: Color scale limits
    """
    logger.info(f"Creating animation with {len(datacube.time)} frames...")
    
    # Auto-scale
    if vmin is None:
        vmin = np.nanpercentile(datacube, 2)
    if vmax is None:
        vmax = np.nanpercentile(datacube, 98)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Initialize plot
    im = ax.imshow(datacube.isel(time=0), cmap='viridis', vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, label='Intensity')
    title = ax.set_title('', fontsize=14, fontweight='bold')
    
    def update(frame):
        im.set_data(datacube.isel(time=frame))
        date_str = str(datacube.time.values[frame])[:10]
        title.set_text(f'Frame {frame + 1}/{len(datacube.time)} - {date_str}')
        return [im, title]
    
    # Create animation
    anim = FuncAnimation(fig, update, frames=len(datacube.time), interval=1000/fps, blit=True)
    
    # Save
    writer = PillowWriter(fps=fps)
    anim.save(save_path, writer=writer)
    
    plt.close(fig)
    logger.info(f"Saved animation to {save_path}")


def plot_coherence_map(
    coherence_map: xr.DataArray,
    save_path: Optional[Path] = None,
    threshold: float = 0.3
) -> plt.Figure:
    """
    Plot coherence/stability map.
    
    Args:
        coherence_map: Coherence DataArray
        save_path: Path to save figure
        threshold: Coherence threshold for highlighting
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot
    im = coherence_map.plot(
        ax=ax,
        cmap='RdYlGn',
        vmin=0,
        vmax=1,
        cbar_kwargs={'label': 'Coherence (0-1)'}
    )
    
    # Add threshold contour
    if threshold:
        coherence_map.plot.contour(
            ax=ax,
            levels=[threshold],
            colors='black',
            linewidths=2,
            linestyles='--'
        )
    
    ax.set_title('Temporal Coherence Map', fontsize=14, fontweight='bold')
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=Config.DPI, bbox_inches='tight')
        logger.info(f"Saved coherence map to {save_path}")
    
    return fig


def generate_comparison_plots(
    before: xr.DataArray,
    after: xr.DataArray,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Create before/after comparison plots.
    
    Args:
        before: Before image
        after: After image
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Shared color scale
    vmin = min(np.nanpercentile(before, 2), np.nanpercentile(after, 2))
    vmax = max(np.nanpercentile(before, 98), np.nanpercentile(after, 98))
    
    # Before
    before.plot(ax=axes[0], cmap='gray', vmin=vmin, vmax=vmax, add_colorbar=False)
    axes[0].set_title('Before', fontsize=14, fontweight='bold')
    
    # After
    after.plot(ax=axes[1], cmap='gray', vmin=vmin, vmax=vmax, add_colorbar=False)
    axes[1].set_title('After', fontsize=14, fontweight='bold')
    
    # Difference
    diff = after - before
    diff.plot(ax=axes[2], cmap='RdBu_r', cbar_kwargs={'label': 'Change'})
    axes[2].set_title('Difference (After - Before)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=Config.DPI, bbox_inches='tight')
        logger.info(f"Saved comparison plots to {save_path}")
    
    return fig


def plot_multiple_timeseries(
    datacube: xr.DataArray,
    pixel_list: List[Tuple[int, int]],
    labels: Optional[List[str]] = None,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot multiple pixel time series on same axes.
    
    Args:
        datacube: Input time series
        pixel_list: List of (y, x) pixel coordinates
        labels: Labels for each pixel
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    
    for idx, (y, x) in enumerate(pixel_list):
        pixel_ts = datacube[:, y, x]
        label = labels[idx] if labels else f'Pixel ({y}, {x})'
        ax.plot(pixel_ts.time, pixel_ts.values, marker='o', label=label, linewidth=2)
    
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title('Multi-Pixel Time Series Comparison', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=Config.DPI, bbox_inches='tight')
        logger.info(f"Saved multi-pixel time series to {save_path}")
    
    return fig


def plot_results():
    """Plot results."""
    pass
