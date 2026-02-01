"""
SAR_GSD: SAR Ground Surface Deformation Detection Package

A Python package for detecting ground surface changes using Sentinel-1 SAR
imagery through temporal trend analysis.
"""
import xarray as xr
import pandas as pd
import numpy as np
import fiona
import geopandas as gpd

from .config import Config

from .utils import (
    launch_google_earth,
    create_summary_report
)
from .download import (
    build_datacube,
    get_available_s1_dates,
    request_s1_image,
    get_dem
)
from .processing import (
    process_sar_timeseries_tensor,
    get_device,
    calculate_slope
)
from .visualization import (
    plot_sar_intensity,
    plot_change_overlay,
    plot_trend_map,
    plot_time_series,
    create_all_figures,
    display_Folium_map,
)
from .export import (
    save_all_outputs,
    gdf_to_kml
)
from .vector_ops import (
    buff_and_clip_geopackage,
    zonal_statistics,
    calculate_change_areas,
    vectorize_mask,
    vectorize_and_analyze_changes
)