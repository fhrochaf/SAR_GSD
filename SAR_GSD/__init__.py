"""
SAR_GSD: SAR Ground Surface Deformation Detection Package

A Python package for detecting ground surface changes using Sentinel-1 SAR
imagery through temporal trend analysis.
"""
import xarray as xr
import time

from .config import Config
from .download import (
    build_datacube,
    get_available_s1_dates,
    request_s1_image
)
from .processing import (
    process_sar_timeseries,
    get_device
)
from .visualization import (
    plot_sar_intensity,
    plot_change_overlay,
    plot_trend_map,
    plot_time_series,
    create_all_figures,
)
from .export import (
    create_kml_file,
    save_geotiff,
    save_all_outputs,
    launch_google_earth,
    create_summary_report,
)