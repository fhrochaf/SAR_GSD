"""
Simple end-to-end workflow for SAR-based ground deformation detection.

This script demonstrates the complete SAR GRD workflow from data download to visualization.
Uses intensity-based change detection (not interferometric phase).
"""

import logging
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sarGSD.config import Config
from sarGSD.download import SentinelDownloader
from sarGSD.cubes import create_datacube_from_files, save_datacube, load_datacube
from sarGSD.timeseries import detect_linear_trends, coherence_analysis
from sarGSD.deformation import identify_deformation_hotspots, classify_deformation_zones
from sarGSD.visualization import plot_velocity_map, plot_coherence_map, plot_timeseries

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Run complete SAR-based ground deformation detection workflow."""
    
    logger.info("=" * 60)
    logger.info("SAR Ground Deformation Detection - Example Workflow")
    logger.info("=" * 60)
    
    # Configuration
    study_area = Config.get_study_area_config("test_area")
    if not study_area:
        logger.error("Study area configuration not found")
        return
    
    bbox = study_area['bbox']
    start_date = study_area['start_date']
    end_date = study_area['end_date']
    
    logger.info(f"Study Area: {study_area['name']}")
    logger.info(f"Bounding Box: {bbox}")
    logger.info(f"Date Range: {start_date} to {end_date}")
    
    # Step 1: Download Data (optional - comment out if data already downloaded)
    logger.info("\n" + "=" * 60)
    logger.info("Step 1: Data Download")
    logger.info("=" * 60)
    
    try:
        if not Config.validate_credentials():
            logger.warning("Copernicus credentials not configured. Skipping download.")
            logger.info("Please set credentials in .env file to enable data download")
            downloaded_files = []
        else:
            downloader = SentinelDownloader()
            
            # Query available products
            products_df = downloader.query_products(
                bbox=bbox,
                start_date=start_date,
                end_date=end_date,
                product_type="GRD"
            )
            
            logger.info(f"Found {len(products_df)} available products")
            
            # Download (limit to 5 for demo)
            if len(products_df) > 0:
                downloaded_files = downloader.download_products(
                    products_df,
                    max_products=5
                )
            else:
                downloaded_files = []
    except Exception as e:
        logger.error(f"Download failed: {e}")
        logger.info("Continuing with existing data if available...")
        downloaded_files = []
    
    # Step 2: Create Data Cube
    logger.info("\n" + "=" * 60)
    logger.info("Step 2: Create Data Cube")
    logger.info("=" * 60)
    
    datacube_path = Config.PROCESSED_DATA_DIR / "example_datacube.nc"
    
    if downloaded_files and len(downloaded_files) >= 3:
        # Create from downloaded files
        logger.info(f"Creating datacube from {len(downloaded_files)} files")
        
        try:
            datacube = create_datacube_from_files(
                file_paths=downloaded_files,
                band_name="intensity"
            )
            
            # Save datacube
            save_datacube(datacube, datacube_path)
            logger.info(f"Datacube saved to {datacube_path}")
            
        except Exception as e:
            logger.error(f"Failed to create datacube: {e}")
            logger.info("Creating synthetic data for demonstration...")
            datacube = create_synthetic_datacube()
    else:
        logger.info("Insufficient data files. Creating synthetic datacube for demonstration...")
        datacube = create_synthetic_datacube()
        save_datacube(datacube, datacube_path)
    
    logger.info(f"Datacube shape: {datacube.shape}")
    logger.info(f"Time steps: {len(datacube.time)}")
    
    # Step 3: Time Series Analysis
    logger.info("\n" + "=" * 60)
    logger.info("Step 3: Time Series Analysis")
    logger.info("=" * 60)
    
    # Detect linear trends
    logger.info("Detecting linear deformation trends...")
    velocity, pvalue, rsquared = detect_linear_trends(datacube, robust=True)
    
    logger.info(f"Mean velocity: {velocity.mean().values:.2f} mm/year")
    logger.info(f"Velocity range: [{velocity.min().values:.2f}, {velocity.max().values:.2f}] mm/year")
    
    # Compute coherence
    logger.info("Computing temporal coherence...")
    coherence = coherence_analysis(datacube, window_size=3)
    
    # Step 4: Deformation Classification
    logger.info("\n" + "=" * 60)
    logger.info("Step 4: Deformation Classification")
    logger.info("=" * 60)
    
    # Classify deformation zones
    deformation_classes = classify_deformation_zones(velocity)
    
    # Identify hotspots
    hotspots, hotspot_stats = identify_deformation_hotspots(
        velocity,
        threshold=15.0,
        min_size=5
    )
    
    logger.info(f"Identified {len(hotspot_stats)} deformation hotspots")
    if len(hotspot_stats) > 0:
        logger.info("\nTop 3 hotspots:")
        for idx, row in hotspot_stats.head(3).iterrows():
            logger.info(f"  Hotspot {row['hotspot_id']}: "
                       f"{row['size_pixels']} pixels, "
                       f"mean velocity: {row['mean_velocity_mm_year']:.2f} mm/year")
    
    # Step 5: Visualization
    logger.info("\n" + "=" * 60)
    logger.info("Step 5: Visualization")
    logger.info("=" * 60)
    
    output_dir = Config.OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot velocity map
    logger.info("Creating velocity map...")
    plot_velocity_map(
        velocity,
        save_path=output_dir / "velocity_map.png",
        title="Deformation Velocity Map"
    )
    
    # Plot coherence map
    logger.info("Creating coherence map...")
    plot_coherence_map(
        coherence,
        save_path=output_dir / "coherence_map.png"
    )
    
    # Plot time series for center pixel
    logger.info("Creating time series plot...")
    center_y, center_x = datacube.shape[1] // 2, datacube.shape[2] // 2
    plot_timeseries(
        datacube,
        pixel_coords=(center_y, center_x),
        save_path=output_dir / "timeseries_center.png",
        title=f"Time Series at Center Pixel ({center_y}, {center_x})"
    )
    
    logger.info(f"\nAll outputs saved to: {output_dir}")
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Workflow Complete!")
    logger.info("=" * 60)
    logger.info(f"Processed {len(datacube.time)} time steps")
    logger.info(f"Detected {len(hotspot_stats)} deformation hotspots")
    logger.info(f"Mean coherence: {coherence.mean().values:.3f}")
    logger.info(f"Outputs saved to: {output_dir}")


def create_synthetic_datacube():
    """Create synthetic datacube for demonstration purposes."""
    import numpy as np
    import xarray as xr
    from datetime import datetime, timedelta
    import rasterio
    
    logger.info("Creating synthetic datacube...")
    
    # Dimensions
    n_times = 12
    ny, nx = 100, 100
    
    # Create synthetic data with deformation signal
    np.random.seed(42)
    
    # Base intensity
    base = np.random.randn(ny, nx) * 10 + 100
    
    # Add deformation zone (subsidence)
    y_center, x_center = ny // 2, nx // 2
    y_grid, x_grid = np.ogrid[:ny, :nx]
    distance = np.sqrt((y_grid - y_center)**2 + (x_grid - x_center)**2)
    deformation_mask = distance < 20
    
    # Time series with linear deformation
    dates = [datetime(2023, 1, 1) + timedelta(days=30*i) for i in range(n_times)]
    arrays = []
    
    for i in range(n_times):
        # Add temporal deformation
        deformation = np.zeros((ny, nx))
        deformation[deformation_mask] = -i * 2  # 2 mm/month subsidence
        
        # Add noise
        noise = np.random.randn(ny, nx) * 5
        
        # Combine
        image = base + deformation + noise
        arrays.append(image)
    
    # Create transform (fake coordinates)
    transform = rasterio.Affine(0.001, 0, -8.5, 0, -0.001, 40.7)
    
    # Create datacube
    from sarGSD.cubes import create_datacube_from_arrays
    datacube = create_datacube_from_arrays(
        arrays=arrays,
        dates=dates,
        transform=transform,
        crs="EPSG:4326",
        band_name="intensity"
    )
    
    logger.info("Synthetic datacube created with simulated deformation")
    
    return datacube


if __name__ == "__main__":
    main()
