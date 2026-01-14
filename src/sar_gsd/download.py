"""
Data access & download module for Sentinel-1 SAR imagery.

This module provides functionality to search and download Sentinel-1 GRD products
from the Copernicus Data Space Ecosystem.
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import warnings

try:
    from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt
    SENTINELSAT_AVAILABLE = True
except ImportError:
    SENTINELSAT_AVAILABLE = False
    warnings.warn("sentinelsat not installed. Install with: pip install sentinelsat")

import geopandas as gpd
from shapely.geometry import box, Polygon
import pandas as pd
from tqdm import tqdm

from .config import Config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentinelDownloader:
    """
    Download Sentinel-1 SAR data from Copernicus Data Space Ecosystem.
    
    Attributes:
        api: SentinelAPI instance for data access
        download_dir: Directory for downloaded products
    """
    
    def __init__(
        self, 
        username: Optional[str] = None, 
        password: Optional[str] = None,
        download_dir: Optional[Path] = None
    ):
        """
        Initialize Sentinel downloader.
        
        Args:
            username: Copernicus username (defaults to config)
            password: Copernicus password (defaults to config)
            download_dir: Download directory (defaults to Config.RAW_DATA_DIR)
        """
        if not SENTINELSAT_AVAILABLE:
            raise ImportError(
                "sentinelsat is required for data download. "
                "Install with: pip install sentinelsat"
            )
        
        self.username = username or Config.COPERNICUS_USERNAME
        self.password = password or Config.COPERNICUS_PASSWORD
        
        if not self.username or not self.password:
            raise ValueError(
                "Copernicus credentials not found. "
                "Set COPERNICUS_USERNAME and COPERNICUS_PASSWORD in .env file"
            )
        
        self.download_dir = download_dir or Config.RAW_DATA_DIR
        self.download_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize SentinelAPI
        # Note: Copernicus Data Space uses different endpoint than old SciHub
        self.api = SentinelAPI(
            self.username, 
            self.password,
            'https://catalogue.dataspace.copernicus.eu/resto'
        )
        
        logger.info(f"Initialized SentinelDownloader. Download dir: {self.download_dir}")
    
    def query_products(
        self,
        bbox: List[float],
        start_date: str,
        end_date: str,
        product_type: str = "GRD",
        sensor_mode: str = "IW",
        polarization: Optional[str] = None,
        orbit_direction: Optional[str] = None,
        max_cloud_cover: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Query available Sentinel-1 products.
        
        Args:
            bbox: Bounding box [min_lon, min_lat, max_lon, max_lat]
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            product_type: Product type (GRD or SLC)
            sensor_mode: Sensor mode (IW, EW, SM)
            polarization: Polarization (VV, VH, HH, HV)
            orbit_direction: Orbit direction (ASCENDING, DESCENDING)
            max_cloud_cover: Maximum cloud cover percentage (for optical, not SAR)
            
        Returns:
            DataFrame with product metadata
        """
        # Create footprint from bbox
        footprint = box(bbox[0], bbox[1], bbox[2], bbox[3])
        footprint_wkt = footprint.wkt
        
        # Build query parameters
        query_kwargs = {
            'platformname': 'Sentinel-1',
            'producttype': product_type,
            'sensoroperationalmode': sensor_mode,
            'date': (start_date, end_date),
        }
        
        if polarization:
            query_kwargs['polarisationmode'] = polarization
        
        if orbit_direction:
            query_kwargs['orbitdirection'] = orbit_direction
        
        logger.info(f"Querying products from {start_date} to {end_date}")
        logger.info(f"Area: {bbox}")
        
        # Query API
        products = self.api.query(footprint_wkt, **query_kwargs)
        
        if not products:
            logger.warning("No products found matching criteria")
            return pd.DataFrame()
        
        # Convert to DataFrame
        products_df = self.api.to_dataframe(products)
        
        logger.info(f"Found {len(products_df)} products")
        
        return products_df
    
    def download_products(
        self,
        products_df: pd.DataFrame,
        max_products: Optional[int] = None,
        checksum: bool = True
    ) -> List[Path]:
        """
        Download Sentinel-1 products.
        
        Args:
            products_df: DataFrame from query_products()
            max_products: Maximum number of products to download
            checksum: Verify checksums after download
            
        Returns:
            List of downloaded file paths
        """
        if products_df.empty:
            logger.warning("No products to download")
            return []
        
        # Limit number of products
        if max_products:
            products_df = products_df.head(max_products)
        
        logger.info(f"Downloading {len(products_df)} products to {self.download_dir}")
        
        downloaded_files = []
        
        # Download with progress bar
        for idx, (product_id, product_info) in enumerate(
            tqdm(products_df.iterrows(), total=len(products_df), desc="Downloading")
        ):
            try:
                # Download product
                product_path = self.api.download(
                    product_id,
                    directory_path=str(self.download_dir),
                    checksum=checksum
                )
                
                downloaded_files.append(Path(product_path['path']))
                logger.info(f"Downloaded: {product_info['title']}")
                
            except Exception as e:
                logger.error(f"Failed to download {product_id}: {e}")
                continue
        
        logger.info(f"Successfully downloaded {len(downloaded_files)} products")
        
        return downloaded_files
    
    def get_product_metadata(self, products_df: pd.DataFrame) -> gpd.GeoDataFrame:
        """
        Extract and format product metadata as GeoDataFrame.
        
        Args:
            products_df: DataFrame from query_products()
            
        Returns:
            GeoDataFrame with product metadata and footprints
        """
        if products_df.empty:
            return gpd.GeoDataFrame()
        
        # Create GeoDataFrame with footprints
        gdf = gpd.GeoDataFrame(
            products_df,
            geometry=products_df['footprint'].apply(lambda x: Polygon(x) if x else None),
            crs='EPSG:4326'
        )
        
        return gdf
    
    def download_by_area_and_dates(
        self,
        bbox: List[float],
        start_date: str,
        end_date: str,
        max_products: Optional[int] = None,
        **query_kwargs
    ) -> Tuple[pd.DataFrame, List[Path]]:
        """
        Convenience method to query and download in one step.
        
        Args:
            bbox: Bounding box [min_lon, min_lat, max_lon, max_lat]
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            max_products: Maximum number of products to download
            **query_kwargs: Additional query parameters
            
        Returns:
            Tuple of (products DataFrame, list of downloaded file paths)
        """
        # Query products
        products_df = self.query_products(
            bbox=bbox,
            start_date=start_date,
            end_date=end_date,
            **query_kwargs
        )
        
        if products_df.empty:
            return products_df, []
        
        # Download products
        downloaded_files = self.download_products(
            products_df=products_df,
            max_products=max_products
        )
        
        return products_df, downloaded_files


def download_sentinel_data(
    bbox: List[float],
    start_date: str,
    end_date: str,
    output_dir: Optional[Path] = None,
    max_products: int = 10,
    product_type: str = "GRD"
) -> List[Path]:
    """
    Simple function to download Sentinel-1 data.
    
    Args:
        bbox: Bounding box [min_lon, min_lat, max_lon, max_lat]
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        output_dir: Output directory
        max_products: Maximum number of products
        product_type: GRD or SLC
        
    Returns:
        List of downloaded file paths
    """
    downloader = SentinelDownloader(download_dir=output_dir)
    
    products_df, files = downloader.download_by_area_and_dates(
        bbox=bbox,
        start_date=start_date,
        end_date=end_date,
        max_products=max_products,
        product_type=product_type
    )
    
    return files


def main():
    """Command-line interface for downloading data."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download Sentinel-1 SAR data")
    parser.add_argument("--bbox", nargs=4, type=float, required=True,
                       help="Bounding box: min_lon min_lat max_lon max_lat")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--max", type=int, default=10, help="Max products to download")
    parser.add_argument("--type", default="GRD", choices=["GRD", "SLC"],
                       help="Product type")
    parser.add_argument("--output", type=Path, help="Output directory")
    
    args = parser.parse_args()
    
    files = download_sentinel_data(
        bbox=args.bbox,
        start_date=args.start,
        end_date=args.end,
        output_dir=args.output,
        max_products=args.max,
        product_type=args.type
    )
    
    print(f"\nDownloaded {len(files)} files:")
    for f in files:
        print(f"  {f}")


if __name__ == "__main__":
    main()

