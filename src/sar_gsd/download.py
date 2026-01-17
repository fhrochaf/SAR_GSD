"""
Data access & download module for Sentinel-1 SAR imagery.

This module provides functionality to search and download Sentinel-1 GRD products
from the Copernicus Data Space Ecosystem using the OData API.
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import os
import requests
import warnings
import json
import zipfile
import tempfile
import shutil

import geopandas as gpd
from shapely.geometry import box, Polygon
import pandas as pd
from tqdm import tqdm
import rasterio
from rasterio.mask import mask as rio_mask
import rioxarray as rxr

from .config import Config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CDSEDownloader:
    """
    Download Sentinel-1 SAR data from Copernicus Data Space Ecosystem using OData API.
    
    Attributes:
        username: Copernicus username
        download_dir: Directory for downloaded products
    """
    
    def __init__(
        self, 
        username: Optional[str] = None, 
        password: Optional[str] = None,
        download_dir: Optional[Path] = None
    ):
        """
        Initialize CDSE downloader.
        
        Args:
            username: Copernicus username (defaults to config)
            password: Copernicus password (defaults to config)
            download_dir: Download directory (defaults to Config.RAW_DATA_DIR)
        """
        self.username = username or Config.COPERNICUS_USERNAME
        self.password = password or Config.COPERNICUS_PASSWORD
        
        if not self.username or not self.password:
            raise ValueError(
                "Copernicus credentials not found. "
                "Set COPERNICUS_USERNAME and COPERNICUS_PASSWORD in .env file"
            )
        
        self.download_dir = download_dir or Config.RAW_DATA_DIR
        self.download_dir.mkdir(parents=True, exist_ok=True)
        
        self.access_token = self._get_access_token()
        self.session = requests.Session()
        self.session.headers.update({'Authorization': f'Bearer {self.access_token}'})
        
        logger.info(f"Initialized CDSEDownloader. Download dir: {self.download_dir}")

    def _get_access_token(self) -> str:
        """Obtain Keycloak access token."""
        token_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
        data = {
            "client_id": "cdse-public",
            "username": self.username,
            "password": self.password,
            "grant_type": "password",
        }
        try:
            response = requests.post(token_url, data=data)
            response.raise_for_status()
            return response.json()["access_token"]
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Failed to authenticate with CDSE: {e}")

    def query_products(
        self,
        bbox: List[float],
        start_date: str,
        end_date: str,
        product_type: str = "GRD",
        sensor_mode: str = "IW",
        polarization: Optional[str] = None,  # Kept for compatibility, mostly not used in basic OData filter yet
        orbit_direction: Optional[str] = None, # Kept for compatibility
        max_cloud_cover: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Query available Sentinel-1 products using OData API.
        
        Args:
            bbox: Bounding box [min_lon, min_lat, max_lon, max_lat]
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            product_type: Product type (GRD or SLC)
            sensor_mode: Sensor mode (IW, EW, SM)
            polarization: Polarization (VV, VH, HH, HV) (Optional filter)
            orbit_direction: Orbit direction (ASCENDING, DESCENDING) (Optional filter)
            max_cloud_cover: Not applicable for SAR
            
        Returns:
            DataFrame with product metadata
        """
        # Ensure dates are in ISO format with time
        def ensure_iso(d, is_end=False):
            if len(d) == 10:
                time_part = "T23:59:59.999Z" if is_end else "T00:00:00.000Z"
                return f"{d}{time_part}"
            return d

        start_iso = ensure_iso(start_date)
        end_iso = ensure_iso(end_date, is_end=True)
        
        # OData Filter construction
        # Geography
        # bbox is [min_lon, min_lat, max_lon, max_lat]
        # Polygon expects counter-clockwise points, closed loop
        p1 = f"{bbox[0]} {bbox[1]}"
        p2 = f"{bbox[2]} {bbox[1]}"
        p3 = f"{bbox[2]} {bbox[3]}"
        p4 = f"{bbox[0]} {bbox[3]}"
        p5 = f"{bbox[0]} {bbox[1]}" # Close loop
        polygon_str = f"{p1},{p2},{p3},{p4},{p5}"
        
        filter_query = (
            f"OData.CSC.Intersects(area=geography'SRID=4326;POLYGON(({polygon_str}))') and "
            f"ContentDate/Start ge {start_iso} and "
            f"ContentDate/Start le {end_iso} and "
            f"startswith(Name,'S1') and "
            f"Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType' and att/Value eq '{product_type}') and "
            f"Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'operationalMode' and att/Value eq '{sensor_mode}')"
        )
        
        if orbit_direction:
             filter_query += f" and Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'orbitDirection' and att/Value eq '{orbit_direction.upper()}')"
        
        if polarization:
             # Polarization in OData attributes can be tricky, typically "polarisationChannels"
             # Simplified check might be needed or strict check if we assume exact API response structure
             # For now, we omit to prevent errors if the attribute name differs, or check logic:
             pass 

        base_url = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
        params = {
            "$filter": filter_query,
            "$orderby": "ContentDate/Start asc",
            "$top": 1000  # Page size
        }

        logger.info(f"Querying CDSE from {start_date} to {end_date}")
        
        try:
            response = self.session.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()
            products = data.get('value', [])
            
            if not products:
                logger.warning("No products found matching criteria")
                return pd.DataFrame()
            
            # Map OData JSON to DataFrame expected structure
            # We need to extract geometry, id, name, etc.
            
            parsed_products = []
            for idx, p in enumerate(products):
                # Extract footprint
                # The API returns GeoJSON in 'GeoFootprint' usually, or we parse the complex footprint object
                # For `sentinelsat` compatibility we need 'uuid' (Id), 'title' (Name), 'footprint' (WKT or Geom)
                
                # Try to get footprint
                footprint_geom = p.get('GeoFootprint', {}).get('geometry', None) # GeoJSON
                footprint_wkt = None
                if footprint_geom:
                    # Convert GeoJSON to shape and then WKT if needed, or keep as geom object for now
                    # We will use WKT for DataFrame compatibility if we want to follow old structure
                    # But better to use Geometry directly in next step.
                    pass
                
                parsed_products.append({
                    'uuid': p.get('Id'),
                    'title': p.get('Name'),
                    'identifier': p.get('Name'),
                    'start_date': p.get('ContentDate', {}).get('Start'),
                    'end_date': p.get('ContentDate', {}).get('End'),
                    'footprint': footprint_geom, # Keep as dict (GeoJSON geometry)
                    'size_bytes': p.get('ContentLength'),
                    # Add more fields as necessary
                    'product_type': product_type,
                    'sensor_mode': sensor_mode
                })
                
            df = pd.DataFrame(parsed_products)
            logger.info(f"Found {len(df)} products")
            return df
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Query failed: {e}")
            return pd.DataFrame()

    def _clip_product_to_bbox(
        self,
        safe_zip_path: Path,
        bbox: List[float],
        polarization: str = 'vv',
        cleanup_original: bool = True
    ) -> Optional[Path]:
        """
        Extract, clip, and save a Sentinel-1 product to bounding box.

        Args:
            safe_zip_path: Path to .SAFE.zip file
            bbox: Bounding box [min_lon, min_lat, max_lon, max_lat]
            polarization: Polarization to extract ('vv' or 'vh')
            cleanup_original: Delete original zip after clipping

        Returns:
            Path to clipped GeoTIFF, or None if clipping failed
        """
        try:
            # Create temporary extraction directory
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)

                # Extract SAFE archive
                logger.info(f"  Extracting {safe_zip_path.name}...")
                with zipfile.ZipFile(safe_zip_path, 'r') as zf:
                    zf.extractall(temp_path)

                # Find .SAFE directory
                safe_dirs = list(temp_path.glob('*.SAFE'))
                if not safe_dirs:
                    logger.warning(f"  No .SAFE directory found in {safe_zip_path.name}")
                    return None

                safe_dir = safe_dirs[0]

                # Find measurement TIFF
                measurement_dir = safe_dir / 'measurement'
                if not measurement_dir.exists():
                    logger.warning(f"  No measurement directory in {safe_dir.name}")
                    return None

                # Look for polarization-specific TIFF
                pattern = f'*-{polarization.lower()}-*.tiff'
                tiff_files = list(measurement_dir.glob(pattern))

                if not tiff_files:
                    # Try alternative pattern
                    pattern = f'*{polarization.upper()}*.tiff'
                    tiff_files = list(measurement_dir.glob(pattern))

                if not tiff_files:
                    logger.warning(f"  No TIFF with polarization {polarization} found")
                    return None

                tiff_path = tiff_files[0]
                logger.info(f"  Clipping {tiff_path.name} to bounding box...")

                # Open with rioxarray and clip to bbox
                da = rxr.open_rasterio(tiff_path, masked=True)

                # Clip to bounding box
                # bbox format: [min_lon, min_lat, max_lon, max_lat]
                clipped = da.rio.clip_box(
                    minx=bbox[0],
                    miny=bbox[1],
                    maxx=bbox[2],
                    maxy=bbox[3]
                )

                # Create output filename (replace .zip with _clipped.tif)
                output_name = safe_zip_path.stem.replace('.SAFE', f'_clipped_{polarization}.tif')
                output_path = safe_zip_path.parent / output_name

                # Save clipped raster
                logger.info(f"  Saving clipped product to {output_path.name}...")
                clipped.rio.to_raster(output_path, compress='LZW')

                # Cleanup original zip if requested
                if cleanup_original:
                    safe_zip_path.unlink()
                    logger.info(f"  Deleted original zip: {safe_zip_path.name}")

                return output_path

        except Exception as e:
            logger.error(f"  Failed to clip {safe_zip_path.name}: {e}")
            return None

    def download_products(
        self,
        products_df: pd.DataFrame,
        max_products: Optional[int] = None,
        clip_to_bbox: bool = False,
        bbox: Optional[List[float]] = None,
        polarization: str = 'vv',
        checksum: bool = True # Not fully implemented for CDSE yet
    ) -> List[Path]:
        """
        Download Sentinel-1 products using CDSE API.

        Args:
            products_df: DataFrame from query_products()
            max_products: Maximum number of products to download
            clip_to_bbox: If True, clip products to bounding box after download
            bbox: Bounding box [min_lon, min_lat, max_lon, max_lat] (required if clip_to_bbox=True)
            polarization: Polarization to extract when clipping ('vv' or 'vh')
            checksum: Verify checksums (Placeholder)

        Returns:
            List of downloaded file paths (clipped GeoTIFFs if clip_to_bbox=True, else .zip files)
        """
        if products_df.empty:
            logger.warning("No products to download")
            return []

        # Validate bbox if clipping is requested
        if clip_to_bbox and bbox is None:
            raise ValueError("bbox is required when clip_to_bbox=True")

        # Limit number of products
        if max_products:
            products_df = products_df.head(max_products)

        logger.info(f"Downloading {len(products_df)} products to {self.download_dir}")
        if clip_to_bbox:
            logger.info(f"Products will be clipped to bbox: {bbox}")

        downloaded_files = []

        for idx, row in tqdm(products_df.iterrows(), total=len(products_df), desc="Downloading"):
            product_id = row['uuid']
            title = row['title']
            filename = f"{title}.zip"
            filepath = self.download_dir / filename

            # Check if clipped version already exists
            if clip_to_bbox:
                clipped_name = title.replace('.SAFE', f'_clipped_{polarization}.tif')
                clipped_path = self.download_dir / clipped_name
                if clipped_path.exists():
                    logger.info(f"Clipped file {clipped_name} already exists, skipping.")
                    downloaded_files.append(clipped_path)
                    continue

            if filepath.exists():
                # Validate that existing file is a valid zip before skipping
                try:
                    with zipfile.ZipFile(filepath, 'r') as zf:
                        # Test zip integrity
                        if zf.testzip() is None:
                            logger.info(f"File {filename} exists and is valid, skipping.")
                            downloaded_files.append(filepath)
                            continue
                        else:
                            logger.warning(f"File {filename} exists but is corrupted, re-downloading.")
                            filepath.unlink()
                except zipfile.BadZipFile:
                    logger.warning(f"File {filename} exists but is not a valid zip file, re-downloading.")
                    filepath.unlink()
                except Exception as e:
                    logger.warning(f"File {filename} validation failed ({e}), re-downloading.")
                    filepath.unlink()
                
            # Download URL
            # https://catalogue.dataspace.copernicus.eu/odata/v1/Products(uuid)/$value
            download_url = f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products({product_id})/$value"
            
            try:
                # Handle redirects manually to preserve Authorization header
                # requests strips Authorization on redirect to different domain by default
                resp = self.session.get(download_url, stream=True, allow_redirects=False)
                
                if 300 <= resp.status_code < 400:
                   redirect_url = resp.headers['Location']
                   # Create new request to redirect url with same headers
                   resp = self.session.get(redirect_url, stream=True, allow_redirects=True)
                
                with resp as r:
                    r.raise_for_status()
                    total_size = int(r.headers.get('content-length', 0))
                    
                    with open(filepath, 'wb') as f, tqdm(
                        desc=title,
                        total=total_size,
                        unit='iB',
                        unit_scale=True,
                        unit_divisor=1024,
                    ) as bar:
                        for chunk in r.iter_content(chunk_size=8192):
                            size = f.write(chunk)
                            bar.update(size)

                logger.info(f"Downloaded: {title}")

                # Clip to bbox if requested
                if clip_to_bbox:
                    logger.info(f"Clipping {title} to bounding box...")
                    clipped_path = self._clip_product_to_bbox(
                        filepath,
                        bbox,
                        polarization=polarization,
                        cleanup_original=True
                    )
                    if clipped_path:
                        downloaded_files.append(clipped_path)
                    else:
                        logger.warning(f"Failed to clip {title}, skipping.")
                else:
                    downloaded_files.append(filepath)

            except Exception as e:
                logger.error(f"Failed to download {title}: {e}")
                # Cleanup partial file
                if filepath.exists():
                    filepath.unlink()
                continue

        return downloaded_files

    def get_product_metadata(self, products_df: pd.DataFrame) -> gpd.GeoDataFrame:
        """
        Format product metadata as GeoDataFrame.
        
        Args:
            products_df: DataFrame from query_products()
            
        Returns:
            GeoDataFrame with product metadata and footprints
        """
        if products_df.empty:
            return gpd.GeoDataFrame()
        
        # Convert GeoJSON dicts in 'footprint' column to shapely Polygons
        try:
             geoms = products_df['footprint'].apply(
                lambda x: Polygon(x['coordinates'][0]) if x and 'coordinates' in x else None
            )
        except Exception:
             # Fallback if structure is different
             geoms = None

        gdf = gpd.GeoDataFrame(
            products_df,
            geometry=geoms,
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
        Convenience method to query and download.
        """
        products_df = self.query_products(
            bbox=bbox,
            start_date=start_date,
            end_date=end_date,
            **query_kwargs
        )
        
        if products_df.empty:
            return products_df, []
        
        files = self.download_products(products_df, max_products=max_products)
        return products_df, files


def download_sentinel_data(
    bbox: List[float],
    start_date: str,
    end_date: str,
    output_dir: Optional[Path] = None,
    max_products: int = 10,
    product_type: str = "GRD"
) -> List[Path]:
    """
    Simple function to download Sentinel-1 data using CDSE.
    """
    downloader = CDSEDownloader(download_dir=output_dir)
    
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
    
    parser = argparse.ArgumentParser(description="Download Sentinel-1 SAR data (CDSE)")
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
