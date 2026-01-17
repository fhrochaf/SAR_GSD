"""
Tests for the unzip functionality in cubes module.
"""

import pytest
from pathlib import Path
import zipfile
import tempfile
import shutil

from sar_gsd.cubes import _extract_safe_zip, _find_tiff_in_safe


def test_extract_safe_zip():
    """Test extraction of SAFE zip file."""
    # Create a mock .SAFE.zip file for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create a mock SAFE structure
        safe_name = "S1B_IW_GRDH_1SDV_20180110T082038_20180110T082103_009106_01049F_3885.SAFE"
        safe_dir = tmpdir / safe_name
        measurement_dir = safe_dir / "measurement"
        measurement_dir.mkdir(parents=True)

        # Create a dummy TIFF file
        tiff_file = measurement_dir / "s1b-iw-grd-vv-20180110t082038-20180110t082103-009106-01049f-002.tiff"
        tiff_file.write_text("dummy tiff content")

        # Create zip file
        zip_path = tmpdir / f"{safe_name}.zip"
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            zipf.write(tiff_file, arcname=f"{safe_name}/measurement/{tiff_file.name}")

        # Remove the original directory
        shutil.rmtree(safe_dir)

        # Test extraction
        extract_dir = tmpdir / "extracted"
        extracted_safe = _extract_safe_zip(zip_path, extract_dir)

        assert extracted_safe.exists()
        assert extracted_safe.name == safe_name
        assert (extracted_safe / "measurement" / tiff_file.name).exists()


def test_find_tiff_in_safe():
    """Test finding TIFF file within SAFE directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create mock SAFE structure
        safe_dir = tmpdir / "S1B_TEST.SAFE"
        measurement_dir = safe_dir / "measurement"
        measurement_dir.mkdir(parents=True)

        # Create dummy VV and VH TIFF files
        vv_tiff = measurement_dir / "s1b-iw-grd-vv-20180110t082038-20180110t082103-009106-01049f-002.tiff"
        vh_tiff = measurement_dir / "s1b-iw-grd-vh-20180110t082038-20180110t082103-009106-01049f-001.tiff"
        vv_tiff.write_text("vv data")
        vh_tiff.write_text("vh data")

        # Test finding VV polarization
        found_vv = _find_tiff_in_safe(safe_dir, polarization='vv')
        assert found_vv is not None
        assert 'vv' in found_vv.name.lower()

        # Test finding VH polarization
        found_vh = _find_tiff_in_safe(safe_dir, polarization='vh')
        assert found_vh is not None
        assert 'vh' in found_vh.name.lower()


def test_find_tiff_nonexistent_polarization():
    """Test finding TIFF with non-existent polarization."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create mock SAFE structure with only VV
        safe_dir = tmpdir / "S1B_TEST.SAFE"
        measurement_dir = safe_dir / "measurement"
        measurement_dir.mkdir(parents=True)

        vv_tiff = measurement_dir / "s1b-iw-grd-vv-test.tiff"
        vv_tiff.write_text("vv data")

        # Test finding non-existent VH
        found_vh = _find_tiff_in_safe(safe_dir, polarization='vh')
        assert found_vh is None


def test_find_tiff_no_measurement_dir():
    """Test behavior when measurement directory doesn't exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create SAFE directory without measurement subdirectory
        safe_dir = tmpdir / "S1B_TEST.SAFE"
        safe_dir.mkdir()

        # Should return None
        found = _find_tiff_in_safe(safe_dir, polarization='vv')
        assert found is None


if __name__ == "__main__":
    # Run tests if executed directly
    test_extract_safe_zip()
    test_find_tiff_in_safe()
    test_find_tiff_nonexistent_polarization()
    test_find_tiff_no_measurement_dir()
    print("All tests passed!")
