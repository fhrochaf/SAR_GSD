# UNIT TESTS FOR SAR CHANGE DETECTION PIPELINE

import unittest
import numpy as np
import torch
import xarray as xr
import geopandas as gpd
import pandas as pd
from shapely.geometry import box
from rasterio import features
from rasterio.transform import from_bounds
from scipy.ndimage import gaussian_filter
import time

print("UNIT TESTS FOR SAR CHANGE DETECTION PIPELINE")


class TestSARProcessing(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)

        self.n_times = 10
        self.height = 50
        self.width = 50

        data = np.random.rand(
            self.n_times, self.height, self.width
        ).astype(np.float32)

        trend = np.linspace(0, 0.5, self.n_times)
        data[:, 20:30, 20:30] += trend[:, None, None]

        self.datacube = xr.Dataset(
            {"backscatter": (["time", "y", "x"], data)},
            coords={
                "time": pd.date_range(
                    "2020-01-01", periods=self.n_times, freq="ME"
                ),
                "y": np.arange(self.height),
                "x": np.arange(self.width),
            },
        )

    # ------------------------------------------------------------------
    def test_1_numpy_array_operations(self):
        sar = self.datacube["backscatter"].values
        self.assertEqual(sar.shape, (10, 50, 50))

        norm = (sar - sar.min()) / (sar.max() - sar.min())
        self.assertTrue(norm.min() >= 0)
        self.assertTrue(norm.max() <= 1)

        temporal_mean = np.mean(sar, axis=0)
        self.assertEqual(temporal_mean.shape, (50, 50))

    # ------------------------------------------------------------------
    def test_2_tensor_operations(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        arr = self.datacube["backscatter"][0].values
        tensor = torch.from_numpy(arr.astype(np.float32)).to(device)

        self.assertEqual(tensor.shape, torch.Size([50, 50]))
        self.assertEqual(tensor.device.type, device)

        kernel = torch.ones(1, 1, 3, 3, device=device) / 9.0
        out = torch.nn.functional.conv2d(
            tensor.unsqueeze(0).unsqueeze(0),
            kernel,
            padding=1,
        )

        self.assertEqual(out.squeeze().shape, tensor.shape)

    # ------------------------------------------------------------------
    def test_3_tensor_linear_regression(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        n = 10
        true_slope = 0.5
        true_intercept = 1.0

        t = torch.arange(n, dtype=torch.float32, device=device).view(-1, 1, 1)
        ts = true_slope * t + true_intercept
        ts = ts.expand(-1, 5, 5)

        t_mean = t.mean()
        y_mean = ts.mean(dim=0)

        slope = ((ts - y_mean) * (t - t_mean)).sum(dim=0) / (
            (t - t_mean) ** 2
        ).sum()

        self.assertAlmostEqual(slope[0, 0].item(), true_slope, places=2)

    # ------------------------------------------------------------------
    def test_4_vector_raster_integration(self):
        geom = box(10, 10, 40, 40)
        gdf = gpd.GeoDataFrame({"geometry": [geom]}, crs="EPSG:4326")

        transform = from_bounds(0, 0, 50, 50, 50, 50)
        mask = features.geometry_mask(
            gdf.geometry,
            out_shape=(50, 50),
            transform=transform,
            invert=True,
        )

        self.assertEqual(mask.shape, (50, 50))
        self.assertTrue(mask.sum() > 0)

    # ------------------------------------------------------------------
    def test_5_datacube_operations(self):
        dc = self.datacube

        self.assertIn("time", dc.dims)
        self.assertIn("x", dc.dims)
        self.assertIn("y", dc.dims)

        sub = dc.isel(x=slice(10, 40), y=slice(10, 40))
        self.assertEqual(sub.sizes["x"], 30)
        self.assertEqual(sub.sizes["y"], 30)

        mean_time = dc["backscatter"].mean(dim="time")
        self.assertEqual(mean_time.shape, (50, 50))

    # ------------------------------------------------------------------
    def test_6_change_detection(self):
        trend = np.random.randn(50, 50) * 0.05
        trend[10:20, 10:20] = 0.15
        trend[30:40, 30:40] = -0.15

        threshold = 0.07
        pos = trend > threshold
        neg = trend < -threshold

        self.assertTrue(pos.sum() > 0)
        self.assertTrue(neg.sum() > 0)
        self.assertEqual((pos & neg).sum(), 0)

    # ------------------------------------------------------------------
    def test_7_performance_and_memory(self):
        img = np.random.rand(300, 300).astype(np.float32)

        start = time.time()
        ref = gaussian_filter(img, sigma=2.0)
        t_numpy = time.time() - start

        device = "cuda" if torch.cuda.is_available() else "cpu"
        t_img = torch.from_numpy(img).to(device)

        k = 7
        x = torch.arange(-(k // 2), k // 2 + 1, device=device)
        xx, yy = torch.meshgrid(x, x, indexing="ij")
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * 2.0**2))
        kernel = (kernel / kernel.sum()).view(1, 1, k, k)

        start = time.time()
        out = torch.nn.functional.conv2d(
            t_img.unsqueeze(0).unsqueeze(0),
            kernel,
            padding=k // 2,
        )
        if device == "cuda":
            torch.cuda.synchronize()
        t_torch = time.time() - start

        corr = np.corrcoef(
            ref.ravel(),
            out.squeeze().cpu().numpy().ravel()
        )[0, 1]
        self.assertGreater(corr, 0.80)


# RUN TESTS

suite = unittest.TestLoader().loadTestsFromTestCase(TestSARProcessing)
runner = unittest.TextTestRunner(verbosity=2)
result = runner.run(suite)

print("UNIT TEST SUMMARY")
print(f"Total tests run : {result.testsRun}")
print(f"Tests passed   : {result.testsRun - len(result.failures) - len(result.errors)}")
print(f"Tests failed   : {len(result.failures)}")
print(f"Errors         : {len(result.errors)}")

if result.wasSuccessful():
    print("STATUS: ALL TESTS PASSED SUCCESSFULLY")
else:
    print("STATUS: TEST FAILURES OR ERRORS DETECTED")

