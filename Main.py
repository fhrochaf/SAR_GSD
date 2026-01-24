# import numpy as np
# import matplotlib.pyplot as plt
# from datetime import datetime, date
# import os
# import subprocess
# import requests

# from sentinelhub import (
#     SHConfig, SentinelHubRequest, SentinelHubCatalog,
#     DataCollection, BBox, CRS, bbox_to_dimensions, MimeType
# )

# from scipy.ndimage import gaussian_filter
# import torch
# import torch.nn.functional as F

# import rasterio
# from rasterio.features import shapes
# from shapely.geometry import shape

# import simplekml

# # ============================================================
# # CONFIGURATION
# # ============================================================

# CLIENT_ID = "2e2362c2-38da-4eea-80af-54d8f86a9869"
# CLIENT_SECRET = "DxlMQtk4PWMCeTAWT0NMo92KGVDmUy4S"

# AOI_BBOX = [-42.85, -19.65, -42.48, -19.40]  # lon_min, lat_min, lon_max, lat_max
# RESOLUTION = 20

# START_YEAR = "2017-01-01"
# MAX_DATES = 40

# CHANGE_THRESHOLD = 0.07  # log-intensity / year
# OUTPUT_KML = "sar_change_insar_validated.kml"

# # ============================================================
# # AUTHENTICATION
# # ============================================================

# config = SHConfig()
# config.sh_client_id = CLIENT_ID
# config.sh_client_secret = CLIENT_SECRET

# if not CLIENT_ID or not CLIENT_SECRET:
#     raise RuntimeError("Sentinel Hub credentials missing")

# # ============================================================
# # SENTINEL-1 AVAILABILITY
# # ============================================================

# def get_available_s1_dates(bbox, start_date):
#     catalog = SentinelHubCatalog(config=config)
#     search = catalog.search(
#         DataCollection.SENTINEL1_IW,
#         bbox=bbox,
#         time=(start_date, date.today().isoformat()),
#         fields={"include": ["properties.datetime"], "exclude": []},
#     )
#     return sorted({item["properties"]["datetime"][:10] for item in search})

# # ============================================================
# # SENTINEL-1 GRD REQUEST
# # ============================================================

# def request_s1(date_str, bbox, size):
#     evalscript = """
#     //VERSION=3
#     function setup() {
#         return { input: ["VV"], output: { bands: 1, sampleType: "FLOAT32" }};
#     }
#     function evaluatePixel(s) { return [s.VV]; }
#     """
#     req = SentinelHubRequest(
#         evalscript=evalscript,
#         input_data=[SentinelHubRequest.input_data(
#             DataCollection.SENTINEL1_IW,
#             time_interval=(date_str, date_str),
#             mosaicking_order="mostRecent"
#         )],
#         responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
#         bbox=bbox,
#         size=size,
#         config=config
#     )
#     return req.get_data()[0]

# # ============================================================
# # BUILD DATA CUBE
# # ============================================================

# bbox = BBox(AOI_BBOX, CRS.WGS84)
# size = bbox_to_dimensions(bbox, resolution=RESOLUTION)

# dates = get_available_s1_dates(bbox, START_YEAR)
# dates = dates[::max(1, len(dates)//MAX_DATES)]

# cube, valid_dates = [], []

# for d in dates:
#     img = request_s1(d, bbox, size).astype(np.float32)
#     img[img <= 0] = np.nan
#     if np.all(np.isnan(img)):
#         continue
#     cube.append(img / np.nanmedian(img))
#     valid_dates.append(d)

# if len(cube) < 3:
#     raise RuntimeError("Not enough valid data")

# cube = np.stack(cube)
# log_cube = np.log10(cube)

# # ============================================================
# # TREND ESTIMATION
# # ============================================================

# time = [datetime.fromisoformat(d) for d in valid_dates]
# t = np.array([(d - time[0]).days / 365.25 for d in time])

# trend = np.polyfit(
#     t,
#     log_cube.reshape(len(t), -1),
#     1
# )[0].reshape(log_cube.shape[1:])

# trend = gaussian_filter(trend, 1.5)

# trend_tensor = torch.from_numpy(trend).float()[None, None]
# kernel = torch.ones((1,1,3,3), dtype=torch.float32) / 9
# trend_final = F.conv2d(trend_tensor, kernel, padding=1).squeeze().numpy()

# # ============================================================
# # CHANGE MASKS
# # ============================================================

# positive_mask = trend_final > CHANGE_THRESHOLD
# negative_mask = trend_final < -CHANGE_THRESHOLD

# # ============================================================
# # FIGURE 1 — SAR INTENSITY
# # ============================================================

# plt.figure(figsize=(6,5))
# plt.title("SAR Intensity (latest acquisition)")
# plt.imshow(log_cube[-1], cmap="gray")
# plt.colorbar(label="log backscatter")
# plt.show()

# # ============================================================
# # FIGURE 2 — INTENSITY + CHANGE OVERLAY
# # ============================================================

# overlay = np.zeros((*trend_final.shape, 4))
# overlay[positive_mask] = [0, 0, 1, 0.6]
# overlay[negative_mask] = [1, 0, 0, 0.6]

# plt.figure(figsize=(6,5))
# plt.title("SAR Intensity with Persistent Change")
# plt.imshow(log_cube[-1], cmap="gray")
# plt.imshow(overlay)
# plt.show()


# # ============================================================
# # EXPORT SINGLE KML WITH TWO COLORS
# # ============================================================

# kml = simplekml.Kml()

# style_pos = simplekml.Style()
# style_pos.polystyle.color = simplekml.Color.blue
# style_pos.polystyle.fill = 1

# style_neg = simplekml.Style()
# style_neg.polystyle.color = simplekml.Color.red
# style_neg.polystyle.fill = 1

# ny, nx = trend_final.shape
# lon_min, lat_min, lon_max, lat_max = AOI_BBOX

# transform = rasterio.transform.from_origin(
#     lon_min, lat_max,
#     (lon_max - lon_min) / nx,
#     (lat_max - lat_min) / ny
# )

# for geom, val in shapes(positive_mask.astype(np.uint8), transform=transform):
#     if val == 1:
#         poly = kml.newpolygon(outerboundaryis=list(shape(geom).exterior.coords))
#         poly.style = style_pos

# for geom, val in shapes(negative_mask.astype(np.uint8), transform=transform):
#     if val == 1:
#         poly = kml.newpolygon(outerboundaryis=list(shape(geom).exterior.coords))
#         poly.style = style_neg

# kml.save(OUTPUT_KML)

# print(f"\nKML exported: {OUTPUT_KML}")
# print("BLUE = positive change | RED = negative change")

# # ============================================================
# # OPEN GOOGLE EARTH
# # ============================================================

# for path in [
#     r"C:\Program Files\Google\Google Earth Pro\client\googleearth.exe",
#     r"C:\Program Files (x86)\Google\Google Earth Pro\client\googleearth.exe"
# ]:
#     if os.path.exists(path):
#         subprocess.Popen([path, os.path.abspath(OUTPUT_KML)])
#         break