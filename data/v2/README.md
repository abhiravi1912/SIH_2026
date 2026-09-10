# DRISHTI V2 Land-Cover Dataset Documentation

## 1. Dataset Sources
The V2 dataset is built entirely from real, calibrated **Sentinel-2 Level-2A** Bottom-of-Atmosphere (BOA) surface reflectance imagery accessed via the **Copernicus Open Access Hub** and AWS Open Data Sentinel-2 Cloud-Optimized GeoTIFFs (COGs). No synthetic spectral profiles are used in evaluation.

## 2. Number of Scenes & Composition
- **Total Registered Scenes**: 11 scenes cataloged in `data/v2/metadata/scene_registry.json`.
- **Training Scenes (5 scenes, 871,072 pixels)**:
  - `mixed_ref_2024`: Guwahati Mixed Corridor (2024-03-11, Sentinel-2A)
  - `forest_ref_2024`: Garbhanga Forest Reserve (2024-02-10, Sentinel-2B)
  - `urban_ref_2024`: Dispur Urban Core (2024-02-10, Sentinel-2B)
  - `river_east_ref_2024`: Brahmaputra River East Channel (2024-02-10, Sentinel-2B)
  - `archive_20241226`: Guwahati Winter Solstice Regional Extent (2024-12-26, Sentinel-2B)
- **Validation Scenes (2 scenes, 352,144 pixels)**:
  - `river_west_ref_2023`: Brahmaputra River Western Channel (2023-01-21, Sentinel-2B)
  - `wetland_ref_2024`: Deepor Beel Ramsar Wetland (2024-02-10, Sentinel-2B)
- **Test Scenes (3 scenes, 786,432 pixels — held-out & spatially disjoint)**:
  - `test_scene_1_vit_ap_2026`: VIT-AP Campus, Amaravati, Andhra Pradesh (2026-03-05, Sentinel-2C) — 262,144 px
  - `test_scene_2_river_west_2026`: Brahmaputra River Western Channel, Assam (2026-03-06, Sentinel-2A) — 262,144 px
  - `test_scene_3_vit_ap_2021`: VIT-AP Campus Historical Baseline, Andhra Pradesh (2021-03-06, Sentinel-2B) — 262,144 px
- **Temporal Holdout Scene (1 scene, 90,000 pixels)**:
  - `wetland_tgt_2025`: Deepor Beel Hydrological Holdout (2025-02-09, Sentinel-2A)

## 3. Geographic Coverage
The dataset spans two vastly different geographic and ecological zones separated by ~1,500 km:
1. **Brahmaputra Valley / Guwahati Regional Corridor, Assam, Northeast India**: Sub-tropical humid climate, complex river braided channels, dense semi-evergreen forest canopy, rapid urban sprawl.
2. **Amaravati Capital Region / VIT-AP Campus, Andhra Pradesh, South India**: Semi-arid/tropical climate, red and black soils, intensive campus construction, agricultural plots.

## 4. Temporal Coverage
Multi-temporal spans from **2021 to 2026**:
- 2021-03-06 (Historical baseline)
- 2023-01-21 (Winter river dynamics)
- 2024-02-10 (Post-monsoon dry season)
- 2024-03-11 (Spring pre-monsoon)
- 2024-12-26 (Mid-winter low water)
- 2025-02-09 (Wetland hydrology)
- 2026-03-05 & 2026-03-06 (Current multi-sensor captures)

## 5. Sensors & Resolution
- **Sensor**: Sentinel-2 MultiSpectral Instrument (MSI), Level-2A BOA reflectance.
- **Bands Used**: Band 4 (Red, 665nm), Band 3 (Green, 560nm), Band 2 (Blue, 490nm), Band 8 (NIR, 842nm).
- **Spatial Resolution**: 10.0 meters per pixel native resolution.
- **Coordinate Reference System (CRS)**: WGS 84 (EPSG:4326).

## 6. Features Extracted
Every classifier ingests the identical standardized 6-element feature vector:
1. `R`: Normalized Red band ($B04 / 255.0 \in [0, 1]$)
2. `G`: Normalized Green band ($B03 / 255.0 \in [0, 1]$)
3. `B`: Normalized Blue band ($B02 / 255.0 \in [0, 1]$)
4. `NIR`: Normalized Near-Infrared band ($B08 / 255.0 \in [0, 1]$)
5. `NDVI`: Normalized Difference Vegetation Index:
   $$\text{NDVI} = \frac{\text{NIR} - \text{Red}}{\text{NIR} + \text{Red} + 10^{-5}}$$
6. `NDWI`: Normalized Difference Water Index (McFeeters):
   $$\text{NDWI} = \frac{\text{Green} - \text{NIR}}{\text{Green} + \text{NIR} + 10^{-5}}$$

## 7. Labeling Methodology & Class Mapping
Classes are matched with the project standard (`backend/models/segmentation.py`):
- `0`: Bare land (soil, unpaved surfaces, sandbars)
- `1`: Vegetation (forest canopy, shrubs, cropland)
- `2`: Water (river channels, lake, wetland open water)
- `3`: Road (asphalt, concrete roads, runways)
- `4`: Building (roofs, concrete built-up structures)

## 8. Weak Labels vs Real Annotated Labels (Two Evaluation Tracks)
The project architecture maintains strict separation between two tracks:
- **Track A (Weak-Label Benchmark)**:
  Uses deterministic, physics-based spectral thresholding applied to real Sentinel-2 bands:
  - Water: $\text{NDWI} > 0.15$
  - Vegetation: $\text{NDVI} > 0.25$
  - Building: $\text{Red} > 150 \land \text{NIR} > 130 \land \text{NDVI} < 0.10$
  - Road: $|\text{Red} - \text{Green}| < 15 \land |\text{Green} - \text{Blue}| < 15 \land 60 \le \text{Red} \le 150$
  - Bare land: All remaining unassigned pixels.
  Stored under `data/v2/labels/weak_labels/`.
- **Track B (Annotated Ground Truth Benchmark)**:
  Dedicated interface under `data/v2/labels/annotated/` for human-verified, polygon-annotated ground-truth rasters.
  *Note*: In the current execution environment, external hosting repositories (Zenodo LoveDA, EuroSAT) return HTTP 403 or host lookup failures. Thus, results presented in the primary report are strictly designated as Track A, and Track B will be evaluated when offline ground-truth GeoTIFFs are ingested.

## 9. Leakage Prevention (Strict Spatial & Temporal Disjointness)
1. **Scene-Level Separation**: Pixels from a given scene never span multiple splits.
2. **Zero Spatial Overlap**: Train scenes (Guwahati Region) and Test scenes (Amaravati Region and Western Brahmaputra Channel) have **0.00%** geographic bounding-box overlap.
3. **Centroid Independence**: Centroid distance between Train scenes and Test scenes exceeds 1,500 km for Andhra Pradesh scenes and > 5 km across the non-overlapping river channel.
4. **Scaler Isolation**: Feature standardizers (`StandardScaler`) are fit strictly on training splits.
5. **Automated Verification**: Enforced via `eval/v2/leakage_check.py`.

## 10. Known Limitations
1. Spectral threshold labels cannot resolve fine-grained ambiguities between concrete road surfaces and bare soil without spatial/texture context.
2. Spectral rules approximate human annotations; benchmark ranks reflect model ability to learn complex multi-spectral boundaries rather than absolute real-world ground truth.

## 11. Reproducibility Instructions
To regenerate all processed feature arrays, weak label arrays, and split manifests from scratch:
```bash
python data/v2/prepare_dataset.py
```
To verify leakage across all splits:
```bash
python eval/v2/leakage_check.py
```
