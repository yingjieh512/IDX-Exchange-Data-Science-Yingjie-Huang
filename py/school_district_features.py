"""Spatial features from CDE's 2024-25 California school district boundaries.

The source is the California Department of Education's public GeoJSON export.
Elementary and high school district areas overlap by design, so each type is
queried separately. A unified district supplies both grade-range features.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import shapely
from shapely.geometry import shape
from shapely.strtree import STRtree


SOURCE_URL = (
    "https://gis.data.ca.gov/api/download/v1/items/"
    "b0e3b936426a47ce9d9a2e77e2bb86cc/geojson?layers=0"
)
SOURCE_PAGE = (
    "https://data.ca.gov/@california-department-of-education/"
    "california-school-district-areas-2024-25"
)
REFERENCE_SHA256 = "a57442a35f7238ec30f97270e39ad14a94706a8d352254ccb3a7e9a97f833d61"
DEFAULT_CACHE_PATH = Path("data/ca_school_district_areas_2024_25.geojson")
OUTPUT_COLUMNS = ("ElementaryDistrictCode", "HighDistrictCode")
_DISTRICT_TYPES = ("Elementary", "High", "Unified")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download_if_absent(cache_path: Path) -> None:
    if cache_path.is_file():
        return
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".school_district_", suffix=".geojson", dir=cache_path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            request = Request(SOURCE_URL, headers={"User-Agent": "idx-exchange-week6/1.0"})
            with urlopen(request, timeout=120) as response:
                shutil.copyfileobj(response, output)
        # Check the completed download before it becomes the reusable cache.
        _load_boundaries(temporary_path)
        os.replace(temporary_path, cache_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _load_boundaries(path: Path) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            collection = json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read school district GeoJSON at {path}") from error

    if collection.get("type") != "FeatureCollection" or not collection.get("features"):
        raise ValueError("School district source must be a nonempty GeoJSON FeatureCollection")
    crs_name = collection.get("crs", {}).get("properties", {}).get("name", "CRS84")
    if "CRS84" not in crs_name and "4326" not in crs_name:
        raise ValueError(f"School district coordinates must be longitude/latitude: {crs_name}")

    geometries: dict[str, list] = {district_type: [] for district_type in _DISTRICT_TYPES}
    codes: dict[str, list[str]] = {district_type: [] for district_type in _DISTRICT_TYPES}
    seen_codes: set[str] = set()
    repaired_count = 0
    for feature in collection["features"]:
        properties = feature.get("properties") or {}
        district_type = properties.get("DistrictType")
        code = properties.get("CDCode")
        if properties.get("Year") != "2024-25":
            raise ValueError("School district source contains an unexpected boundary year")
        if district_type not in geometries:
            raise ValueError(f"Unexpected school district type: {district_type}")
        if not isinstance(code, str) or len(code) != 7 or not code.isdigit():
            raise ValueError(f"Invalid school district CDCode: {code}")
        if code in seen_codes:
            raise ValueError(f"Duplicate school district CDCode: {code}")
        seen_codes.add(code)
        geometry = shape(feature.get("geometry"))
        if geometry.is_empty or geometry.geom_type not in ("Polygon", "MultiPolygon"):
            raise ValueError(f"Invalid polygon for school district {code}")
        if not geometry.is_valid:
            geometry = shapely.make_valid(geometry)
            repaired_count += 1
        if geometry.geom_type not in ("Polygon", "MultiPolygon"):
            raise ValueError(f"Cannot repair polygon for school district {code}")
        geometries[district_type].append(geometry)
        codes[district_type].append(code)

    layers = {
        district_type: (
            np.asarray(geometries[district_type], dtype=object),
            np.asarray(codes[district_type], dtype=str),
        )
        for district_type in _DISTRICT_TYPES
    }
    metadata = {
        "boundary_feature_count": len(seen_codes),
        "boundary_type_counts": {name: len(codes[name]) for name in _DISTRICT_TYPES},
        "repaired_boundary_count": repaired_count,
        "source_sha256": _sha256(path),
    }
    metadata["source_matches_reference_sha256"] = (
        metadata["source_sha256"] == REFERENCE_SHA256
    )
    return layers, metadata


def _query_layer(
    points: np.ndarray, geometries: np.ndarray, codes: np.ndarray
) -> tuple[np.ndarray, int]:
    """Return one deterministic CDCode per point and the ambiguous point count."""
    matches = np.full(len(points), None, dtype=object)
    if not len(points) or not len(geometries):
        return matches, 0
    point_positions, polygon_positions = STRtree(geometries).query(
        points, predicate="intersects"
    )
    if not len(point_positions):
        return matches, 0
    candidate_codes = codes[polygon_positions]
    # On shared boundaries or overlapping same-type areas, choose the lowest
    # CDCode. Sorting makes the result independent of GeoJSON feature order.
    order = np.lexsort((candidate_codes, point_positions))
    sorted_positions = point_positions[order]
    first_positions = np.unique(sorted_positions, return_index=True)[1]
    selected = order[first_positions]
    matches[point_positions[selected]] = candidate_codes[selected]
    ambiguous_count = int(np.count_nonzero(np.bincount(point_positions) > 1))
    return matches, ambiguous_count


def join_school_districts(
    frame: pd.DataFrame, *, cache_path: Path = DEFAULT_CACHE_PATH
) -> tuple[pd.DataFrame, dict]:
    """Join property coordinates to CDE elementary and high district codes.

    Coordinates must be longitude/latitude in plausible California bounds.
    Rows flagged by either optional ``missingindicator_Latitude`` or
    ``missingindicator_Longitude`` are not joined, even if their coordinates
    were subsequently imputed. The output preserves row count and index,
    including duplicate index labels. Unmatched codes are pandas string NA.
    """
    missing_columns = {"Longitude", "Latitude"} - set(frame.columns)
    if missing_columns:
        raise KeyError(f"Missing coordinate columns: {sorted(missing_columns)}")
    cache_path = Path(cache_path)
    _download_if_absent(cache_path)
    layers, metadata = _load_boundaries(cache_path)

    latitude = pd.to_numeric(frame["Latitude"], errors="coerce").to_numpy(
        dtype=float, na_value=np.nan
    )
    longitude = pd.to_numeric(frame["Longitude"], errors="coerce").to_numpy(
        dtype=float, na_value=np.nan
    )
    imputed = np.zeros(len(frame), dtype=bool)
    for column in ("missingindicator_Latitude", "missingindicator_Longitude"):
        if column in frame:
            indicator = frame[column]
            numeric_flag = pd.to_numeric(indicator, errors="coerce").fillna(0).ne(0)
            text_flag = indicator.astype("string").str.strip().str.lower().isin(
                ("true", "t", "yes", "y")
            )
            imputed |= (numeric_flag | text_flag).to_numpy(dtype=bool)
    plausible_coordinates = (
        np.isfinite(latitude)
        & np.isfinite(longitude)
        & (latitude >= 32.0)
        & (latitude <= 42.2)
        & (longitude >= -124.7)
        & (longitude <= -114.0)
    )
    eligible = plausible_coordinates & ~imputed
    eligible_positions = np.flatnonzero(eligible)
    points = shapely.points(longitude[eligible], latitude[eligible])

    by_type: dict[str, np.ndarray] = {}
    ambiguous: dict[str, int] = {}
    for district_type in _DISTRICT_TYPES:
        geometries, codes = layers[district_type]
        by_type[district_type], ambiguous[district_type] = _query_layer(
            points, geometries, codes
        )

    elementary = np.where(
        by_type["Elementary"] != None, by_type["Elementary"], by_type["Unified"]
    )
    high = np.where(by_type["High"] != None, by_type["High"], by_type["Unified"])
    elementary_all = np.full(len(frame), None, dtype=object)
    high_all = np.full(len(frame), None, dtype=object)
    elementary_all[eligible_positions] = elementary
    high_all[eligible_positions] = high
    result = pd.DataFrame(
        {
            OUTPUT_COLUMNS[0]: pd.array(elementary_all, dtype="string"),
            OUTPUT_COLUMNS[1]: pd.array(high_all, dtype="string"),
        },
        index=frame.index,
    )
    diagnostics = {
        **metadata,
        "source_url": SOURCE_URL,
        "source_page": SOURCE_PAGE,
        "boundary_year": "2024-25",
        "input_rows": len(frame),
        "eligible_coordinate_rows": int(eligible.sum()),
        "imputed_coordinate_rows": int(imputed.sum()),
        "invalid_coordinate_rows": int((~plausible_coordinates & ~imputed).sum()),
        "elementary_matched_rows": int(pd.notna(elementary_all).sum()),
        "high_matched_rows": int(pd.notna(high_all).sum()),
        "both_matched_rows": int((pd.notna(elementary_all) & pd.notna(high_all)).sum()),
        "unified_matched_rows": int(pd.notna(by_type["Unified"]).sum()),
        "ambiguous_within_type_rows": ambiguous,
    }
    return result, diagnostics
