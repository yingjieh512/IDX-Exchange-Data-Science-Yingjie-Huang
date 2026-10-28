"""Tests for the CDE boundary join's row and grade-range semantics."""

import io
import json
import sys
from pathlib import Path

import pandas as pd
import pytest
from shapely.geometry import box, mapping


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "py"))
import school_district_features as districts


def _fixture_bytes(year="2024-25"):
    areas = [
        ("0100001", "Elementary", box(-120.5, 34.5, -119.5, 35.5)),
        ("0100000", "Elementary", box(-120.0, 34.5, -119.0, 35.5)),
        ("0200000", "High", box(-120.5, 34.5, -119.0, 35.5)),
        ("0300000", "Unified", box(-118.5, 34.5, -117.5, 35.5)),
    ]
    collection = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"Year": year, "CDCode": code, "DistrictType": kind},
                "geometry": mapping(geometry),
            }
            for code, kind, geometry in areas
        ],
    }
    return json.dumps(collection).encode("utf-8")


def test_overlap_unified_imputation_and_invalid_coordinates_keep_one_row(tmp_path):
    cache_path = tmp_path / "districts.geojson"
    cache_path.write_bytes(_fixture_bytes())
    frame = pd.DataFrame(
        {
            "Longitude": [-120.25, -119.75, -118.0, -120.25, -118.0, -120.25, -116.0],
            "Latitude": [35.0, 35.0, 35.0, 35.0, 35.0, 50.0, 35.0],
            "missingindicator_Longitude": [0, 0, 0, 1, 0, 0, 0],
            "missingindicator_Latitude": [0, 0, 0, 0, "True", 0, 0],
        },
        index=[7, 7, 2, 4, 8, 5, 6],
    )

    joined, diagnostic = districts.join_school_districts(frame, cache_path=cache_path)

    assert joined.index.tolist() == frame.index.tolist()
    assert joined.shape == (7, 2)
    assert joined["ElementaryDistrictCode"].tolist()[:3] == [
        "0100001", "0100000", "0300000"
    ]
    assert joined["HighDistrictCode"].tolist()[:3] == [
        "0200000", "0200000", "0300000"
    ]
    assert joined.iloc[3:].isna().all().all()
    assert diagnostic["eligible_coordinate_rows"] == 4
    assert diagnostic["imputed_coordinate_rows"] == 2
    assert diagnostic["invalid_coordinate_rows"] == 1
    assert diagnostic["both_matched_rows"] == 3
    assert diagnostic["ambiguous_within_type_rows"]["Elementary"] == 1
    assert len(diagnostic["source_sha256"]) == 64


def test_missing_cache_downloads_and_reuses_validated_source(tmp_path, monkeypatch):
    cache_path = tmp_path / "data" / "districts.geojson"
    calls = []

    def fake_urlopen(request, timeout):
        calls.append(request.full_url)
        return io.BytesIO(_fixture_bytes())

    monkeypatch.setattr(districts, "urlopen", fake_urlopen)
    frame = pd.DataFrame({"Longitude": [-118.0], "Latitude": [35.0]})
    first, _ = districts.join_school_districts(frame, cache_path=cache_path)
    second, _ = districts.join_school_districts(frame, cache_path=cache_path)

    assert calls == [districts.SOURCE_URL]
    assert cache_path.is_file()
    pd.testing.assert_frame_equal(first, second)


def test_invalid_source_year_is_rejected(tmp_path):
    cache_path = tmp_path / "wrong_year.geojson"
    cache_path.write_bytes(_fixture_bytes(year="2023-24"))
    frame = pd.DataFrame({"Longitude": [-118.0], "Latitude": [35.0]})

    with pytest.raises(ValueError, match="unexpected boundary year"):
        districts.join_school_districts(frame, cache_path=cache_path)
