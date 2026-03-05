from zstack_anno.utils.review_utils import (
    normalize_review_grade,
    is_review_completed,
    windows_to_local_path,
    local_to_windows_path,
    build_inference_name_candidates,
    build_pair_key,
)


def test_normalize_review_grade():
    assert normalize_review_grade("a") == "A"
    assert normalize_review_grade(" B ") == "B"
    assert normalize_review_grade("c") == "C"
    assert normalize_review_grade("") == ""
    assert normalize_review_grade(None) == ""
    assert normalize_review_grade("reject") == ""


def test_windows_to_local_path():
    assert (
        windows_to_local_path(r"D:\Research\Image Analysis\file.tif")
        == "/mnt/d/Research/Image Analysis/file.tif"
    )
    assert windows_to_local_path("/mnt/d/foo/bar.tif") == "/mnt/d/foo/bar.tif"


def test_local_to_windows_path():
    assert (
        local_to_windows_path("/mnt/d/Research/Image Analysis/file.tif")
        == r"D:\Research\Image Analysis\file.tif"
    )
    assert local_to_windows_path(r"D:\already\windows.tif") == r"D:\already\windows.tif"


def test_windows_to_local_path_with_drive_map(monkeypatch):
    monkeypatch.setenv("ZSTACK_WINDOWS_DRIVE_MAP", "D=/data/myelin;E=/mnt/extra")
    assert (
        windows_to_local_path(r"D:\Research\Image Analysis\file.tif")
        == "/data/myelin/Research/Image Analysis/file.tif"
    )


def test_local_to_windows_path_with_drive_map(monkeypatch):
    monkeypatch.setenv("ZSTACK_WINDOWS_DRIVE_MAP", "D=/data/myelin")
    assert (
        local_to_windows_path("/data/myelin/Research/Image Analysis/file.tif")
        == r"D:\Research\Image Analysis\file.tif"
    )


def test_build_inference_name_candidates():
    names = build_inference_name_candidates("2501_60_R_PL_S00.ome")
    assert names == [
        "2501_60_R_PL_S00.pred.ome.tif",
        "2501_60_R_PL_S00.ome.pred.ome.tif",
    ]

    names = build_inference_name_candidates("slice_1")
    assert names == ["slice_1.pred.ome.tif"]


def test_is_review_completed():
    assert is_review_completed(True) is True
    assert is_review_completed("1") is True
    assert is_review_completed("completed") is True
    assert is_review_completed("", "2026-02-26 12:00:00") is True
    assert is_review_completed("", "") is False


def test_build_pair_key():
    assert build_pair_key("2501_60_R_PL_S00.ome.tif") == "2501_60_r_pl_s00"
    assert (
        build_pair_key("2501_60_R_PL_S00.pred.ome.tif", is_prediction=True)
        == "2501_60_r_pl_s00"
    )
    assert (
        build_pair_key("sample_A01_mask.tif", is_prediction=True)
        == "sample_a01"
    )
    assert (
        build_pair_key("2501_60_R_M1_S00.ome_dz0p396.tif")
        == "2501_60_r_m1_s00"
    )
    assert (
        build_pair_key("2501_60_R_M1_S00.ome_pred.tif", is_prediction=True)
        == "2501_60_r_m1_s00"
    )
