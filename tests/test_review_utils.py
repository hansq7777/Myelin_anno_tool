from zstack_anno.utils.review_utils import (
    normalize_review_grade,
    windows_to_local_path,
    local_to_windows_path,
    build_inference_name_candidates,
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


def test_build_inference_name_candidates():
    names = build_inference_name_candidates("2501_60_R_PL_S00.ome")
    assert names == [
        "2501_60_R_PL_S00.pred.ome.tif",
        "2501_60_R_PL_S00.ome.pred.ome.tif",
    ]

    names = build_inference_name_candidates("slice_1")
    assert names == ["slice_1.pred.ome.tif"]
