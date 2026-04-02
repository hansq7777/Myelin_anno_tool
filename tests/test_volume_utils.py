import json
import sys
from pathlib import Path

import numpy as np
import pytest
import tifffile

root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from zstack_anno.utils.volume_utils import (
    build_mask_source_with_reference_extent,
    build_resampled_three_d_raw_source,
    find_matching_confocal_inference,
    infer_spacing_xyz_from_reference,
    normalize_stack_id,
)


OME_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">
  <Image ID="Image:0" Name="stack">
    <Pixels DimensionOrder="XYZCT" ID="Pixels:0" Type="uint8"
      SizeX="4" SizeY="4" SizeZ="3" SizeC="1" SizeT="1"
      PhysicalSizeX="{sx}" PhysicalSizeY="{sy}" PhysicalSizeZ="{sz}">
    </Pixels>
  </Image>
</OME>
"""


def _write_ome_tiff(path: Path, data: np.ndarray, *, sx: float, sy: float, sz: float) -> None:
    tifffile.imwrite(path, data, description=OME_TEMPLATE.format(sx=sx, sy=sy, sz=sz))


def test_normalize_stack_id_handles_raw_pred_and_resampled_suffixes():
    assert normalize_stack_id("2502_60_L_M1_S00.ome.tif") == "2502_60_L_M1_S00"
    assert normalize_stack_id("2502_60_L_M1_S00.pred.ome.tif") == "2502_60_L_M1_S00"
    assert normalize_stack_id("2502_60_L_M1_S00.ome_dz0p396.tif") == "2502_60_L_M1_S00"


def test_find_matching_confocal_inference_discovers_2p5d_and_3d_bundle(tmp_path: Path):
    raw_dir = tmp_path / "Confocal Myelin data" / "202512_8rats_3ROIs" / "2502_60_L_M1"
    raw_dir.mkdir(parents=True)
    raw_path = raw_dir / "2502_60_L_M1_S00.ome.tif"
    _write_ome_tiff(raw_path, np.zeros((3, 4, 4), dtype=np.uint8), sx=0.2, sy=0.2, sz=0.5)

    inference_root = tmp_path / "Confocal Myelin data" / "Inference"
    two_p_dir = inference_root / "2026-02-05_7ch_combo_pred_zstacks"
    two_p_dir.mkdir(parents=True)
    tifffile.imwrite(two_p_dir / "2502_60_L_M1_S00.pred.ome.tif", np.ones((2, 4, 4), dtype=np.uint8))

    bundle = inference_root / "2026-02-11_232719_3d_vanilla_5fold_665_detached_3d_vanilla_dataset007_665_chunked"
    (bundle / "original_zstacks").mkdir(parents=True)
    (bundle / "meta" / "all_592stacks").mkdir(parents=True)
    tifffile.imwrite(bundle / "original_zstacks" / "2502_60_L_M1_S00.ome_dz0p396.tif", np.ones((2, 4, 4), dtype=np.uint8))
    (bundle / "meta" / "all_592stacks" / "2502_60_L_M1_S00.ome.json").write_text(
        json.dumps({"dz_target": 0.396}),
        encoding="utf-8",
    )

    match = find_matching_confocal_inference(str(raw_path))

    assert match["stack_id"] == "2502_60_L_M1_S00"
    assert match["two_p_five_d_mask_path"] is not None
    assert match["three_d_resampled_raw_path"] is not None
    assert match["three_d_meta_json_path"] is not None
    assert match["three_d_bundle_root"] is not None


def test_infer_spacing_xyz_from_reference_matches_physical_extent():
    spacing_xyz = infer_spacing_xyz_from_reference(
        (65, 512, 512),
        (0.43929408516605245, 0.43929408516605245, 0.20147761763974992),
        (33, 512, 512),
    )

    assert spacing_xyz[0] == pytest.approx(0.43929408516605245)
    assert spacing_xyz[1] == pytest.approx(0.43929408516605245)
    assert spacing_xyz[2] == pytest.approx(0.39684985292678016)


def test_build_mask_source_with_reference_extent_infers_missing_spacing(tmp_path: Path):
    mask_path = tmp_path / "pred.ome.tif"
    tifffile.imwrite(mask_path, np.ones((33, 512, 512), dtype=np.uint8))

    source = build_mask_source_with_reference_extent(
        str(mask_path),
        reference_shape_zyx=(65, 512, 512),
        reference_spacing_xyz=(0.43929408516605245, 0.43929408516605245, 0.20147761763974992),
        label="2.5D nnUNet prediction",
        note="Loaded from sibling inference folder.",
    )

    assert source.spacing_xyz[0] == pytest.approx(0.43929408516605245)
    assert source.spacing_xyz[1] == pytest.approx(0.43929408516605245)
    assert source.spacing_xyz[2] == pytest.approx(0.39684985292678016)
    assert "inferred from current raw stack extent" in (source.note or "")


def test_build_resampled_three_d_raw_source_uses_meta_target_spacing(tmp_path: Path):
    raw_path = tmp_path / "raw.ome.tif"
    resampled_path = tmp_path / "raw_dz0p396.tif"
    meta_path = tmp_path / "raw.ome.json"

    _write_ome_tiff(raw_path, np.zeros((6, 4, 4), dtype=np.uint8), sx=0.455, sy=0.455, sz=0.2015)
    tifffile.imwrite(resampled_path, np.zeros((3, 4, 4), dtype=np.uint8))
    meta_path.write_text(json.dumps({"dz_target": 0.396}), encoding="utf-8")

    source = build_resampled_three_d_raw_source(
        str(raw_path),
        str(resampled_path),
        str(meta_path),
    )

    assert source.label == "3D aligned raw"
    assert source.spacing_xyz == (0.455, 0.455, 0.396)
    assert "Z spacing 0.2015 -> 0.3960" in (source.note or "")
