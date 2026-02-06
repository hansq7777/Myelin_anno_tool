# Development Log — Myelin_anno_tool

Timestamp: 2026-01-29 23:18:30 CST
Current version (from zstack_anno.__version__): 0.1.0

## Purpose
Unsupervised/semi‑automated annotation tool for microscopy Z‑stack data, aimed at preparing segmentation masks for deep learning datasets.

## Scope and Primary Users
- Biomedical imaging / microscopy workflows (TIFF/CZI Z‑stacks)
- Researchers or engineers building segmentation datasets
- Users who want a GUI with scripting automation and evaluation utilities

## Key Features
- GUI Z‑stack viewer and annotation tool (PyQt5)
- Load raw microscope data (TIFF, CZI) and export CZI XML metadata
- Mask editing: brush paint/erase, delete by rectangle, undo/redo
- Morphology operations: dilate, erode, close, skeletonize, small‑object filter
- Thresholding: absolute and normalized
- Region growing: seed generation and intensity‑based growth
- Image processing: background filtering, histogram stretch, Gaussian blur, reverse intensity, resampling
- Advanced linear/segmentation filters: Frangi, Sato, Meijering, Hessian, Gabor
  (scikit‑image + OpenCV), Structure Tensor, OpenCV Ridge, Steger Ridge,
  Felzenszwalb, Chan‑Vese, CED, TubeTK
- Seed‑based segmentation: Watershed IFT, scikit‑fmm distance, Fast Marching,
  TubeTK seed path
- Script Editor for building and running automation workflows (JSON sequences),
  including shortest‑path tracing and thin skeletonization
- Strategy Comparison and Validation Viewer with precision/recall metrics,
  TP/FP/FN overlays, and optional grid search
- CLI pipeline for batch strategy evaluation and overlay output

## High‑Level Architecture
- `zstack_anno/`
  - `controllers/`: GUI control flow and action handlers
  - `models/`: data model for stacks/masks and image processing state
  - `views/`: GUI widgets (canvas, dialogs, script editor)
  - `utils/`: I/O helpers (CZI/OME), morphology utilities, logging
  - `pipeline.py`: CLI for strategy execution and evaluation
- `tests/`: pytest test suite
- `legacy/`: older GUI reference

## Entry Points
- GUI: `python -m zstack_anno`
- CLI evaluation: `python -m zstack_anno.pipeline --stack ... --groundtruth ... --strategies ... --output ...`
- CZI metadata export: `python -m zstack_anno.utils.czi_utils myfile.czi -o meta.json`

## Dependencies
- PyQt5
- tifffile
- numpy
- scipy
- scikit‑image
- pytest (tests)
- Optional: OpenCV (ximgproc), ridge‑detector, SimpleITK, ITK/TubeTK, scikit‑fmm

## Data Formats
- Input: `.tif/.tiff`, `.ome.tif`, `.czi`
- Masks: TIFF stacks
- Scripts: JSON action sequences

## Operational Notes
- Mask visibility is adjustable via slider in the GUI
- Many operations reapply to the original image to avoid compounded effects
- Missing `scipy`/`scikit‑image` triggers slower NumPy fallbacks

## Suggested Next Documentation
- Example script JSON recipes
- Recommended parameter presets for common microscopy modalities
- Standard output directory conventions
