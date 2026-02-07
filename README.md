# Myelin_anno_tool
unsupervised annotation tool for deep learning data collection

## Running the GUI

Install the required packages using:

```bash
pip install -r requirements.txt
```

After installation, run the application with:

```bash
python -m zstack_anno
```

This will launch the annotation window defined in `zstack_anno`.  The
**Open** action accepts both TIFF and CZI files so you can view raw microscope
data without converting first.

Chinese guide for the review workflow:
`REVIEW_QUICK_AUDIT_GUIDE_zh.md`

Installing both `scipy` and `scikit-image` enables faster morphology
operations. If either package is missing, slower NumPy fallbacks will be used
and a warning will be issued at runtime.

### Menu overview

Main operations are available both as toolbar buttons and in the menu bar.
The **Mask** menu contains dilation, erosion, size filtering, background removal,
and seeding. Seeding places random seed pixels above an intensity percentile,
sampling a percentage of the image size. A Mask Visibility slider in the mask
panel adjusts segmentation mask opacity from 0 (hidden) to 100 (solid color).
The **Image** menu provides histogram stretch
(0% resets to the original image), Gaussian blur with a toggle to show the
unprocessed original, and an option to clear the blur.  Each adjustment is reapplied to
the original image so changing strength values will not compound effects.
Undo and redo are found under
**Edit**.

The **File** menu also includes an option to export the raw metadata from a CZI
image. This writes the XML metadata to disk so you can inspect stage
coordinates and other acquisition details.

### Linear menu (advanced filters)

The **Linear** menu exposes raw-image and seed-mask operations geared toward
line-like structures:

- Raw Image: Frangi, Sato, Meijering, Hessian, Gabor (scikit-image), OpenCV
  Gabor, Structure Tensor, Felzenszwalb, OpenCV Ridge, Steger Ridge, Chan-Vese,
  CED Filter, TubeTK Tubes.
- Seed Mask: Watershed IFT, scikit-fmm, Fast Marching, TubeTK Seed Path.
- Thin Skeleton: one-click thinning for existing masks.

Some of these actions rely on optional dependencies (OpenCV/ximgproc,
ridge-detector, SimpleITK, ITK/TubeTK, scikit-fmm). If a dependency is missing,
the action will fall back to a no-op or zero mask and log a warning.

### Script Editor

The **Tools** menu opens a Script Editor for building simple automation
workflows. The editor appears in its own window so you can keep interacting
with the main interface while it is open. Drag actions from the list on the
right into the sequence on the left, adjust their parameters, then run, pause or
stop the script. Sequences can be saved to or loaded from JSON files.
Script actions include the Linear menu filters plus utilities like shortest
path tracing.

### Strategy Comparison and Validation Viewer

The **Tools** menu also includes:

- **Strategy Comparison**: run multiple script JSON strategies on a stack (with
  optional grid search) and compare overlays/metrics across slices.
- **Validation Viewer**: view predicted masks vs ground truth with TP/FP/FN
  overlays and whole-stack statistics.

### Review workflow (raw + prediction QC)

The app now supports a fast review loop for selecting high-quality predictions
for finetuning:

- Open **Review -> Open Tracker...** and choose your tracker `.xlsx`.
- The current stack loads with prediction overlay (raw image + red mask).
- Use **Prev Stack/Next Stack** (or `Alt+,` / `Alt+.`) to move between zstacks.
- Mark quality in one click:
  - `A` = usable as-is
  - `B` = usable after light edits
  - `C` = not selected for the next round
  (`Alt+1` / `Alt+2` / `Alt+3`).
- Use the **Filter** dropdown (`All`, `Unreviewed`, `A`, `B`, `C`) to focus on a
  subset.
- Use **Save Corrected Mask** to write edited masks into
  `review_corrected_masks/<GRADE>/...` and store the path back into the tracker.

Tracker columns are auto-created if missing:
`review_grade`, `review_status`, `review_note`, `review_updated_at`,
`review_corrected_mask_path`, `review_corrected_saved_at`.

### CLI pipeline (batch evaluation)

To evaluate one or more strategy JSON files against a stack and ground truth:

```bash
python -m zstack_anno.pipeline --stack stack.tif --groundtruth gt.tif --strategies a.json b.json --output pipeline_results
```

The command writes per-slice overlay images and prints precision/recall to
stdout.

## Running the tests

To execute the test suite:

```bash
pytest
```

### Extracting CZI metadata

Stage positions and pixel sizes can be retrieved from a `.czi` file with:

```bash
python -m zstack_anno.utils.czi_utils myfile.czi -o meta.json
```

The generated JSON lists the physical resolution (`pixel_size`), the number of
stacks captured (`stack_count`) and the stage coordinates for each stack.
You can also choose **File → Export CZI Metadata…** in the GUI to save the raw
XML metadata for manual inspection.

### Writing and reading OME-TIFF metadata

Image stacks can be saved as OME-TIFF while preserving all metadata. The
`tifffile` library writes the provided XML directly into the file. After saving
you can recover and inspect the metadata using
`parse_zeiss_ome_metadata` from `zstack_anno.utils.ome_utils`:

```python
from tifffile import imwrite, TiffFile
from zstack_anno.utils.ome_utils import parse_zeiss_ome_metadata

# ``xml`` contains the metadata string extracted from a CZI file
imwrite("stack.ome.tif", data, ome=xml)

with TiffFile("stack.ome.tif") as tif:
    ome_xml = tif.ome_metadata
info = parse_zeiss_ome_metadata(ome_xml)
print(info["document"])  # access name, user and creation date
```

The **Stack Info** menu item uses this parser to display details such as file
and user information, image dimensions and pixel scaling, channel names,
instrument models and acquisition settings.

### Shortcuts

- `P` – toggle brush painting
- `[` and `]` – change brush size
- `H` – switch to hand tool (panning)
- Arrow keys – navigate slices
- Right click drag – delete masks touching the selection rectangle
- `⌘S` (macOS) / `Alt+S` (Windows) – quick save masks
- `⌘E` or `⌥E` (macOS) / `Alt+E` (Windows) – open the Script Editor
- `⌘D` or `⌥D` (macOS) / `Alt+D` (Windows) – clear foreground on the current slice
