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

On Linux you can also use:

```bash
./start_gui.sh
```

Full Linux deployment and troubleshooting guide:
`Linux_DEPLOY.md`

For Linux desktop use, start it inside a GUI session (X11/Wayland). If your
system reports missing Qt platform plugin `xcb`, install distro Qt/X11 runtime
libs first (package names vary by distro).

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

- Use **Review -> Build Tracker from Folders...** to select:
  - raw stack folder
  - prediction folder
  - tracker output path (`.xlsx` or `.csv`, create/refresh supported)
- Build mode matches raw/prediction by normalized file name key and writes
  mapping/status columns (`matched`, `raw_only`, `pred_only`).
- Open **Review -> Open Tracker...** and choose your tracker (`.xlsx` or `.csv`).
- Native Linux path mapping for Windows trackers:
  if tracker path columns still contain `D:\...` style paths, set
  `ZSTACK_WINDOWS_DRIVE_MAP` before launch, for example:
  `export ZSTACK_WINDOWS_DRIVE_MAP="D=/data/confocal;E=/mnt/extra"`.
- The current stack loads with prediction overlay (raw image + red mask).
- In `All`/`Unreviewed` filter modes, loading uses random unfinished pairs to
  improve sample diversity.
- On load, the app defaults to **Unreviewed** and opens the first unreviewed
  zstack automatically (falls back to **All** if none are unreviewed).
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
- Saving corrected masks also marks the item as completed
  (`review_completed=1`, `review_completed_at=...`), and completed items are
  excluded from random selection.
- Corrected masks now embed pairing metadata in TIFF `ImageDescription` (JSON):
  raw/pred paths, zstack id, review grade, shape mapping and resample policy.
- If raw/prediction dimensions differ, review loading now uses a **mask-grid**
  workflow: raw stack is downsampled/resampled in memory to the prediction
  stack shape for stable overlay and editing.
- Corrected masks are saved on the prediction grid (same dimensions as
  inference), so exported training masks stay consistent with downstream
  downsampled-DZ training pipelines.
- This keeps review/edit/export aligned with model-training data that already
  uses a unified downsampled `dz`.
- Use **Export Final Masks** to build a unified export under
  `review_final_masks/<GRADE>/...` for all reviewed stacks:
  - if a corrected mask exists, export the corrected mask
  - otherwise export the original inference mask
  - stale files in `review_final_masks` can be cleaned during export
- Use **Quick Auto Script** (or `Alt+Q`) to run the default one-click pipeline
  on the current slice:
  `Seed -> Dilate -> Background Filter -> Intensity Grow -> Background Filter`.
  The current version adds tail cleanup:
  `Background Filter(5%, bins=5) x5 + remove components <20 px`.
- Use **Clear <= Slice** to clear labels on the current slice and all previous
  slices.
- Use **Clear >= Slice** to clear labels on the current slice and all following
  slices.
- Use **Auto Preset** to switch quick-auto parameters:
  `Conservative / Balanced / Aggressive`.
- Use **Quick Auto Stack** (or `Alt+W`) to run the quick-auto strategy on:
  all slices, a slice range, or key slices (first/middle/last). The viewer
  stops at the last processed slice for manual review.
- A post-run quality gate checks abnormal foreground growth; if triggered, you
  can revert in one click to the pre-run snapshot using the popup or
  **Revert Auto Snapshot** (`Alt+Shift+Q`).
- The gate also checks foreground coverage ratio to avoid large background
  takeover.
- Quick auto now includes two safeguards for small GT preservation and BG
  suppression:
  - keep only intensity-supported new additions
  - protect small original components from accidental removal

Tracker columns are auto-created if missing:
`review_grade`, `review_status`, `review_note`, `review_updated_at`,
`review_corrected_mask_path`, `review_corrected_saved_at`,
`review_final_mask_path`, `review_final_mask_source`,
`review_final_exported_at`.

### Shortcut Reference

| Shortcut | Action |
| --- | --- |
| `Up/Left` | Previous slice |
| `Down/Right` | Next slice |
| `Alt+,` | Previous review stack |
| `Alt+.` | Next review stack |
| `Alt+1 / Alt+2 / Alt+3` | Mark current stack as A/B/C |
| `Alt+Shift+F` | Export reviewed final masks (A/B/C) |
| `Alt+Q` | Run quick auto on current slice |
| `Alt+W` | Run quick auto on stack/range |
| `Alt+Shift+Q` | Revert to pre-auto snapshot |
| `Alt+S` / `Ctrl+S` / `Meta+S` | Quick save masks |
| `Alt+D` / `Meta+D` | Clear foreground on current slice |
| `D` / `E` | Dilate / Erode |
| `Z` / `X` | Undo / Redo |
| `P` | Toggle brush mode |
| `L` | Toggle eraser mode (paint background) |
| `[` / `]` | Brush size down/up |
| `H` | Hand tool (panning) |

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
