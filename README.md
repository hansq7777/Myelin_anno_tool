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

This will launch the annotation window defined in `zstack_anno`.

Installing both `scipy` and `scikit-image` enables faster morphology
operations. If either package is missing, slower NumPy fallbacks will be used
and a warning will be issued at runtime.

### Menu overview

Main operations are available both as toolbar buttons and in the menu bar.
The **Mask** menu contains dilation, erosion, size filtering, background removal,
and seeding. Seeding places random seed pixels above an intensity percentile,
sampling a percentage of the image size.
The **Image** menu provides histogram stretch
(0% resets to the original image), Gaussian blur with a toggle to show the
unprocessed original, and an option to clear the blur.  Each adjustment is reapplied to
the original image so changing strength values will not compound effects.
Undo and redo are found under
**Edit**.

The **File** menu also includes an option to export the raw metadata from a CZI
image. This writes the XML metadata to disk so you can inspect stage
coordinates and other acquisition details.

### Script Editor

The **Tools** menu opens a Script Editor for building simple automation
workflows. The editor appears in its own window so you can keep interacting
with the main interface while it is open. Drag actions from the list on the
right into the sequence on the left, adjust their parameters, then run, pause or
stop the script. Sequences can be saved to or loaded from JSON files.

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
