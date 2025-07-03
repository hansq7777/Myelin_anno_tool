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
The **Mask** menu contains dilation, erosion, size and linearity filtering,
background removal, and seeding. Linearity filtering splits skeletons at
junctions and keeps only elongated segments.  The **Image** menu provides histogram stretch
(0% resets to the original image), Gaussian blur with a toggle to show the
original, and an option to clear the blur.  Each adjustment is reapplied to
the original image so changing strength values will not compound effects.
Undo and redo are found under
**Edit**.

## Running the tests

To execute the test suite:

```bash
pytest
```

### Shortcuts

- `P` – toggle brush painting
- `[` and `]` – change brush size
- `H` – switch to hand tool (panning)
- Arrow keys – navigate slices
- Right click drag – delete masks touching the selection rectangle
- `⌘S` (macOS) / `Alt+S` (Windows) – quick save masks
