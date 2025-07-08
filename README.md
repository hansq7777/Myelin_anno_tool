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
and seeding.  The **Image** menu provides histogram stretch
(0% resets to the original image), Gaussian blur with a toggle to show the
original, and an option to clear the blur.  Each adjustment is reapplied to
the original image so changing strength values will not compound effects.
Undo and redo are found under
**Edit**.

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

### Shortcuts

- `P` – toggle brush painting
- `[` and `]` – change brush size
- `H` – switch to hand tool (panning)
- Arrow keys – navigate slices
- Right click drag – delete masks touching the selection rectangle
- `⌘S` (macOS) / `Alt+S` (Windows) – quick save masks
- `⌘E` or `⌥E` (macOS) / `Alt+E` (Windows) – open the Script Editor
