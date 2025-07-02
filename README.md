# Myelin_anno_tool
unsupervised annotation tool for deep learning data collection

## Running the GUI

Install the dependencies (e.g. `PyQt5`, `tifffile`, and `numpy`) and
run the application with:

```bash
python -m zstack_anno
```

This will launch the annotation window defined in `zstack_anno`.

### Menu overview

Main operations are available both as toolbar buttons and in the menu bar.
The **Mask** menu contains dilation, erosion, filtering, background removal,
and seeding.  The **Image** menu provides histogram stretch
(0% resets to the original image), Gaussian blur with a toggle to show the
original, and an option to clear the blur.  Each adjustment is reapplied to
the original image so changing strength values will not compound effects.
Undo and redo are found under
**Edit**.

### Shortcuts

- `P` – toggle brush painting
- `[` and `]` – change brush size
- `H` – switch to hand tool (panning)
- Arrow keys – navigate slices
- Right click drag – delete masks touching the selection rectangle
