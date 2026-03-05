# Linux Deployment Guide

This document provides a reproducible Linux deployment path for launching the
`Myelin_anno_tool` GUI (`PyQt5`).

## 1. Scope

- Target: Ubuntu/Debian-like Linux desktop environment
- GUI stack: `PyQt5` + X11/Wayland
- Entry points:
  - `./start_gui.sh`
  - `python -m zstack_anno`

## 2. Prerequisites

You must run in a GUI session (local desktop, X11 forwarding, or Wayland with
display access). Pure headless SSH without display will not open the UI.

### 2.1 System packages (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install -y \
  python3 python3-venv python3-pip \
  libgl1 libxkbcommon-x11-0 libxcb-xinerama0
```

If you see `Qt platform plugin "xcb"` errors, install additional runtime libs:

```bash
sudo apt install -y \
  libxcb1 libx11-xcb1 libxcb-render0 libxcb-shape0 libxcb-xfixes0 \
  libxcb-randr0 libxcb-keysyms1 libxcb-image0 libxcb-icccm4 libxcb-sync1 \
  libxcb-xkb1 libxkbcommon0 libxkbcommon-x11-0
```

## 3. Clone and checkout

```bash
git clone git@github.com:hansq7777/Myelin_anno_tool.git
cd Myelin_anno_tool
git checkout main
git pull origin main
```

## 4. Python environment setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt
```

## 5. Startup checks

### 5.1 Basic interpreter check

```bash
./start_gui.sh --check
```

### 5.2 Dependency import check

```bash
python - << 'PY'
import PyQt5
import zstack_anno
print("PyQt5 + zstack_anno import OK")
PY
```

## 6. Launch GUI

```bash
chmod +x start_gui.sh
./start_gui.sh
```

Or:

```bash
python -m zstack_anno
```

## 7. Functional smoke checklist (after window opens)

1. Window opens without Qt plugin error
2. `File -> Open` can load an image stack
3. Slice navigation works (`Up/Down` or toolbar)
4. Mask panel/overlay updates are visible
5. `Tools` menu opens Script Editor and Review entries

## 8. Troubleshooting

### 8.1 `Could not load the Qt platform plugin "xcb"`

- Install the packages listed in section 2.1
- Confirm GUI session variables are set:
  - `echo $DISPLAY`
  - `echo $WAYLAND_DISPLAY`

### 8.2 Runs over SSH but no window

- Use X11 forwarding (`ssh -X`) or run on local desktop session
- Verify remote host has GUI libs and active display

### 8.3 `ModuleNotFoundError`

- Ensure venv is activated: `source .venv/bin/activate`
- Reinstall deps: `pip install -r requirements.txt`

### 8.4 Slow morphology operations

- Ensure `scipy` and `scikit-image` are installed (already in requirements)
- The app will fallback to slower NumPy paths when optional packages are missing

## 9. One-line quickstart

```bash
sudo apt update && sudo apt install -y python3 python3-venv python3-pip libgl1 libxkbcommon-x11-0 libxcb-xinerama0 && \
python3 -m venv .venv && source .venv/bin/activate && pip install -U pip && pip install -r requirements.txt && \
./start_gui.sh
```

