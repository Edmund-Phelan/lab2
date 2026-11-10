# Source videos

Original phone recordings (Samsung Galaxy Z Fold6, main camera, 1920x1080, landscape),
kept so every file in `in/` and `out/` can be rebuilt.

| File | Original name | Becomes |
|---|---|---|
| `01_calibration_raw.mp4` | `20261110_122051.mp4` | `in/01_calibration.mp4`, `in/1.png` ... `in/20.png` |
| `01_01_raw.mp4` | `20261110_132912.mp4` | `in/01_01.mp4` |

Rebuild (from the repo root):

```
# Shrink the calibration video to 640x360 and pick 20 calibration frames
python -m mypylibs.prepare_calibration assets/source/01_calibration_raw.mp4 1

# Shrink the scene video with the same pre-processing
python -c "from mypylibs.video import preprocess_video; preprocess_video('assets/source/01_01_raw.mp4', 'in/01_01.mp4')"

# Calibrate (writes out/01_K.csv, out/01_D.csv)
python main.py

# Visualisations
python -m mypylibs.visualize in/01_calibration.mp4 1      # out/01_calibration_corners.mp4
python -m mypylibs.visualize_scene in/01_01.mp4           # out/01_01_motion.mp4
```
