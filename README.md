# Golf Swing Biomechanics Analyzer

A computer vision system that analyzes a golf swing video (or any full-body movement) and produces:
- A colored skeleton overlay on the subject (cyan = lead leg, orange = trail leg, green = spine)
- A live dashboard of real biomechanical angles: spine tilt, knee flexion (both legs), hip and shoulder rotation, and X-Factor
- Automatic swing phase detection (Address / Backswing / Top / Downswing / Impact / Follow-Through)

Built entirely with vector math and a lightweight state machine — **no custom machine learning model**. MediaPipe is used only for body landmark extraction.

## Project Structure

```
golf_swing_analyzer/
├── config.py             # All constants and settings (colors, sensitivity, bar ranges...)
├── pose_extraction.py    # Part 1 - Body landmark extraction (MediaPipe Tasks API)
├── skeleton_drawing.py   # Part 2 - Colored skeleton overlay
├── geometry_engine.py    # Part 3 - Spine tilt and knee flexion calculations
├── rotation_engine.py    # Part 4 - Hip/shoulder rotation and X-Factor
├── phase_detector.py     # Part 5 - Swing phase detector (state machine)
├── dashboard.py          # Part 6 - Live dashboard rendering
├── pipeline.py           # Part 7 - Full pipeline assembly
├── main.py               # Command-line entry point
├── requirements.txt
└── models/               # Place the model file here (see below)
```

## Installation

```bash
python -m venv venv
venv\Scripts\activate        # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Downloading the MediaPipe Model (one-time, required)

Recent versions of MediaPipe require downloading the model file separately — it is not bundled with the pip package:

```bash
# Pick one, depending on your needs:
# lite  = fastest, lower accuracy | full = balanced (recommended) | heavy = most accurate, slower

wget -O models/pose_landmarker_full.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task
```

If `wget` isn't available on Windows, open the link in your browser, download it manually, and place it in the `models/` folder.

## Usage

```bash
python main.py --video path/to/your_swing.mp4
```

The output video (`output_with_dashboard.mp4` by default) is saved in the same directory.

Additional options:
```bash
python main.py --video swing.mp4 --output result.mp4 --model models/pose_landmarker_lite.task
```

## Important Notes Before Running

1. **The input video doesn't have to be golf specifically.** The pipeline tracks any full-body movement — every calculation is pure geometry and motion, not an ML classification of "this is golf."
2. **`HAND_SPEED_THRESHOLD` in `config.py`** (default: `8`) needs to be tuned to your video's resolution and frame rate. If phase transitions (Backswing/Top/...) happen too early or too late compared to the actual movement, try raising or lowering this value.
3. If the phase never advances past `ADDRESS`: the velocity the code is computing is staying below `HAND_SPEED_THRESHOLD` — lower the value.
4. The pipeline assumes the first valid frame of the video is the Address position (the player standing still before starting the motion). If the video begins mid-motion, the computed metrics will be affected.

## File Overview

| File | Responsibility |
|---|---|
| `pose_extraction.py` | Converts any video frame into 33 body landmarks (pixel space + real-world meters) |
| `skeleton_drawing.py` | Draws colored lines connecting the joints on each frame |
| `geometry_engine.py` | Computes angles from pixel coordinates (simple vector math) |
| `rotation_engine.py` | Computes body rotation from real-world (metric) coordinates |
| `phase_detector.py` | Determines the swing phase from hand movement speed |
| `dashboard.py` | Renders the side panel (labels + bars) |
| `pipeline.py` | Ties everything together into one loop that reads the video and exports the result |
| `main.py` | Command-line entry point |
