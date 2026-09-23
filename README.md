# Golf Swing Analyzer

A computer-vision pipeline that watches a golf swing video and reports the
biomechanics behind it — plus a local LLM agent that can answer natural-language
questions about the resulting data. No wearables, no motion-capture suit: pose
estimation from a single camera angle.

```
raw video  ──▶  vision/  ──▶  annotated video + CSV dataset  ──▶  agent/  ──▶  chat
              (pose, angles,                                   (Qwen3-4B,
               phase detection,                                 tool-calling,
               live dashboard)                                  safety layers)
```

The two halves are fully independent. `vision/` never imports anything from
`agent/`, and `agent/` never touches OpenCV/MediaPipe — the CSV file is the
entire handoff. You can run swing analysis on a laptop with no GPU at all,
and you can develop or test the agent against any CSV that matches the
schema, without a camera or a video in sight.

---

## Features

### `vision/` — swing analysis
- **Pose extraction** — 33-point body pose via MediaPipe Pose Landmarker, in
  both pixel space and real-world meters.
- **Colored skeleton overlay** — lead leg/arm, trail leg/arm, and spine drawn
  distinctly on the video.
- **Joint-angle geometry engine** — spine tilt, knee flexion (both legs),
  elbow flexion (both arms) from plain vector math, with **every metric
  gated on landmark visibility** — a low-confidence joint reports `None`
  instead of a guessed number.
- **Rotation & X-Factor engine** — hip rotation, shoulder rotation, and their
  separation (X-Factor), computed from world-space coordinates so it holds
  up even though the camera only sees 2D.
- **Swing-phase state machine** — `READY → ADDRESS → BACKSWING → TOP →
  DOWNSWING → IMPACT → FOLLOW_THROUGH → READY`, driven by rotation velocity
  and hand height, with:
  - automatic address-position calibration (no manual setup step)
  - waggle cancellation (a practice waggle doesn't get counted as a swing)
  - scene-cut detection (a replay/angle change doesn't get parsed as a swing)
  - per-phase timeouts so a stuck detector always self-resets
  - swing numbering that survives a body-tracking dropout mid-video
- **Quality scoring** — each metric at TOP / IMPACT / FOLLOW_THROUGH is
  classified against literature-based **PRO** / **AMATEUR** / **OUT** ranges,
  rolled up into a per-swing `quality_score_pct` (plus a
  `quality_confidence_pct` that tells you how many metrics that score is
  actually based on — a 100% score from 2 measured metrics isn't the same
  claim as 100% from 12).
- **Live dashboard overlay** — phase indicator + labeled, colored bars for
  every metric, composited next to the video frame.
- **Dataset export** — one CSV per video, plus a de-duplicated master CSV
  across every video/player you've analyzed, with a `*_class` (PRO/AMATEUR/OUT)
  column alongside every metric, tempo computed from raw frame counts (not
  rounded seconds), and a comparison plot (`.png`) across a video's swings.

### `agent/` — the "Golf Swing Analyst"
- **Local LLM, tool-calling only** — Qwen3-4B-Instruct (4-bit quantized) is
  never allowed to state a number it didn't get from a tool call; it can
  only *read* the dataset through four whitelisted, read-only tools
  (`query_stats`, `compare`, `get_catalog`, `get_table`).
- **Sandboxed expression language** — `query_stats` accepts small expressions
  like `slope(top_x_factor_deg)` or `last(quality_score_pct) - first(...)`,
  validated with a deny-by-default AST walk (not `eval`) before anything
  runs: unknown functions, unknown columns, and anything resembling code
  execution are rejected before the model's request ever touches the data.
- **Three-layer safety pipeline**:
  1. **L1 — request screen**: banned-phrase check (prompt-injection,
     jailbreak attempts, path/file access, requests to read or edit the
     source) on the raw question, before any model call.
  2. **L2 — tool-call validation**: every tool call the model proposes is
     parsed and checked against the real dataset (known columns, known
     players) — a malformed or out-of-scope call is rejected and the model
     is told exactly why, so it can self-correct.
  3. **L3 — numbers guard**: every number in the model's final answer must
     be traceable to an actual tool result (within a small rounding
     tolerance) or the answer is rejected and regenerated.
  - **Bilingual by design** — the request screen and scope detection run in
    **Arabic and English** in parallel, since that's how the intended users
    actually write.
- **Conversation memory** — short per-player notes persisted to
  `agent_memory.json` across sessions (used only as a *hint*; the model
  still has to verify every number via a tool call every time), plus
  same-session context for short follow-up replies ("amir" typed alone
  after the agent asks "which player?").
- **Graceful degradation** — a hard per-question attempt budget with a
  fallback: if the model can't land a clean final answer in time, the agent
  summarizes whatever real tool results it already collected instead of
  just failing outright.
- **13-scenario compliance suite** (`scripts/run_agent_tests.py`) covering
  scope detection, tool use, refusals, and clarification requests, in both
  languages.

---

## Project structure

```
golf-swing-analyzer/
├── vision/                    # computer-vision pipeline (no agent code, no LLM deps)
│   ├── config.py              #   every CV constant in one place
│   ├── model_loader.py        #   downloads/caches the MediaPipe pose model
│   ├── pose.py                #   Part 1 — pose landmark extraction
│   ├── skeleton.py            #   Part 2 — skeleton overlay drawing
│   ├── geometry.py            #   Part 3 — joint-angle math
│   ├── rotation.py            #   Part 4 — hip/shoulder rotation, X-Factor
│   ├── phase_detector.py      #   Part 5 — swing-phase state machine
│   ├── quality.py             #   Part 5A — PRO/AMATEUR quality standards
│   ├── export.py              #   Part 5B — CSV dataset export + plots
│   ├── dashboard.py           #   Part 6 — live dashboard overlay
│   └── pipeline.py            #   Part 7 — main video-processing loop
│
├── agent/                     # the LLM analyst (no OpenCV/MediaPipe deps)
│   ├── config.py              #   model settings + the system prompt template
│   ├── llm.py                 #   local model backend (load + generate)
│   ├── data_loader.py         #   loads the CSV, builds the injected prompt
│   ├── security.py            #   L1/L2/L3 safety layers
│   ├── tools.py                #   the 4 whitelisted read-only data tools
│   ├── memory.py               #   durable + session conversation memory
│   ├── engine.py                #   GolfSwingAgent — the main ask() loop
│   ├── chat.py                  #   interactive terminal chat loop
│   └── tests.py                 #   13-scenario compliance suite
│
├── scripts/
│   ├── analyze_video.py        # CLI: video → annotated video + CSV dataset
│   ├── run_agent_chat.py       # CLI: start an interactive chat session
│   └── run_agent_tests.py      # CLI: run the compliance suite
│
├── tests/                      # fast, GPU-free unit tests (pytest)
│   ├── test_vision_geometry.py #   angle/rotation math
│   └── test_agent_security.py  #   security layer + tool execution
│
├── requirements.txt             # vision/ dependencies
├── requirements-agent.txt       # + agent/ dependencies (torch, transformers...)
├── requirements-dev.txt         # + pytest
└── pytest.ini
```

---

## Installation

```bash
git clone <your-repo-url>
cd golf-swing-analyzer
python -m venv .venv && source .venv/bin/activate   # optional but recommended

pip install -r requirements.txt              # vision/ only
pip install -r requirements-agent.txt         # + the agent (needs a GPU for real-time replies)
pip install -r requirements-dev.txt           # + to run the test suite
```

The MediaPipe pose model is **not** bundled in the repo — `scripts/analyze_video.py`
downloads and caches it on first run.

---

## Usage

### 1. Analyze a swing video

```bash
python scripts/analyze_video.py path/to/swing.mp4

# with options
python scripts/analyze_video.py path/to/swing.mp4 \
    --player ahmed \
    --output ahmed_annotated.mp4 \
    --model-variant full \
    --actual-swings 11        # optional: prints detection accuracy vs. ground truth
```

This writes:
- an annotated video with the skeleton overlay + live dashboard side by side
- `swing_data/swings_<player>.csv` — this video's swings
- `swing_data/all_swings_dataset.csv` — the accumulating master dataset (what the agent reads)
- `swing_data/comparison_<player>.png` — a comparison plot across the video's swings

Run it again for more players/videos — the master CSV accumulates, replacing
only that player's rows each time you re-analyze their video.

### 2. Chat with the analyst

```bash
python scripts/run_agent_chat.py
```

```
you: متوسط لفة الكتف عند القمة لـ ahmed كام؟
agent (2.1s | attempts: 2): متوسط لفة الكتف عند القمة لـ ahmed هو 92.4 درجة — ضمن نطاق المحترفين (85-110).

you: what's his tempo trend across the swings?
agent (1.8s | attempts: 2): Tempo is trending up slightly (slope +0.03/swing) toward the pro range of 2.5-3.5:1 — steady rhythm, no red flags.
```

### 3. Run the compliance suite

```bash
python scripts/run_agent_tests.py
```

### 4. Run the unit tests (no GPU, no dataset needed)

```bash
python -m pytest
```

---

## Dataset schema

One row per swing in `swing_data/all_swings_dataset.csv`:

| Column | Meaning |
|---|---|
| `player`, `video`, `swing_id`, `swing_uid` | identity (`swing_uid` = e.g. `ahmed#3`) |
| `start_s` / `top_s` / `down_s` / `impact_s` / `end_s` | phase timestamps (seconds) |
| `backswing_time_s`, `downswing_time_s`, `tempo_ratio` | timing (pro tempo ≈ 2.5–3.5:1) |
| `top_shoulder_rotation_deg`, `top_hip_rotation_deg`, `top_x_factor_deg`, `top_spine_tilt_deg`, `top_lead_knee_flexion_deg`, `top_trail_knee_flexion_deg` | metrics at the top of the backswing |
| `impact_hip_rotation_deg`, `impact_spine_tilt_deg`, `impact_lead_knee_flexion_deg`, `impact_lead_elbow_flexion_deg` | metrics at impact |
| `follow_shoulder_rotation_deg`, `follow_hip_rotation_deg` | metrics at the finish |
| `*_class` (one per metric above) | `PRO` / `AMATEUR` / `OUT` |
| `quality_score_pct` | % of measured metrics that landed in the PRO range |
| `quality_confidence_pct` | % of all *possible* metrics that were actually measured this swing (low value = take the score with a grain of salt — a joint was occluded a lot) |
| `shallow_top_flag` | `True` if the backswing barely cleared the TOP threshold |

Rotation angle **sign** is an axis convention, not a quality signal — the
agent's system prompt explicitly teaches it to read magnitude against the
pro range and describe the physical meaning (e.g. "coiled 90° at the top"),
not to treat a negative follow-through value as a decline.

---

## Notes on the two fixes made while restructuring

While splitting the original single-file script into this package, two real
bugs were fixed (not just reorganized):

1. **The response timeout was dead code.** `TimeoutStoppingCriteria` was
   defined but never actually passed into `model.generate()`, so the
   documented 15-second cap on a reply never took effect. It's now wired in
   (`agent/llm.py`).
2. The security/scope regular expressions are bilingual (Arabic + English)
   on purpose, since that's the real usage pattern — this was preserved
   carefully during the split rather than accidentally narrowed to English.

---

## Requirements

- Python 3.10+
- `vision/`: CPU is fine; a GPU speeds up MediaPipe but isn't required.
- `agent/`: a CUDA GPU with ~6 GB+ VRAM is strongly recommended for the
  4-bit Qwen3-4B model. It will run on CPU, but replies take minutes instead
  of seconds.

## License

No license file is included yet — add one (MIT is a common default for a
project like this) before you rely on others being able to reuse the code.
