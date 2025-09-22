# FacadeTrack — Street‑Level Occupancy Inference Toolkit

FacadeTrack is a lightweight, modular toolkit that implements the end‑to‑end street‑level pipeline described in the paper ***FacadeTrack: Linking Street View Imagery and Language Models for Post-Disaster Recovery***. It converts panoramic drive‑through video + GPS into rectified facade views and uses a vision‑language pipeline to infer parcel‑level occupancy with interpretable attributes and change analysis.

## What This Toolkit Provides

- Street‑level data prep: match building footprints to GPS, extract video frames, estimate heading, and dewarp panoramic frames to facade‑centered views.
- Interpretable VLM pipeline: elicits nine human‑readable attributes from each facade image and applies decision logic to output Occupied/Not Occupied.
- Two operating modes: a transparent one‑stage rule and a conservative two‑stage (vision + reasoning) strategy, consistent with the paper.
- Change analysis ready: produce per‑visit labels that can be aggregated into Recovered / Deteriorated / Stable classes.

## Components

- `vrtoolkit/spatial_matching.py` — Match footprints to GPS tracks (frame‑number based).
- `vrtoolkit/frame_extraction.py` — Extract frames with FFmpeg.
- `vrtoolkit/orientation.py` — Estimate vehicle heading per matched frame.
- `vrtoolkit/dewarp.py` — Dewarp panoramic frames to facade‑centered rectilinear views.
- `vrtoolkit/vision.py` — Vision‑language helpers (uses env vars for keys/models).
- `vrtoolkit/VLMpipeline.py` — Attribute extraction + decision logic (one‑stage and two‑stage).
- `vrtoolkit/cli.py` — Command‑line interface wiring the steps.
- `samples/` — Minimal CSVs and image stubs for quick trial.

## Paper Summary

- Goal: infer post‑disaster building occupancy at parcel scale from street‑view imagery with interpretable evidence and change tracking.
- Method: fuse street‑level, facade‑rectified views with a vision‑language prompt that outputs nine attributes, then decide Occupied/Not Occupied via either a simple scoring rule (one‑stage) or a conservative text‑reasoning step (two‑stage).
- Why it helps: nadir imagery misses facade/access cues; this street‑level, language‑guided pipeline surfaces those cues and explains decisions with intermediate attributes.

### Key Attributes

`house_destruction, structural_damage, exterior_debris, open_doors_windows, site_accessible, exterior_mud, emergency_markings, major_repairs, vehicle_presence`

### Results Snapshot

- Not Occupied treated as positive class. On the evaluation set, the two‑stage strategy showed higher recall and agreement with a modest precision trade‑off:
  - Precision: one‑stage 0.943, two‑stage 0.927
  - Recall: one‑stage 0.728, two‑stage 0.781
  - F1: one‑stage 0.822, two‑stage 0.848
  - Cohen’s κ: one‑stage 0.789, two‑stage 0.818
- McNemar and paired bootstrap tests indicated differences were not statistically significant at the current operating point, but the two‑stage mode better matched net recovery counts in change analysis.


## Install

- System: FFmpeg (`apt-get install ffmpeg` or `brew install ffmpeg`).
- Python:

```bash
pip install -r requirements.txt
```

## CLI

Run options:

- From repo root (this folder):
  - `python -m cli <command> [args]`
- After editable install (`pip install -e .`):
  - `vrtoolkit <command> [args]`

Commands:

```bash
match REF_CSV GPS_FOLDER OUT_CSV [--buffer 25.0]
extract MATCHES_CSV VIDEO_ROOT OUTPUT_DIR
orient MATCHES_CSV GPS_FOLDER OUT_CSV [--window 15]
dewarp CSV INPUT_DIR OUTPUT_DIR [--h_fov 90 --pitch 0 --width 1920 --aspect 16:9 --yaw_offset -90]
dewarp-smart CSV INPUT_DIR OUTPUT_DIR [same options as dewarp]
vlm CSV IMAGE_DIR OUT_CSV [--no-llm] [--gt_csv FILE] [--image_id_column COL]
```

## Typical Workflow

Quick demo (using included samples):

Run the one-liner demo script from the repo root:

```bash
bash demo.sh
```

Or run the two steps manually:

1) Generate placeholder images for sample IDs

```bash
python -m scripts.generate_sample_images --csv samples/csv/sample_matches_orientation.csv --id-col ObjectId --out samples/images/frames_dewarped
```

2) Run VLM pipeline on sample data

```bash
python -m cli vlm samples/csv/sample_matches_orientation.csv samples/images/frames_dewarped ./building_occupancy_result/sample_predictions.csv --gt_csv samples/csv/sample_ground_truth.csv --image_id_column ObjectId
```

Full workflow (your data):

1) Match footprints to GPS frames

```bash
python -m cli match <your_footprints.csv> <GPS_FOLDER> matched_filtered_buildings.csv
```

2) Extract frames from videos

```bash
python -m cli extract matched_filtered_buildings.csv <VIDEO_ROOT_DIR> ./frames_output
```

3) Compute orientations

```bash
python -m cli orient matched_filtered_buildings.csv <GPS_FOLDER> matched_filtered_buildings_orientation.csv
```

4) Dewarp frames

```bash
python -m cli dewarp matched_filtered_buildings_orientation.csv ./frames_output ./frames_dewarped --h_fov 90 --pitch 0 --width 1920 --aspect 16:9 --yaw_offset -90
```

5) Run VLM pipeline on dewarped frames

```bash
python -m cli vlm matched_filtered_buildings_orientation.csv ./frames_dewarped ./building_occupancy_result/occupancy.csv --gt_csv ./building_occupancy_result/ground_truth.csv
```

## Samples

See `samples/`:

- Example CSVs in `samples/csv/` show required columns for various stages.
- Place example images as `<ObjectId>.jpg` under `samples/images/frames_dewarped/` to test the VLM pipeline:

```bash
python -m cli vlm samples/csv/sample_matches_orientation.csv samples/images/frames_dewarped ./building_occupancy_result/sample_predictions.csv --gt_csv samples/csv/sample_ground_truth.csv --image_id_column ObjectId
```

### Generate Placeholder Images

You can generate placeholder images (so you can run the VLM pipeline without real imagery) using:

```python
python -m scripts.generate_sample_images --csv samples/csv/sample_matches_orientation.csv --id-col ObjectId --out samples/images/frames_dewarped
```

Or for specific IDs:

```python
python -m scripts.generate_sample_images --ids 1 2 3 --out samples/images/frames_dewarped
```

## Data Formats

- GPS_FOLDER (example: `GPS/` or your own path)
  - Contains one CSV per video: `<video_basename>_GoPro Max-GPS5.csv`
  - Required columns (headers):
    - `GPS (Long.) [deg]`
    - `GPS (Lat.) [deg]`
  - One row per frame, in chronological order; row index is treated as `frame_number`.
  - Example sample file: `samples/GPS/GH010001_GoPro Max-GPS5.csv`.

- VIDEO_ROOT_DIR (example: `videos/` or your own path)
  - Contains video files with basenames matching `matched_file` (from the match step).
  - Supported extensions: `.mp4`, `.MP4`, `.mov`, `.MOV` (also tries GH/GL name swaps).
  - Example expected names: `GH010001.mp4` or `GH010001.MOV`.

- Footprints CSV (input to `match`)
  - Must contain longitude/latitude columns; auto-detected from common names:
    - (`long`,`lat`) or (`longitude`,`latitude`) or (`x`,`y`) or (`Center_Longitude`,`Center_Latitude`).
  - Example: `samples/csv/sample_footprints.csv`.

- Matches CSV (output of `match`; input to `extract`/`orient`)
  - Required columns:
    - `ObjectId`, `Center_Longitude`, `Center_Latitude`, `vehicle_x`, `vehicle_y`, `matched_file`, `frame_number`.
  - Example: `samples/csv/sample_matches.csv`.

- Orientation CSV (output of `orient`; input to `dewarp`/`vlm`)
  - Same as Matches CSV plus `orientation` column.
  - Example: `samples/csv/sample_matches_orientation.csv`.

Directory placeholders included in this repo:

```bash
GPS/            # place your real GPS CSVs here (or point to your own folder)
videos/         # place your videos here (or point to your own folder)
```

## APIs

Set environment variables (your own OpenAI API keys) before using `videoprocessing.vision` or `vlm`:

```bash
export OPENAI_API_KEY=your_key
export OPENAI_VISION_MODEL=gpt-4o
export OPENAI_TEXT_MODEL=gpt-4o
```

## Notes

- Frame‑number‑based processing avoids timestamp drift.
- Column names are normalized where feasible; otherwise errors are explicit.
- The VLM stage can run in two modes (one‑stage vs two‑stage). Choose based on recall needs and operating conservatism.


