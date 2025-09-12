# Video Processing Toolkit

This repository organizes the functionality from the original notebook into a small Python package with clear modules and a CLI.

## Overview

- `vrtoolkit/spatial_matching.py`: Match footprints to GPS tracks (frame-number based).
- `vrtoolkit/frame_extraction.py`: Extract frames with FFmpeg.
- `vrtoolkit/orientation.py`: Compute vehicle orientations per building.
- `vrtoolkit/dewarp.py`: Dewarp frames to center the target building.
- `vrtoolkit/vision.py`: OpenAI helpers (no keys hardcoded; uses env vars).
- `vrtoolkit/VLMpipeline.py`: VLM occupancy pipeline (script + callable function).
- `vrtoolkit/cli.py`: Command-line interface.
- `samples/`: Example CSVs and empty image folders to illustrate structure.

## Install

- System: FFmpeg (`apt-get install ffmpeg` or `brew install ffmpeg`).
- Python:

```bash
python -m venv .venv
. .venv/bin/activate   # Windows: .venv\\Scripts\\activate
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

```
python -m scripts.generate_sample_images --csv samples/csv/sample_matches_orientation.csv --id-col ObjectId --out samples/images/frames_dewarped
```

Or for specific IDs:

```
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

```
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

- Frame-number-based processing avoids timestamp drift.
- Column names are normalized where feasible; otherwise errors are explicit.
