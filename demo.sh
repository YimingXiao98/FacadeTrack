#!/usr/bin/env bash
set -euo pipefail

# Demo: generate placeholder images and run the VLM pipeline on samples

echo "[1/2] Generating placeholder images for sample IDs..."
python -m scripts.generate_sample_images \
  --csv samples/csv/sample_matches_orientation.csv \
  --id-col ObjectId \
  --out samples/images/frames_dewarped

echo "[2/2] Running VLM pipeline on sample data..."
mkdir -p ./building_occupancy_result
python -m cli vlm \
  samples/csv/sample_matches_orientation.csv \
  samples/images/frames_dewarped \
  ./building_occupancy_result/sample_predictions.csv \
  --gt_csv samples/csv/sample_ground_truth.csv \
  --image_id_column ObjectId

echo "Done. Results: ./building_occupancy_result/sample_predictions.csv"
