#!/bin/bash
set -e

TEST_DATA_DIR=$1
PRED_DIR=$2

echo "📂 Test Data Directory: $TEST_DATA_DIR"
echo "📂 Prediction Output Directory: $PRED_DIR"

python3 entrypoints/infer.py --test_data_dir "$TEST_DATA_DIR" --pred_dir "$PRED_DIR"

 
