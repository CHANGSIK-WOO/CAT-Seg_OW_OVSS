#!/bin/bash

# Open-World CAT-Seg Demo Script
# Usage: ./demo.sh [model_weights] [input_path] [output_path]

set -e

# Default configuration
CONFIG_FILE="configs/ow_vitb_384_size_down.yaml"
MODEL_WEIGHTS=""
INPUT_PATH=""
OUTPUT_PATH=""

# Parse arguments
if [ $# -eq 0 ]; then
    echo "Usage: $0 [model_weights] [input_path] [output_path]"
    echo "  model_weights: Path to model checkpoint (.pth file)"
    echo "  input_path: Path to input image(s) or directory"
    echo "  output_path: (Optional) Path to save output images"
    echo ""
    echo "Examples:"
    echo "  $0 output/model_final.pth demo_images/"
    echo "  $0 output/model_final.pth demo_images/ output_vis/"
    echo "  $0 output/model_final.pth image.jpg"
    echo "  $0 output/model_final.pth --webcam"
    exit 1
fi

MODEL_WEIGHTS=$1
INPUT_PATH=$2
OUTPUT_PATH=${3:-""}

# Validate model weights
if [ ! -f "$MODEL_WEIGHTS" ]; then
    echo "Error: Model weights file not found: $MODEL_WEIGHTS"
    exit 1
fi

# Check if input is webcam
if [ "$INPUT_PATH" = "--webcam" ]; then
    echo "Starting webcam demo..."
    python demo/ow_demo.py \
        --config-file $CONFIG_FILE \
        --model-weights $MODEL_WEIGHTS \
        --webcam \
        --enable-ow-mode \
        --show-class-names \
        --show-unknown-regions \
        --alpha 0.7
    exit 0
fi

# Validate input path
if [ ! -e "$INPUT_PATH" ]; then
    echo "Error: Input path not found: $INPUT_PATH"
    exit 1
fi

OUTPUT_ARG=""
if [ -n "$OUTPUT_PATH" ]; then
    if [ -f "$OUTPUT_PATH" ]; then
        echo "Error: Output path exists as a file: $OUTPUT_PATH"
        echo "Please specify a directory name for output"
        exit 1
    fi

    if [ ! -d "$OUTPUT_PATH" ]; then
        echo "Creating output directory: $OUTPUT_PATH"
        mkdir -p "$OUTPUT_PATH"
    else
        echo "Using existing output directory: $OUTPUT_PATH"
    fi

    OUTPUT_ARG="--output $OUTPUT_PATH"
fi

echo "Running Open-World CAT-Seg Demo..."
echo "Config: $CONFIG_FILE"
echo "Model: $MODEL_WEIGHTS"
echo "Input: $INPUT_PATH"
echo "Output: ${OUTPUT_PATH:-'Display only'}"
echo ""

# Run demo
python demo/ow_demo.py \
    --config-file $CONFIG_FILE \
    --model-weights $MODEL_WEIGHTS \
    --input $INPUT_PATH \
    $OUTPUT_ARG \
    --enable-ow-mode \
    --show-class-names \
    --show-unknown-regions \
    --alpha 0.7 \
    --confidence-threshold 0.5

echo "Demo completed!"

# Additional commands for different scenarios
echo ""
echo "Additional usage examples:"
echo ""
echo "# Disable Open-World mode (known classes only):"
echo "python ow_demo.py --config-file $CONFIG_FILE --model-weights $MODEL_WEIGHTS --input $INPUT_PATH --opts MODEL.SEM_SEG_HEAD.ENABLE_OW_MODE False"
echo ""
echo "# Process video file:"
echo "python ow_demo.py --config-file $CONFIG_FILE --model-weights $MODEL_WEIGHTS --video-input video.mp4 --output output_video.mp4"
echo ""
echo "# Batch process with different settings:"
echo "python ow_demo.py --config-file $CONFIG_FILE --model-weights $MODEL_WEIGHTS --input 'images/*.jpg' --output results/ --alpha 0.5"