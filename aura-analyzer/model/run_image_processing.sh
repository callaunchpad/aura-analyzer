#!/bin/bash

# Check if an argument is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <image_file>"
    exit 1
fi

# Assign the argument to a variable
IMAGE_FILE=$1

# Define directories relative to the script's location
BASE_DIR=$(dirname "$0")  # The directory where the script is located
INPUT_DIR="$BASE_DIR/../input-imgs"
INTERMEDIATE_DIR="$BASE_DIR/../intermediate-imgs"
OUTPUT_DIR="$BASE_DIR/../output-imgs"

# Ensure directories exist
mkdir -p "$INPUT_DIR" "$INTERMEDIATE_DIR" "$OUTPUT_DIR"

# Check if the image is in Downloads, and copy it if it exists there
DOWNLOADS_PATH="$HOME/Downloads/$IMAGE_FILE"
if [ -f "$DOWNLOADS_PATH" ]; then
    echo "Copying $IMAGE_FILE from Downloads to input-imgs..."
    cp "$DOWNLOADS_PATH" "$INPUT_DIR"
fi

# Check if the input file now exists in input-imgs
if [ ! -f "$INPUT_DIR/$IMAGE_FILE" ]; then
    echo "Error: Input file $IMAGE_FILE does not exist in input-imgs or Downloads."
    exit 1
fi

# Run color correction
echo "Running color correction on $IMAGE_FILE..."
python3 "$BASE_DIR/../../color-correction/PyTorch/demo_single_image.py" --i "$INPUT_DIR/$IMAGE_FILE" -t AWB -o "$INTERMEDIATE_DIR"

# Set the expected temp file
TEMP_FILE="$INTERMEDIATE_DIR/temp.png"

# Verify the color correction step created the temp file
if [ ! -f "$TEMP_FILE" ]; then
    echo "Error: Color correction failed to produce $TEMP_FILE."
    exit 1
fi

# Run facial detection
echo "Running facial detection on $TEMP_FILE..."
python3 "$BASE_DIR/../../face-detect/facedec/facedetect.py" --i "$TEMP_FILE"

# Set the expected cropped file output from face detection
CROPPED_FILE="$OUTPUT_DIR/cropped.jpg"

# Check if facial detection produced the cropped file
if [ ! -f "$CROPPED_FILE" ]; then
    echo "Error: Facial detection failed to produce $CROPPED_FILE."
    exit 1
fi

# Run color analysis
echo "Running color analysis on $CROPPED_FILE..."
python3 "$BASE_DIR/../../color-analysis/color_analysis.py" --i "$CROPPED_FILE"

# Run the season prediction model
echo "Running season prediction on $CROPPED_FILE..."
python3 "$BASE_DIR/new_model.py" "$CROPPED_FILE"

echo "Processing complete."

# Instructions for running the script:
# chmod +x run_image_processing.sh
# ./run_image_processing.sh jenna.jpg
