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

# Ensure directories exist, if not, create them
mkdir -p "$INPUT_DIR" "$INTERMEDIATE_DIR" "$OUTPUT_DIR"

# Check if input file exists
if [ ! -f "$INPUT_DIR/$IMAGE_FILE" ]; then
    echo "Error: $INPUT_DIR/$IMAGE_FILE does not exist."
    exit 1
fi

# Run the Python script for color correction
echo "Running color correction on $IMAGE_FILE..."
python3 "$BASE_DIR/../../color-correction/PyTorch/demo_single_image.py" --i "$INPUT_DIR/$IMAGE_FILE" -t AWB -o "$INTERMEDIATE_DIR"

# Set the expected intermediate file output
TEMP_FILE="$INTERMEDIATE_DIR/temp.png"  

# Check if the intermediate temp file was created
if [ ! -f "$TEMP_FILE" ]; then
    echo "Error: $TEMP_FILE was not created by the color correction script."
    exit 1
fi

# Run facial detection on the temp image
echo "Running facial detection on $TEMP_FILE..."
python3 "$BASE_DIR/../../face-detect/facedec/facedetect.py" --i "$TEMP_FILE"

# Define the expected cropped file output from face detection
CROPPED_FILE="$OUTPUT_DIR/cropped.jpg"  

# Check if the cropped image was created
if [ ! -f "$CROPPED_FILE" ]; then
    echo "Error: $CROPPED_FILE was not created by the facial detection script."
    exit 1
fi

# Run color analysis on the cropped image
echo "Running color analysis on $CROPPED_FILE..."
python3 "$BASE_DIR/../../color-analysis/color_analysis.py" --i "$CROPPED_FILE"

# Run the season prediction using new_model.py
echo "Running season prediction on $IMAGE_FILE..."
python3 "$BASE_DIR/../../model/new_model.py" "$CROPPED_FILE"

echo "Processing complete."

# run below script in command line to test 
# chmod +x run_image_processing.sh
# ./run_image_processing.sh IMG_6119.jpg
