#!/bin/bash

# Check if an argument is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <image_file>"
    exit 1
fi

# Assign the argument to a variable
IMAGE_FILE=$1

# Run the Python script with the JPEG file
python3 ../../color-correction/PyTorch/demo_single_image.py --i "$IMAGE_FILE" -t AWB -o ../output-imgs
