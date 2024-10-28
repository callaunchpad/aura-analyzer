#!/bin/bash

# Check if an argument is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <image_file>"
    exit 1
fi

# Assign the argument to a variable
IMAGE_FILE=$1

# Run the Python script with the JPEG file
python3 ../../color-correction/PyTorch/demo_single_image.py --i "../input-imgs/$IMAGE_FILE" -t AWB  -o ../intermediate-imgs 

# Run facial detection
python3 ../../face-detect/facedec/facedetect.py --i  "../intermediate-imgs/temp.png"
