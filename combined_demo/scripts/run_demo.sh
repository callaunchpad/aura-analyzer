#!/bin/bash

# Check if an argument is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <image_file>"
    exit 1
fi

# Assign the argument to a variable
IMAGE_FILE=$1

# Run the Python script with the JPEG file
python3 ../../color_correction/PyTorch/demo_single_image.py --i "../input-imgs/$IMAGE_FILE" -t AWB  -o ../intermediate-imgs 

# Run facial detection
python3 ../../face_detect/facedec/facedetect.py --i  "../intermediate-imgs/white-balanced.jpg"

# Run color analysis
python3 ../../color_analysis/color_analysis.py --i "../output-imgs/cropped.jpg"

# run below script in command line to test 
# ./run_demo.sh [image-name.jpg]
# example: ./run_demo.sh IMG_6119.jpg