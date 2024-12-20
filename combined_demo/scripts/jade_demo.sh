#!/bin/bash

# Check if an argument is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <image_file>"
    exit 1
fi

# Assign the argument to a variable
IMAGE_FILE=$1

# Run the Python script with the JPEG file
python ../../color_correction/PyTorch/demo_single_image.py --i "../input-imgs/$IMAGE_FILE" -t AWB  -o ../intermediate-imgs 

# Run facial detection
python ../../face_detect/facedec/facedetect.py --i  "../intermediate-imgs/white-balanced.jpg"

# Run color analysis
python ../../color_analysis/color_analysis.py --i "../output-imgs/cropped.jpg"

# run below script in command line to test 
# ./jade_demo.sh [image-name.jpg]
# example: ./jade_demo.sh IMG_6119.jpg
