#!/bin/bash

# delete all files in combined_demo/output-imgs and combined_demo/intermediate-imgs
rm -rf combined_demo/output-imgs/*
rm -rf combined_demo/intermediate-imgs/*

# Check if an argument is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <image_file>"
    exit 1
fi

# Assign the argument to a variable
IMAGE_FILE=$1

#Check if  file exists
if [ ! -f "combined_demo/input-imgs/$IMAGE_FILE.jpg" ]; then
    echo "Error: File '$IMAGE_FILE' does not exist."
    exit 1
fi

#Checks if file is an image
if ! file "combined_demo/input-imgs/$IMAGE_FILE.jpg" | grep -iE 'image|bitmap'; then
    echo "Error: '$image_file' is not a valid image file."
    exit 1
fi

# Run the Python script with the JPEG file
python3 color_correction/PyTorch/demo_single_image.py --i "combined_demo/input-imgs/$IMAGE_FILE.jpg" -t AWB -o "combined_demo/intermediate-imgs/" --file_name $IMAGE_FILE

# Run facial detection
python3 face_detect/facedec/facedetect.py -i  "combined_demo/intermediate-imgs/$IMAGE_FILE-awb.jpg" --file_name $IMAGE_FILE
if [ $? -eq 0 ]; then
    echo "Script ran successfully"
else
    echo "Script encountered an error"
fi

# Run color analysis
python3 color_analysis/color_analysis_model.py --i "combined_demo/output-imgs/$IMAGE_FILE-cropped.jpg" --file_name $IMAGE_FILE

# run below script in command line to test 
# ./run_demo.sh [image-name.jpg]
# example: ./run_demo.sh IMG_6119.jpg
