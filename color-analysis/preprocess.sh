#!/bin/bash

# # Check if an argument is provided
# if [ "$#" -ne 1  ]; then
#     echo "Usage: $0 <image_file>"
#     exit 1
# fi

# Assign the argument to a variable

input_image=$1
output_path=$2
# output_path2=$3

# Intermediate directory for temporary files
intermediate_dir="../processed_images"
mkdir -p "$intermediate_dir"  # Create intermediate directory if it doesn't exist

python3 ../../color-correction/PyTorch/demo_single_image.py --i "$input_image" -t AWB  -o "../../combined-demo/processed_images"

mv "processed_images/temp.png" "$output_path"

# Run facial detection
python3 ../../face-detect/facedec/facedetect.py --i  "$output_path"

mv "../output-imgs/cropped.jpg" "$output_path"
