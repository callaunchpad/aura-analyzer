#!/bin/bash

# Source directory containing files
SOURCE_DIR="fashion-dataset/images"

# Target directory to copy files
TARGET_DIR="fashion-dataset-small/images"

# Number of files to sample
NUM_FILES=1000

# Create the target directory if it doesn't exist
mkdir -p "$TARGET_DIR"

# Get a list of all files in the source directory
FILES=("$SOURCE_DIR"/*)

# Check if there are enough files to sample
if [[ ${#FILES[@]} -lt $NUM_FILES ]]; then
  echo "Not enough files in $SOURCE_DIR to sample $NUM_FILES files."
  exit 1
fi

# Randomly sample files using awk
SELECTED_FILES=$(printf "%s\n" "${FILES[@]}" | awk -v num=$NUM_FILES 'BEGIN {srand()} {print rand(), $0}' | sort -n | head -n "$NUM_FILES" | cut -d' ' -f2-)

# Copy the selected files to the target directory
while IFS= read -r FILE; do
  cp "$FILE" "$TARGET_DIR"
done <<< "$SELECTED_FILES"

echo "Successfully copied $NUM_FILES random files to $TARGET_DIR."
