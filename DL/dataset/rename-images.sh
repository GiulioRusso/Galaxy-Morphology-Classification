#!/bin/bash

# HOW TO LAUNCH THE SCRIPT
# 1. Specify the images directory IMAGE_DIR (commented for safety reasons)
# 2. With Terminal, go to the folder where this script is located
# 3. Use: 'chmod +x rename-images.sh' to make the script executable
# 4. Launch it with: './rename-images.sh'

# specify the directory containing the files (remember to add the '/' at the end of the path to specify it is a directory)
IMAGE_DIR=#""

index=0
for filename in $IMAGE_DIR/Galaxy10_DECals-dataset-*.png; do
    new_filename=$(printf "Galaxy10_DECals-dataset-%05d.png" $index)
    mv "$filename" "$IMAGE_DIR/$new_filename"
    index=$((index + 1))
done

# remove the file extension, and save to name-list.txt
ls "$IMAGE_DIR"/Galaxy10_DECals-dataset-* | sed 's/\.png$//' > annotations/filename-list.txt





