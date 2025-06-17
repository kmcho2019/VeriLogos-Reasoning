#!/bin/bash

# A simple script to split a file containing a list of modules into N smaller files.
# This is useful for running multiple instances of the HDL generator in parallel.

# --- Argument Validation ---

# Check if exactly two arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <input_file> <number_of_splits>"
    echo "Example: ./split_resume_file.sh my_unfinished_modules.txt 4"
    exit 1
fi

INPUT_FILE="$1"
NUM_SPLITS="$2"

# Check if the input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file '$INPUT_FILE' not found."
    exit 1
fi

# Check if the number of splits is a positive integer
if ! [[ "$NUM_SPLITS" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: Number of splits must be a positive integer."
    exit 1
fi

# --- Splitting Logic ---

# Get the total number of lines in the input file
total_lines=$(wc -l < "$INPUT_FILE")

if [ "$total_lines" -eq 0 ]; then
    echo "Warning: Input file '$INPUT_FILE' is empty. No files will be created."
    exit 0
fi

# Calculate the number of lines per split file (using ceiling division)
# This ensures all lines are distributed, even if not perfectly divisible.
lines_per_file=$(( (total_lines + NUM_SPLITS - 1) / NUM_SPLITS ))

# Define the prefix for the output files
OUTPUT_PREFIX="${INPUT_FILE%.*}_part_"

echo "Input file: '$INPUT_FILE' ($total_lines lines)"
echo "Splitting into $NUM_SPLITS files (~$lines_per_file lines each)."
echo "Output files will have the prefix: '$OUTPUT_PREFIX'"

# Use the 'split' command to do the work.
# -l: Specifies the maximum number of lines per output file.
# --additional-suffix: Adds a .txt extension to the generated files.
# --numeric-suffixes=1: Creates files like part_01, part_02 instead of part_aa, part_ab
# The prefix is the last argument.
split -l "$lines_per_file" --numeric-suffixes=1 --additional-suffix=.txt "$INPUT_FILE" "$OUTPUT_PREFIX"

echo "--------------------------------------------------"
echo "Success! The file has been split into the following parts:"
ls -1 "${OUTPUT_PREFIX}"*.txt
echo "--------------------------------------------------"