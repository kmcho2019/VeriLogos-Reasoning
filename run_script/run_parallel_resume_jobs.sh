#!/bin/bash

# This script runs 40 python commands in parallel, each with a different
# resume file, and logs their output to separate files.

# --- Configuration ---
# Base directory for resume files
RESUME_DIR="./resume_file"

# Base directory for logs
LOG_BASE_DIR="./resume_file/logs"

# The date string used in your file names
DATE_STR="20250609"

# The specific directory for today's logs
LOG_DIR="${LOG_BASE_DIR}/code_resume_file_${DATE_STR}"

# --- Script Start ---
echo "🚀 Starting parallel execution of 40 jobs..."

# Create the log directory if it doesn't exist
# The -p flag creates parent directories as needed and doesn't error if it already exists.
mkdir -p "$LOG_DIR"
echo "Logs will be stored in: $LOG_DIR"
echo "--------------------------------------------------"

# Loop from 1 to 40 to launch each job
for i in {1..40}
do
    # Format the number with a leading zero (e.g., 1 -> 01, 10 -> 10)
    num=$(printf "%02d" $i)

    # Construct the full path for the resume file and the log file
    resume_file="${RESUME_DIR}/code_resume_file_${DATE_STR}_part_${num}.txt"
    log_file="${LOG_DIR}/${num}.log"

    echo "Launching job #${num}:"
    echo "  - Resuming from: ${resume_file}"
    echo "  - Logging to:    ${log_file}"

    # Execute the python command in the background (&)
    # Redirect both standard output and standard error to the log file.
    python3 main.py GEN_HDL -im deepseek-reasoner -d test_code_output_mask_test_user_role_fixed_prompt_changed.jsonl -i 0 -be api -ap deepseek --resume_from_file "$resume_file" > "$log_file" 2>&1 &
done

# --- Wait for all jobs to complete ---
echo "--------------------------------------------------"
echo "✅ All 40 jobs have been launched in the background."
echo "Waiting for all processes to complete... (This may take a while)"

# The 'wait' command pauses the script until all background jobs started in this script have finished.
wait

echo "--------------------------------------------------"
echo "🎉 All jobs have completed!"
echo "Check the log directory for output: $LOG_DIR"