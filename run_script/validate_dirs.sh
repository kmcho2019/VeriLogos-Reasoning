#!/bin/bash

#
# This script finds all-numeric directories within a base path,
# then checks their subdirectories for *.score files.
# If a subdirectory contains a .score file with "1.0", it is ignored.
# Otherwise, the subdirectory path is added to validation_failed_list.txt.
#

# --- Configuration ---
# Use the first command-line argument as the base path, or default to the current directory "."
BASE_PATH=${1:-.}
OUTPUT_FILE="validation_failed_list.txt"

# --- Pre-run Checks ---
# Ensure the base path exists
if [ ! -d "$BASE_PATH" ]; then
    echo "Error: Base path '$BASE_PATH' does not exist."
    exit 1
fi

# --- Main Logic ---

# Clear or create the output file for a fresh run
> "$OUTPUT_FILE"

echo "🔎 Starting search in: $(realpath "$BASE_PATH")"
echo "📝 Failed directories will be saved to: $OUTPUT_FILE"
echo "---"

# Find all directories directly under the BASE_PATH
# We pipe to a 'while read' loop to process each directory individually.
find "$BASE_PATH" -maxdepth 1 -type d | while read -r numeric_dir; do
    # Extract the directory's name from its path
    dir_name=$(basename "$numeric_dir")

    # Check if the directory name consists ONLY of numbers (e.g., "0", "99", "123")
    if [[ "$dir_name" =~ ^[0-9]+$ ]]; then
        echo "Processing numeric directory: '$dir_name'"

        # Now, find all subdirectories within this numeric directory
        find "$numeric_dir" -maxdepth 1 -type d | while read -r sub_dir; do
            # The find command also lists the parent directory itself, so we skip it.
            if [ "$sub_dir" == "$numeric_dir" ]; then
                continue
            fi

            # We assume the directory fails validation by default.
            # We will set this to true only if we find a *.score file containing "1.0".
            validation_passed=false

            # Use grep -q to quietly and efficiently search for the exact line "1.0".
            # The -F option ensures we search for a fixed string, not a pattern.
            # The -x option ensures the entire line must match "1.0".
            # The search is performed on all *.score files within the subdirectory.
            # We add a check `[ -n "$(find "$sub_dir" -maxdepth 1 -name '*.score' -print -quit)" ]`
            # to ensure we only run grep if .score files actually exist.
            if [ -n "$(find "$sub_dir" -maxdepth 1 -name '*.score' -print -quit)" ] && grep -qFx "1.0" "$sub_dir"/*.score; then
                validation_passed=true
                echo "  ✅ Subdirectory '$sub_dir' PASSED validation."
            fi

            # If after checking all score files, validation has not passed, add it to our list.
            if [ "$validation_passed" = false ]; then
                echo "  ❌ Subdirectory '$sub_dir' FAILED validation. Adding to list."
                # Append the full path of the failed subdirectory to the output file.
                echo "$dir_name" >> "$OUTPUT_FILE"
            fi
        done
    fi
done

echo "---"
echo "✨ Script finished."
