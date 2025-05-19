#!/bin/bash

# Configure these variables
ACCELERATE_CONFIG_FILE="{path/to/your/accelerate_config.yaml}" # If using accelerate for anything, else remove
PYTHON_EXEC="python" # Or "accelerate launch --config_file $ACCELERATE_CONFIG_FILE" if main script uses it

REFERENCE_CODE_DIR="./data/reference_verilog_snippets"  # Create this directory and populate with .v files
SYNTHETIC_TRACES_OUTPUT_FILE="./exp/synthetic_traces/my_reasoning_traces.jsonl"

R1_API_PROVIDER="deepseek_api"  # 'together', 'fireworks', or 'deepseek_api'
R1_API_KEY="YOUR_ACTUAL_API_KEY" # Replace with your key
# For DeepSeek API, input_model could be 'deepseek-chat' or 'deepseek-coder' (check their docs)
# For Together/Fireworks, use the model identifier they provide for the R1 distilled model
R1_MODEL_FOR_API="deepseek-chat" # Example: "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" if provider supports it directly

# Ensure reference directory exists
mkdir -p $REFERENCE_CODE_DIR
# Add a dummy file if empty for testing
# echo "module dummy(input a, output b); assign b=a; endmodule" > $REFERENCE_CODE_DIR/dummy.v

echo "Generating synthetic reasoning traces..."

$PYTHON_EXEC main.py GEN_SYNTHETIC_TRACES \
    --reference_code_dir "$REFERENCE_CODE_DIR" \
    --synthetic_traces_file "$SYNTHETIC_TRACES_OUTPUT_FILE" \
    --r1_api_provider "$R1_API_PROVIDER" \
    --r1_api_key "$R1_API_KEY" \
    --input_model "$R1_MODEL_FOR_API" # This is used as r1_model_name in trace_generator

echo "Synthetic trace generation process finished."
echo "Traces saved to $SYNTHETIC_TRACES_OUTPUT_FILE"