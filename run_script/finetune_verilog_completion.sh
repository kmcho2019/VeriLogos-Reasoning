#!/bin/bash

ACCELERATE_CONFIG_FILE="{path/to/your/accelerate_config.yaml}"
# SFT typically uses accelerate
ACCELERATE_LAUNCH="accelerate launch --config_file $ACCELERATE_CONFIG_FILE"

# Model to be fine-tuned (e.g., the R1 distilled model)
BASE_MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" 

# Output name for the fine-tuned model
OUTPUT_MODEL_NAME="VeriLogos-R1-Distill-Qwen-7B-VerilogCompletion" 

# Path to the generated and filtered synthetic traces
SYNTHETIC_TRACES_FILE="./exp/synthetic_traces/my_reasoning_traces.jsonl" # Should match output of gen_synthetic_traces.sh

if [ ! -f "$SYNTHETIC_TRACES_FILE" ]; then
    echo "Error: Synthetic traces file not found at $SYNTHETIC_TRACES_FILE"
    echo "Please run gen_synthetic_traces.sh first."
    exit 1
fi

echo "Starting fine-tuning for Verilog completion..."

$ACCELERATE_LAUNCH main.py FINETUNE_VERILOG_COMPLETION \
    -im "$BASE_MODEL_NAME" \
    -om "$OUTPUT_MODEL_NAME" \
    --synthetic_traces_file "$SYNTHETIC_TRACES_FILE"
    # Add other SFT parameters if needed (e.g., learning rate, epochs) by modifying main.py or sft.py

echo "Fine-tuning process finished."
echo "Fine-tuned model artifacts should be in ./cache/$OUTPUT_MODEL_NAME"