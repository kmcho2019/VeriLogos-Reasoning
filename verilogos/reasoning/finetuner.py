# verilogos/reasoning/finetuner.py
import os
from verilogos.trainer.sft import sft as perform_sft # Alias to avoid name clash if needed

def finetune_on_traces(input_model: str, output_model: str, traces_file: str, cache_dir: str, data_dir: str):
    """
    Fine-tunes a model (e.g., R1 distilled) on the synthetically generated and filtered traces
    for Verilog completion.
    """
    print(f"Starting fine-tuning for Verilog completion using SFT.")
    print(f"Input Model: {input_model}")
    print(f"Output Model: {output_model}")
    print(f"Traces File (data_jsonl for SFT): {traces_file}")

    # The SFT script expects the data in a specific JSONL format,
    # where each line has a "messages" key.
    # The generate_and_filter_traces function should already produce this.

    perform_sft(
        input_model=input_model,
        output_model=output_model,
        data_jsonl=traces_file, # This is the key: use the generated traces
        cache_dir=cache_dir,
        data_dir=data_dir # sft.py might use data_dir for structuring dataset path, ensure it's okay
    )
    print(f"Fine-tuning complete. Model saved to {os.path.join(cache_dir, output_model)}")

# if __name__ == '__main__':
#     # Example usage (called from main.py)
#     finetune_on_traces(
#         input_model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", # Or your base model
#         output_model="VeriLogos-R1-Distill-Qwen-7B-VerilogCompletion",
#         traces_file="./exp/synthetic_traces/filtered_traces.jsonl", # Output from previous step
#         cache_dir="./cache",
#         data_dir="./data"
#     )