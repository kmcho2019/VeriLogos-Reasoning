import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def merge_lora(base_model_path, lora_path, merged_model_path):
    """
    Merges a LoRA adapter with a base model and saves the resulting model.

    Args:
        base_model_path (str): The path to the base model.
        lora_path (str): The path to the LoRA adapter.
        merged_model_path (str): The path to save the merged model.
    """
    print(f"Loading base model from {base_model_path}...")
    # Load the base model with a specific dtype for memory efficiency
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype="auto",  # Use bfloat16 for modern GPUs, or float16
        device_map="auto"           # Automatically place layers on available devices (GPU/CPU)
    )

    print(f"Loading LoRA adapters from {lora_path}...")
    # Load the PeftModel by applying the LoRA adapter to the base model
    peft_model = PeftModel.from_pretrained(base_model, lora_path)

    print("Merging the LoRA weights...")
    # Merge the adapter weights into the base model. This returns the base model with merged weights.
    merged_model = peft_model.merge_and_unload()

    print(f"Saving merged model to {merged_model_path}...")
    # Save the merged model
    merged_model.save_pretrained(merged_model_path)

    # It's crucial to also save the tokenizer for the model to be easily usable later
    print(f"Saving tokenizer to {merged_model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.save_pretrained(merged_model_path)

    print(f"Successfully merged and saved model to {merged_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge a LoRA adapter into a base model and save it.")
    parser.add_argument('--base_model', type=str, required=True, help="Path to the base model (e.g., Qwen/Qwen3-8B)")
    parser.add_argument('--lora_model', type=str, required=True, help="Path to the LoRA adapter directory")
    parser.add_argument('--output_model', type=str, required=True, help="Path to save the merged model")

    args = parser.parse_args()

    merge_lora(args.base_model, args.lora_model, args.output_model)
