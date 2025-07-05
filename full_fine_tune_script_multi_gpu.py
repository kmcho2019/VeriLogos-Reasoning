import os
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
from datetime import datetime
import re

# --- Argument Parsing ---
def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune a causal language model.")
    parser.add_argument(
        "--input_model",
        type=str,
        default="/data/genai/models/Qwen3-14B",
        help="Path to the pretrained model from which to load the tokenizer and model."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/data/genai/kmcho/Reasoning_Model/Qwen3-14B-exp-20250605-v1-full",
        help="The base directory where the final model and logs will be saved."
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/data/genai/kmcho/Reasoning_Model/VeriLogos-Reasoning/data/jsonl/test_code_output_mask_test_user_role_fixed_prompt_changed_with_verified_synthetic_reasoning_traces_sft_data_20250529.jsonl",
        help="Path to the JSONL file containing the reasoning dataset for training."
    )
    return parser.parse_args()

# --- Load Model and Tokenizer ---
def load_model_and_tokenizer(model_name_or_path):
    """Loads the model and tokenizer from the specified path."""
    print(f"Loading model and tokenizer from {model_name_or_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        use_cache=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    print(f"Tokenizer pad token ID: {tokenizer.pad_token_id}, EOS token ID: {tokenizer.eos_token_id}")
    return model, tokenizer

# --- Preprocess Dataset for SFT (Updated Logic) ---
def preprocess_sft_data(examples, tokenizer, max_length=8192):
    """
    Preprocesses the dataset for supervised fine-tuning (SFT) with custom truncation for reasoning traces.
    """
    all_input_ids = []
    all_labels = []
    all_attention_masks = []

    think_start_token = "<think>"
    think_end_token = "</think>"

    for conversation_turns in examples["messages"]:
        if not isinstance(conversation_turns, list) or not conversation_turns:
            continue

        # Find the last assistant message
        assistant_idx = -1
        for i in range(len(conversation_turns) - 1, -1, -1):
            if conversation_turns[i]['role'] == 'assistant':
                assistant_idx = i
                break
        
        if assistant_idx == -1 or assistant_idx == 0:
            continue

        prompt_messages = conversation_turns[:assistant_idx]
        assistant_message = conversation_turns[assistant_idx]
        assistant_content = assistant_message['content']

        # Custom truncation logic
        prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
        prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)

        # Split assistant content into reasoning and non-reasoning parts
        think_match = re.search(r"<think>(.*?)</think>", assistant_content, re.DOTALL)
        if think_match:
            reasoning_text = think_match.group(1)
            other_text = assistant_content.replace(think_match.group(0), "")
            
            reasoning_tokens = tokenizer.encode(reasoning_text, add_special_tokens=False)
            other_tokens = tokenizer.encode(other_text, add_special_tokens=False)
            
            if len(prompt_tokens) + len(reasoning_tokens) + len(other_tokens) > max_length:
                # Truncate the reasoning part from the end
                allowed_reasoning_len = max_length - len(prompt_tokens) - len(other_tokens)
                if allowed_reasoning_len > 0:
                    truncated_reasoning_tokens = reasoning_tokens[:allowed_reasoning_len]
                    truncated_reasoning_text = tokenizer.decode(truncated_reasoning_tokens)
                    assistant_content = f"{think_start_token}{truncated_reasoning_text}{think_end_token}{other_text}"
                else:
                    assistant_content = other_text # No space for reasoning
        
        # Ensure content ends with EOS token
        if not assistant_content.strip().endswith(tokenizer.eos_token):
            assistant_content += tokenizer.eos_token

        full_chat = prompt_messages + [{"role": "assistant", "content": assistant_content}]

        try:
            full_text = tokenizer.apply_chat_template(
                full_chat, tokenize=False, add_generation_prompt=False
            )
            model_inputs = tokenizer(
                full_text,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors=None
            )
        except Exception as e:
            print(f"Error during tokenization: {e}")
            continue

        current_input_ids = model_inputs['input_ids']
        current_attention_mask = model_inputs['attention_mask']
        current_labels = list(current_input_ids)

        prompt_len = len(prompt_tokens)
        
        # Mask prompt tokens and padding tokens
        for i in range(len(current_labels)):
            if i < prompt_len or current_input_ids[i] == tokenizer.pad_token_id:
                current_labels[i] = -100

        all_input_ids.append(current_input_ids)
        all_labels.append(current_labels)
        all_attention_masks.append(current_attention_mask)

    return {
        "input_ids": all_input_ids,
        "labels": all_labels,
        "attention_mask": all_attention_masks,
    }

# --- Fine-tuning ---
def fine_tune_model(model, tokenizer, dataset, output_dir):
    """
    Configures and runs the fine-tuning process.
    """
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        num_train_epochs=3,
        save_steps=500,
        save_total_limit=2,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=10,
        load_best_model_at_end=True,
        save_strategy="epoch",
        do_train=True,
        do_eval=True,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        gradient_accumulation_steps=8,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False,
        optim="paged_adamw_8bit", # Use 8-bit AdamW optimizer for memory efficiency
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=dataset, # Ideally use a separate validation set
        tokenizer=tokenizer,
    )

    print("Starting full fine-tuning...")
    trainer.train()

# --- Save Model ---
def save_model(model, tokenizer, output_dir):
    """
    Saves the full fine-tuned model and tokenizer.
    """
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Full model and tokenizer saved to {output_dir}")

# --- Main Execution ---
def main(args):
    """
    Main function to run the script.
    """
    torch.set_float32_matmul_precision("high")

    safe_model_name = args.input_model.replace('/', '_')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_output_dir = os.path.join(args.output_dir, f"{safe_model_name}_{timestamp}")
    os.makedirs(final_output_dir, exist_ok=True)
    print(f"Final output will be saved to: {final_output_dir}")

    model, tokenizer = load_model_and_tokenizer(args.input_model)

    print(f"Loading dataset from {args.dataset_path}...")
    try:
        raw_dataset = load_dataset('json', data_files=args.dataset_path, split="train")
    except Exception as e:
        print(f"Failed to load dataset. Error: {e}")
        return

    print("Preprocessing dataset...")
    tokenized_dataset = raw_dataset.map(
        preprocess_sft_data,
        fn_kwargs={"tokenizer": tokenizer, "max_length": 8192},
        batched=True,
        remove_columns=raw_dataset.column_names,
        num_proc=max(1, os.cpu_count() // 2)
    )

    if not tokenized_dataset:
        print("Dataset is empty after preprocessing. Check your data and preprocessing logic.")
        return

    fine_tune_model(model, tokenizer, tokenized_dataset, final_output_dir)
    save_model(model, tokenizer, final_output_dir)
    print("Full model fine-tuning complete.")

if __name__ == "__main__":
    args = parse_arguments()
    main(args)