import os
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType
from datasets import Dataset, DatasetDict, load_dataset
from datetime import datetime


# --- Argument Parsing ---
def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune a causal language model with LoRA.")
    parser.add_argument(
        "--input_model",
        type=str,
        default="/data/genai/models/Qwen3-14B",
        help="Path to the pretrained model from which to load the tokenizer and model."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/data/genai/kmcho/Reasoning_Model/Qwen3-14B-exp-20250605-v1",
        help="The base directory where the final LoRA weights and logs will be saved."
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/data/genai/kmcho/Reasoning_Model/VeriLogos-Reasoning/data/jsonl/test_code_output_mask_test_user_role_fixed_prompt_changed_with_verified_synthetic_reasoning_traces_sft_data_20250529.jsonl",
        help="Path to the JSONL file containing the reasoning dataset for training."
    )
    return parser.parse_args()


# --- Configuration ---
# os.environ["CUDA_VISIBLE_DEVICES"] = "4"#"5,6,7" # Ensure this matches your available GPUs
MODEL_NAME_OR_PATH = "/data/genai/models/Qwen3-14B" #"/data/genai/models/DeepSeek-R1-Distill-Qwen-7B"
TOKENIZER_PATH = MODEL_NAME_OR_PATH
OUTPUT_BASE_DIR = "/data/genai/kmcho/Reasoning_Model/Qwen3-14B-exp-20250605-v1"#"./data/lora_weights" #"/data/genai/kmcho/Reasoning_Model/DeepSeek-R1-Distill-Qwen-7B-exp-20250526-v1"

safe_model_name = MODEL_NAME_OR_PATH.replace('/', '_').replace('-', '_')
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR, safe_model_name + "_" + timestamp)


# V1 reasoning dataset with 985 samples generated from R1 and then verified by synopsys formality compared to original code
v1_reasoning_dataset_jsonl_path = "/home/kmcho/2_Project/LX_Semicon_GenAI/Reasoning_Model_Exp/VeriLogos-Reasoning/data/jsonl/test_code_output_mask_test_user_role_fixed_prompt_changed_with_verified_synthetic_reasoning_traces_sft_data_20250525.jsonl"

# V2 reasoning dataset with 2156 samples generated from R1 and then verified by synopsys formality compared to original code
# V2 is a superset of V1, so it includes all V1 samples plus additional ones
v2_reasoning_dataset_jsonl_path = "/home/kmcho/2_Project/LX_Semicon_GenAI/Reasoning_Model_Exp/VeriLogos-Reasoning/data/jsonl/test_code_output_mask_test_user_role_fixed_prompt_changed_with_verified_synthetic_reasoning_traces_sft_data_20250529.jsonl"

v2_reasoning_dataset_jsonl_path = "/data/genai/kmcho/Reasoning_Model/VeriLogos-Reasoning/data/jsonl/test_code_output_mask_test_user_role_fixed_prompt_changed_with_verified_synthetic_reasoning_traces_sft_data_20250529.jsonl"

REASONING_DATASET_JSONL_PATH = v2_reasoning_dataset_jsonl_path#v1_reasoning_dataset_jsonl_path#"/data/genai/kmcho/Reasoning_Model/VeriLogos-Reasoning/data/jsonl/test_code_output_mask_test_user_role_fixed_prompt_changed_with_verified_synthetic_reasoning_traces_sft_data_20250525.jsonl" #"/data/genai/kmcho/Reasoning_Model/output.jsonl" # Your correctly formatted file

# --- LoRA configuration ---
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], # Verify these against your model's architecture.
                                         # Common for Llama/Mistral: "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

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

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, tokenizer

# --- Preprocess Dataset for SFT (Updated Logic) ---
def preprocess_sft_data(examples, tokenizer, max_length=8192):
    all_input_ids = []
    all_labels = []
    all_attention_masks = []

    for conversation_turns in examples["messages"]:
        if not conversation_turns or not isinstance(conversation_turns, list):
            print(f"Skipping: Invalid conversation_turns format: {conversation_turns}")
            continue

        # Identify prompt messages and the final assistant message.
        # The prompt includes all messages before the *final* assistant message.
        # The final assistant message is the target completion.
        assistant_idx = -1
        for i in range(len(conversation_turns) - 1, -1, -1): # Iterate backwards to find the last assistant message
            if conversation_turns[i]['role'] == 'assistant':
                assistant_idx = i
                break
        
        if assistant_idx == -1:
            print(f"Skipping: No assistant message found in {conversation_turns}")
            continue
        
        if assistant_idx == 0: # Assistant message is first, no preceding prompt
            print(f"Skipping: Assistant message is first, no prompt: {conversation_turns}")
            continue

        prompt_messages_list = conversation_turns[:assistant_idx] # All messages before the identified assistant message
        assistant_message_dict = conversation_turns[assistant_idx]
        assistant_content = assistant_message_dict['content']

        # Ensure assistant content ends with EOS token for proper generation termination
        if not assistant_content.strip().endswith(tokenizer.eos_token):
            assistant_content_final = assistant_content + tokenizer.eos_token
        else:
            assistant_content_final = assistant_content
        
        # These are the messages that will be fully tokenized for input_ids (prompt + completion)
        full_chat_for_model_input = prompt_messages_list + [{"role": "assistant", "content": assistant_content_final}]
        
        try:
            # Tokenize the full conversation for input_ids using the model's chat template
            formatted_text = tokenizer.apply_chat_template(
                full_chat_for_model_input,
                tokenize=False,
                add_generation_prompt=False # Important for training: includes the assistant's response
            )
            model_inputs = tokenizer(
                formatted_text,
                max_length=max_length,
                padding="max_length", # Pad to max_length for uniform tensor sizes
                truncation=True,
                return_tensors=None # .map expects lists
            )
        except Exception as e:
            print(f"Error during tokenization of full chat: {e}")
            print(f"Problematic chat for model input: {full_chat_for_model_input}")
            continue # Skip this problematic example

        current_input_ids = model_inputs['input_ids']
        current_attention_mask = model_inputs['attention_mask']
        current_labels = list(current_input_ids) # Initialize labels as a copy of input_ids

        # --- Masking the prompt tokens in the labels ---
        # We only want to compute loss on the assistant's actual response tokens.
        try:
            # Create the prompt part by applying the template to user/system turn(s) + assistant generation prompt
            prompt_masking_context_text = tokenizer.apply_chat_template(
                prompt_messages_list, # This includes system and user messages leading up to the assistant's turn
                tokenize=False,
                add_generation_prompt=True # Crucial: appends the template's cue for the assistant to start
            )
            
            # Tokenize just the prompt part to find its length in tokens.
            # add_special_tokens=False is often used here if the chat template handles special tokens (BOS/EOS) internally.
            # This needs to be precise for correct masking.
            tokenized_prompt_mask_part = tokenizer(prompt_masking_context_text, add_special_tokens=False)
            prompt_tokens_length = len(tokenized_prompt_mask_part.input_ids)
        except Exception as e:
            print(f"Error during tokenization of prompt for masking: {e}")
            print(f"Problematic messages for prompt masking: {prompt_messages_list}")
            continue # Skip this problematic example

        # Mask the prompt tokens by setting their labels to -100
        for i in range(prompt_tokens_length):
            if i < len(current_labels): # Ensure index is within bounds
                current_labels[i] = -100
        
        # Also ensure padding tokens are masked in labels
        # This must be done *after* prompt masking if prompt can be shorter than padding.
        for i in range(len(current_labels)):
            if current_input_ids[i] == tokenizer.pad_token_id:
                current_labels[i] = -100
            
        all_input_ids.append(current_input_ids)
        all_labels.append(current_labels)
        all_attention_masks.append(current_attention_mask)

    if not all_input_ids:
        print("Warning: No valid examples were processed in this batch.")
        return {"input_ids": [], "labels": [], "attention_mask": []}
        
    return {
        "input_ids": all_input_ids,
        "labels": all_labels,
        "attention_mask": all_attention_masks,
    }

# --- Fine-tuning ---
def fine_tune_model(model, tokenizer, dataset, output_dir):
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        per_device_train_batch_size=1, # Adjust based on GPU memory; 1 is low, try 2 or 4 if memory allows
        per_device_eval_batch_size=1,
        num_train_epochs=3,
        save_steps=500, # Consider aligning with evaluation frequency or making it larger
        save_total_limit=2, # Saves the best and the last checkpoint
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=10, # Log more frequently for smaller datasets/debugging
        load_best_model_at_end=True,
        save_strategy="epoch",
        do_train=True,
        do_eval=True,
        bf16=(torch.cuda.get_device_capability()[0] >= 8) and (torch.cuda.is_bf16_supported()), # Enable bfloat16 if supported
        fp16=not (torch.cuda.get_device_capability()[0] >= 8) and (torch.cuda.is_bf16_supported()) and torch.cuda.is_available(), # Fallback to fp16 if bfloat16 not supported
        gradient_accumulation_steps=8, # Effective batch size = 1 (batch_size) * num_gpus * 8
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        #report_to="tensorboard",
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False, # Set to False if not using DDP or if you are sure about parameters
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=dataset, # Ideally, use a separate validation set
        tokenizer=tokenizer, # Useful for data collator if not padding fully in map
        # data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False), # Not strictly needed if map pads to max_length
    )

    print("Starting training...")
    trainer.train()

# --- Save LoRA Weights ---
def save_lora_weights(model, output_dir):
    model.save_pretrained(output_dir) # Saves LoRA adapter weights
    # tokenizer.save_pretrained(OUTPUT_DIR) # Also save tokenizer for completeness
    print(f"LoRA weights and tokenizer saved to {output_dir}")

# --- Main Execution ---
def main(args):
    # Enable TF32 tensorcores
    torch.set_float32_matmul_precision("high")


    # --- Dynamic Output Directory Configuration ---
    safe_model_name = args.input_model.replace('/', '_').replace('-', '_')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_output_dir = os.path.join(args.output_dir, f"{safe_model_name}_{timestamp}")
    os.makedirs(final_output_dir, exist_ok=True)
    print(f"Final output will be saved to: {final_output_dir}")

    model, tokenizer = load_model_and_tokenizer(args.input_model)

    print(f"Loading dataset from {args.dataset_path}...")
    try:
        # Ensure your output.jsonl is fixed: each line is {"messages": [your_array_here]}
        raw_dataset = load_dataset('json', data_files=args.dataset_path, split="train")
    except Exception as e:
        print(f"Failed to load dataset. Ensure '{args.dataset_path}' has one JSON object per line, with a 'messages' key.")
        print(f"Error: {e}")
        return

    if len(raw_dataset) == 0:
        print("Raw dataset is empty. Please check your JSONL file.")
        return
    print(f"Raw dataset loaded. Number of examples: {len(raw_dataset)}. Example: {raw_dataset[0]}")

    print("Preprocessing dataset...")
    tokenized_dataset = raw_dataset.map(
        preprocess_sft_data,
        fn_kwargs={"tokenizer": tokenizer, "max_length": 8192}, # Max_length from your original script
        batched=True,
        remove_columns=raw_dataset.column_names,
        num_proc=max(1, os.cpu_count() // 2) # Use multiple processes for mapping if beneficial
    )
    
    if len(tokenized_dataset) == 0:
        print("Dataset is empty after preprocessing. Please check your data and preprocessing logic (e.g., console for skipping messages).")
        return
    print(f"Tokenized dataset ready. Number of examples: {len(tokenized_dataset)}. Example input_ids: {tokenized_dataset[0]['input_ids'][:10]}...") # Print first 10 tokens

    fine_tune_model(model, tokenizer, tokenized_dataset, final_output_dir)
    save_lora_weights(model, final_output_dir)
    tokenizer.save_pretrained(final_output_dir) # Save tokenizer with the fine-tuned model
    print("Fine-tuning complete and LoRA weights saved.")

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
