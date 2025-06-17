import os
import torch
import transformers
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.pipelines.pt_utils import KeyDataset
from tqdm import tqdm

from openai import OpenAI

# Imports for lm-deluge batch inference
from lm_deluge import LLMClient, SamplingParams, Conversation as DelugeConversation, Message as DelugeMessage

# Imports for PEFT (LoRA) support
from peft import PeftModel, PeftConfig

# Import time for retry delay
import time

_REGISTRY = {
    "openai":   ("https://api.openai.com/v1",      "OPENAI_API_KEY"),
    "groq":     ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "together": ("https://api.together.xyz/v1",    "TOGETHER_API_KEY"),
    "fireworks":("https://api.fireworks.ai/v1",    "FIREWORKS_API_KEY"),
    "deepseek":("https://api.deepseek.com", "DEEPSEEK_API_KEY"),
    "openrouter":("https://openrouter.ai/v1", "OPENROUTER_API_KEY"),
    "gemini":("https://generativelanguage.googleapis.com/v1beta/openai/", "GEMINI_API_KEY"),
}

def get_client(provider: str = "openai", **kw) -> OpenAI:
    """Return an OpenAI-compatible client for the chosen provider."""
    base_url, env_key = _REGISTRY[provider]
    return OpenAI(
        base_url=base_url.format(**kw),                # Azure는 {deployment} 치환
        api_key=kw.get("api_key") or os.getenv(env_key),
        organization=kw.get("org"),
        timeout=kw.get("timeout", 600),
    )


def gen_hdl(
    model,                  # Hugging-Face repo OR OpenAI model id
    data_jsonl,
    idx_code,
    cache_dir,
    data_dir,
    exp_dir,
    num_process=None,
    idx_process=None,
    backend="hf",           # "hf" (default) | "api"
    provider="openai",      # "openai" | "together" | "fireworks" | "deepseek_api"
    api_key=None,           # set here or via $OPENAI_API_KEY
    resume_generation=False,# checks save directory for existing files and resumes generation
    batch_inference=False,  # Use lm-deluge for batch inference (API backend)
    hf_batch_size=1,        # Batch size for Hugging Face local inference
    force_thinking=False,   # Force thinking for Hugging Face local inference (specific models, eg. DeepSeek-R1-Distill)
    temperature=1.0,        # Temperature for model generation
    lora_weights=None,      # Path to LoRA adapter weights directory for Hugging Face models
    create_resume_file=None,# Path to save a file with names of unfinished modules
    resume_from_file=None,  # Path to a file with module names to process   
    **provider_kw,
):
    """
    Generate Verilog HDL either with a local Hugging-Face model
    or with an OpenAI hosted model (ChatCompletion API).
    Supports batching for Hugging Face backend and lm-deluge for API backend.
    """
    print(f'[GEN_HDL]: Generating Verilog HDL code with {model}, lora_weights={lora_weights}, temperature={temperature}, backend={backend}, batch_inference={batch_inference}, hf_batch_size={hf_batch_size}, force_thinking={force_thinking}')

    base_model_identifier = model # Use the original model arg as the base identifier

    # Determine the model identifier for saving paths
    lora_adapter_name_part = ""
    if backend == "hf" and lora_weights:
        # Ensure the LoRA weights path is valid and points to the adapter directory
        lora_adapter_path_for_name = lora_weights
        if os.path.isfile(lora_weights): # If path is to a file, assume parent dir
             lora_adapter_path_for_name = os.path.dirname(lora_weights)
        
        # Check if the path is a directory before getting its basename
        if os.path.isdir(lora_adapter_path_for_name):
            # Use basename of the normalized path for a cleaner name
            lora_adapter_name_part = f"+{os.path.basename(os.path.normpath(lora_adapter_path_for_name))}"
        else: # Fallback if path is not a dir (e.g. user provides a non-existent path)
            lora_adapter_name_part = "+lora_unknown" 
            
    model_identifier_for_paths = f"{base_model_identifier}{lora_adapter_name_part}"

    print(f'[GEN_HDL]: Base model identifier: {base_model_identifier}')
    if lora_weights and backend == "hf":
        print(f'[GEN_HDL]: LoRA adapter specified: {lora_weights}')
        print(f'[GEN_HDL]: Model identifier for output paths: {model_identifier_for_paths}')
    else:
        print(f'[GEN_HDL]: Model identifier for output paths: {model_identifier_for_paths}')

    # ─────────────────────────────────────────  Dataset  ────────────────────────
    data_path = os.path.join(data_dir, "jsonl", data_jsonl)
    
    # Load the full dataset definition to get all possible module names and prompts
    full_ds = load_dataset("json", data_files=data_path, split="train")
    all_module_names = [str(item.get("name", str(item.get("index")))) for item in full_ds]

    # Mode to discover unfinished modules and save to a file, then exit
    if create_resume_file:
        print(f"[GEN_HDL]: Discovery mode active. Finding unfinished modules to save to '{create_resume_file}'.")
        unfinished_modules = []
        for module_name in tqdm(all_module_names, desc="Discovering unfinished modules"):
            module_path = f"{exp_dir}/{model_identifier_for_paths}/{module_name}/gen_{module_name}_{idx_code}.v"
            if not (os.path.exists(module_path) and os.path.getsize(module_path) > 0):
                unfinished_modules.append(module_name)
        
        with open(create_resume_file, 'w') as f:
            for name in unfinished_modules:
                f.write(f"{name}\n")
        
        print(f"[GEN_HDL]: Found {len(unfinished_modules)} unfinished modules. Their names have been saved to '{create_resume_file}'. Exiting.")
        return

    # These lists will hold the items that actually need to be processed
    modules_to_process_names = []
    # For HF, we will filter the Dataset object directly.
    # For API (sequential), we need the list of message dicts.
    prompts_for_sequential_api = []
    # For API (lm-deluge), we need (system_prompt, user_assistant_messages) tuples.
    prompts_tuples_for_deluge = []
    # This will hold the final dataset slice for the HF backend
    ds_for_hf = None
    

    # Determine which modules to process based on the resume mode
    # Mode to process a specific list of modules from a file
    if resume_from_file:
        print(f"[GEN_HDL]: Resuming generation from module list: '{resume_from_file}'")
        if not os.path.exists(resume_from_file):
            print(f"[GEN_HDL]: ERROR - Resume file not found: {resume_from_file}. Exiting.")
            return

        with open(resume_from_file, 'r') as f:
            # Read names from file, stripping whitespace/newlines
            module_names_from_file = [line.strip() for line in f if line.strip()]

        print(f"[GEN_HDL]: Found {len(module_names_from_file)} modules to process in the file.")
        
        # Create a lookup dictionary for fast access to prompts
        prompt_lookup = {str(item.get("name", str(item.get("index")))): item["messages"] for item in full_ds}
        
        # Prepare data structures for the modules listed in the file
        filtered_data_for_hf = {'name': [], 'messages': []}
        for name in module_names_from_file:
            if name in prompt_lookup:
                modules_to_process_names.append(name)
                messages = prompt_lookup[name]
                prompts_for_sequential_api.append(messages)

                # For HF
                filtered_data_for_hf['name'].append(name)
                filtered_data_for_hf['messages'].append(messages)

                # For Deluge
                system_prompt = next((msg['content'] for msg in messages if msg['role'] == 'system'), "You are a helpful Verilog code generator.")
                user_assistant_msgs = [msg for msg in messages if msg['role'] in ['user', 'assistant']]
                prompts_tuples_for_deluge.append((system_prompt, user_assistant_msgs))
            else:
                print(f"[GEN_HDL]: Warning: Module '{name}' from file not found in dataset '{data_jsonl}'. Skipping.")
        
        ds_for_hf = Dataset.from_dict(filtered_data_for_hf)

    # Original resume logic: check output directory for all modules in the dataset
    elif resume_generation:
        print(f"[GEN_HDL]: Checking for existing files to resume generation. Total modules in dataset: {len(all_module_names)}")
        indices_to_keep_for_hf = []
        for i, module_name in enumerate(all_module_names):
            module_path = f"{exp_dir}/{model_identifier_for_paths}/{module_name}/gen_{module_name}_{idx_code}.v"
            if os.path.exists(module_path) and os.path.getsize(module_path) > 0:
                continue
            
            modules_to_process_names.append(module_name)
            indices_to_keep_for_hf.append(i)
            messages = full_ds[i]["messages"]
            prompts_for_sequential_api.append(messages)
            
            system_prompt = next((msg['content'] for msg in messages if msg['role'] == 'system'), "You are a helpful Verilog code generator.")
            user_assistant_msgs = [msg for msg in messages if msg['role'] in ['user', 'assistant']]
            prompts_tuples_for_deluge.append((system_prompt, user_assistant_msgs))
        
        ds_for_hf = full_ds.select(indices_to_keep_for_hf)

    # Default logic: process all modules from the dataset
    else:
        print(f"[GEN_HDL]: Processing all {len(all_module_names)} modules in the dataset.")
        modules_to_process_names = all_module_names
        ds_for_hf = full_ds
        for item in ds_for_hf:
            messages = item['messages']
            prompts_for_sequential_api.append(messages)
            system_prompt = next((msg['content'] for msg in messages if msg['role'] == 'system'), "You are a helpful Verilog code generator.")
            user_assistant_msgs = [msg for msg in messages if msg['role'] in ['user', 'assistant']]
            prompts_tuples_for_deluge.append((system_prompt, user_assistant_msgs))
    
    if not modules_to_process_names:
        print("[GEN_HDL]: No modules to process based on the current mode. All work may be complete. Exiting.")
        return
    
    # Slice the identified work for the current process if multi-processing is enabled
    if num_process is not None and idx_process is not None:
        # This manual slicing is now more complex because data isn't always in a single Dataset object.
        # We will slice the lists directly, which works for all modes.
        print(f"[GEN_HDL]: Slicing work for process {idx_process}/{num_process}.")
        total_size = len(modules_to_process_names)
        chunk_size = (total_size + num_process - 1) // num_process # Ceiling division
        start_idx = idx_process * chunk_size
        end_idx = min((idx_process + 1) * chunk_size, total_size)

        modules_to_process_names = modules_to_process_names[start_idx:end_idx]
        prompts_for_sequential_api = prompts_for_sequential_api[start_idx:end_idx]
        prompts_tuples_for_deluge = prompts_tuples_for_deluge[start_idx:end_idx]
        if ds_for_hf: # ds_for_hf is already filtered, so we slice it further
            ds_for_hf = ds_for_hf.select(range(start_idx, end_idx))


    if not modules_to_process_names:
        print(f"[GEN_HDL]: No modules to process for this process slice (idx={idx_process}). Exiting.")
        return

    print(f"[GEN_HDL]: This process will generate code for {len(modules_to_process_names)} modules.")

    # ───────────────────────────────────────  Back-end - HF  ───────────────────
    if backend == "hf":
        if not ds_for_hf:
            print("[GEN_HDL - HF]: No data to process for Hugging Face backend after filtering. Exiting.")
            return
        print(f"[GEN_HDL - HF]: Processing {len(modules_to_process_names)} modules with Hugging Face backend.")
        print(f"[GEN_HDL - HF]: Hugging Face Batch Size: {hf_batch_size}")
        if not ds_for_hf: # Check if the dataset for HF is empty after filtering
            print("[GEN_HDL - HF]: No data to process for Hugging Face backend after filtering. Exiting.")
            return

        # Use base_model_identifier for loading weights and tokenizer
        pretrained_path = (
            f"{cache_dir}/{base_model_identifier}" if "VeriLogos" in base_model_identifier else base_model_identifier
        )
        tok = AutoTokenizer.from_pretrained(
            pretrained_path, padding_side="left",
            trust_remote_code=True, cache_dir=cache_dir
        )
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        print(f"[GEN_HDL - HF]: Loading base model from {pretrained_path}...")
        base_model_hf = AutoModelForCausalLM.from_pretrained(
            pretrained_path, torch_dtype="auto",
            device_map="auto", cache_dir=cache_dir
        )
        net = base_model_hf # Start with the base model
        
        if lora_weights:
            print(f"[GEN_HDL - HF]: Loading LoRA weights from {lora_weights}...")
            # Load the LoRA adapter weights
            lora_config = PeftConfig.from_pretrained(lora_weights)
            net = PeftModel.from_pretrained(base_model_hf, lora_weights, torch_dtype="auto")
            net.eval()

        # Apply chat template to the (potentially filtered) ds_for_hf
        ds_for_hf_formatted = ds_for_hf.map(
            lambda x: {"formatted_messages": tok.apply_chat_template(
                x["messages"], tokenize=False, add_generation_prompt=True
            )}
        )

        pipe = transformers.pipeline(
            task="text-generation",
            model=net,
            tokenizer=tok,
            framework="pt",
            batch_size=hf_batch_size,  # <<< MODIFIED: Using hf_batch_size parameter
            num_workers=torch.cuda.device_count() * 4, # Adjusted num_workers, tune as needed
            device_map="auto",
            torch_dtype="auto"
        )
        gen_args = dict(
            max_new_tokens=4096, do_sample=True,
            temperature=temperature, top_k=50, top_p=1.0,
            return_full_text=False,
            eos_token_id=tok.eos_token_id, # Explicitly set eos_token_id
            pad_token_id=tok.pad_token_id  # Explicitly set pad_token_id
        )

        # If the model is a reasoning model, we might want to force it to start with "<think>"
        # This is particularly useful for models like DeepSeek-R1-Distill series.
        # This is a specific feature for models that support reasoning and thinking.
        # If force_thinking is True, we will ensure that the generation starts with "<think>".
        if force_thinking:
            think_token_str = "<think>"
            try:
                think_token_ids = tok.encode(think_token_str, add_special_tokens=False)
                if think_token_ids:
                    # `forced_decoder_ids` expects a list of tuples: (index_in_generation, token_id)
                    # Index 0 is the first token generated, 1 is the second, and so on.
                    forced_ids_list = []
                    for i, token_id in enumerate(think_token_ids):
                        forced_ids_list.append([i, token_id])
                    
                    net.config.forced_decoder_ids = forced_ids_list
                    #gen_args["forced_decoder_ids"] = forced_ids_list
                    print(f"[GEN_HDL - HF]: Forcing generation to start with '{think_token_str}' (token_ids: {think_token_ids}).")
                    # Optional: Adjust max_new_tokens if the forced prefix is very long,
                    # though for "<think>" it's likely not an issue.
                    # gen_args["max_new_tokens"] = gen_args.get("max_new_tokens", 4096) + len(think_token_ids)
                else:
                    print(f"[GEN_HDL - HF]: Warning: Tokenizer encoded '{think_token_str}' to an empty list. Cannot force prefix.")
            except Exception as e:
                print(f"[GEN_HDL - HF]: Warning: Error encoding '{think_token_str}': {e}. Cannot force prefix.")

        print(f"[GEN_HDL - HF]: Starting generation for {len(ds_for_hf_formatted)} prompts with batch size {hf_batch_size}.")
        for i, out in tqdm(enumerate(pipe(KeyDataset(ds_for_hf_formatted, "formatted_messages"), **gen_args)), total=len(ds_for_hf_formatted)):
            module_name = modules_to_process_names[i] # Aligned with ds_for_hf_formatted
            code = out[0]["generated_text"]
            _dump(code, exp_dir, model_identifier_for_paths, module_name, idx_code)

    # ───────────────────────────────────  Back-end - OpenAI  ───────────────────
    elif backend == "api":
        if not modules_to_process_names: # Check if any modules to process for API
            print("[GEN_HDL - API]: No data to process for API backend after filtering. Exiting.")
            return
        
        # Use base_model_identifier for API model name
        current_api_model_id = base_model_identifier

        if batch_inference is True:
            deluge_client = LLMClient(model_names=[current_api_model_id], max_requests_per_minute=5_000, max_tokens_per_minute=1_000_000, max_concurrent_requests=1_000, sampling_params=SamplingParams(
                temperature=temperature, top_p=1.0, max_new_tokens=4096))
            
            deluge_conversations = []
            for system_prompt_content, user_assistant_messages in prompts_tuples_for_deluge:
                conversation = DelugeConversation.system(system_prompt_content)
                for msg in user_assistant_messages:
                    if msg["role"] == "user":
                        conversation.add(DelugeMessage.user(msg["content"]))
                    elif msg["role"] == "assistant":
                        conversation.add(DelugeMessage.assistant(msg["content"]))
                deluge_conversations.append(conversation)
            
            if not deluge_conversations:
                print("[GEN_HDL - API Deluge]: No prompts to process with lm-deluge. Exiting.")
                return

            # Initial list of work items, combining module name and conversation object
            work_items = list(zip(modules_to_process_names, deluge_conversations))
            retry_count = 0
            max_retries = 3
            retry_delay = 5
            DELUGE_BATCH_SIZE = 20
            deluge_batch_size = min(DELUGE_BATCH_SIZE, len(work_items))  # Ensure batch size is not larger than work items

            while work_items and retry_count < max_retries:
                if retry_count > 0:
                    print(f"\n[GEN_HDL]: Retrying {len(work_items)} failed modules. Attempt {retry_count + 1}/{max_retries}.")
                    print(f"[GEN_HDL]: Waiting for {retry_delay} seconds before next attempt...")
                    time.sleep(retry_delay)

                # Get data for the current attempt
                current_module_names = [item[0] for item in work_items]
                current_conversations = [item[1] for item in work_items]
                num_prompts_this_attempt = len(current_conversations)

                # Batch the current work
                batched_deluge_prompts = [current_conversations[i:i + deluge_batch_size] for i in range(0, num_prompts_this_attempt, deluge_batch_size)]
                batched_module_names_deluge = [current_module_names[i:i + deluge_batch_size] for i in range(0, num_prompts_this_attempt, deluge_batch_size)]

                failed_items_this_attempt = []

                print(f"[GEN_HDL]: Processing {num_prompts_this_attempt} prompts in {len(batched_deluge_prompts)} batches (size={deluge_batch_size}).")
                for idx, batch_of_convs in enumerate(batched_deluge_prompts):
                    print(f"[GEN_HDL]: Processing lm-deluge batch {idx + 1}/{len(batched_deluge_prompts)} with {len(batch_of_convs)} prompts.")
                    current_module_names_batch = batched_module_names_deluge[idx]
                    
                    try:
                        responses = deluge_client.process_prompts_sync(batch_of_convs, show_progress=True)
                        
                        for j, rsp in enumerate(responses):
                            module_name = current_module_names_batch[j]
                            code = rsp.completion
                            thinking = rsp.thinking
                            
                            if code is None:
                                print(f"[GEN_HDL]: Error generating code for module {module_name}. Response is None. Scheduling for retry.")
                                failed_items_this_attempt.append((module_name, batch_of_convs[j]))
                            else:
                                _dump(code, exp_dir, model_identifier_for_paths, module_name, idx_code)
                                if current_api_model_id in ("deepseek-reasoner", "deepseek-r1") and thinking:
                                    trace_path = f"{exp_dir}/{model_identifier_for_paths}/{module_name}/gen_{module_name}_{idx_code}_trace.txt"
                                    os.makedirs(os.path.dirname(trace_path), exist_ok=True)
                                    with open(trace_path, "w") as trace_file:
                                        trace_file.write(str(thinking))

                    except Exception as e:
                        print(f"[GEN_HDL]: An exception occurred during batch processing: {e}. Scheduling all modules in this batch for retry.")
                        for j, conv in enumerate(batch_of_convs):
                             failed_items_this_attempt.append((current_module_names_batch[j], conv))

                # Prepare for the next iteration by setting work_items to the list of failures
                work_items = failed_items_this_attempt
                retry_count += 1
                # Print the number of remaining work items
                print(f"[GEN_HDL]: {len(work_items)} modules failed in this attempt {retry_count}/{max_retries}. Retrying...")

            # After the loop, report any modules that permanently failed
            if work_items:
                print(f"\n[GEN_HDL]: CRITICAL: After {max_retries} attempts, the following {len(work_items)} modules still failed to generate:")
                for module_name, _ in work_items:
                    print(f"  - {module_name}")
            else:
                print("\n[GEN_HDL]: All modules processed successfully.")                                
        else: # Sequential API calls (no external batching library)
            client = get_client(provider=provider, **provider_kw)
            # Default system prompt, will be prepended if item["messages"] doesn't have one or to standardize
            generic_sys_prompt_content = "You are an AI programming assistant specialized in Verilog. Follow the user's requirements and content closely.\n"#"You are a helpful Verilog code generator."

            print(f"[GEN_HDL - API Sequential]: Starting generation for {len(modules_to_process_names)} prompts.")
            for i, module_name in tqdm(enumerate(modules_to_process_names), total=len(modules_to_process_names)):
                # prompts_for_sequential_api contains the full message list for each module to process
                current_item_messages = prompts_for_sequential_api[i]
                
                # Prepare messages for OpenAI API:
                # Option 1: Use messages as is if they are complete (incl. system).
                # Option 2: Prepend a generic system prompt, like the original code.
                # The original code prepended a sys_prompt. Let's follow that,
                # but ensure we don't duplicate system prompts if not desired.
                # For simplicity, we'll mimic the original structure.
                
                api_payload_messages = [{"role": "system", "content": generic_sys_prompt_content}]
                has_user_or_assistant = False
                for msg in current_item_messages:
                    # Filter out any system messages from the original item if we're adding our own generic one
                    # Or, if item["messages"] is trusted to be complete, use it directly.
                    # For this example, let's assume item["messages"] might or might not have a system prompt,
                    # and we are standardizing with 'generic_sys_prompt_content'.
                    if msg["role"] != "system":
                        api_payload_messages.append(msg)
                        if msg["role"] in ["user", "assistant"]:
                            has_user_or_assistant = True
                
                # Ensure there's at least one non-system message if the original prompt was only a system message.
                if not has_user_or_assistant and current_item_messages and current_item_messages[0]["role"] == "user":
                     # This case is unlikely if prompts are well-formed (system, user, assistant...)
                     # but if current_item_messages was just [ {'role':'user', 'content':'...'} ],
                     # the above loop would have added it.
                     pass # This logic might need refinement based on exact prompt structures.


                # Deepseek-reasoner specific handling (original logic)
                if current_api_model_id != "deepseek-reasoner":
                     # Add empty assistant message to prompt for response, if not already ending with one.
                     if not api_payload_messages or api_payload_messages[-1]["role"] != "assistant":
                        api_payload_messages.append({"role": "assistant", "content": ""})
                
                try:
                    rsp = client.chat.completions.create(
                        model=current_api_model_id,
                        messages=api_payload_messages,
                        max_tokens=4096*8, temperature=temperature, top_p=1.0,
                    )
                    code = rsp.choices[0].message.content
                    _dump(code, exp_dir, model_identifier_for_paths, module_name, idx_code)

                    if current_api_model_id == "deepseek-reasoner" and hasattr(rsp.choices[0].message, 'reasoning_content'):
                        trace = rsp.choices[0].message.reasoning_content
                        if trace: # Ensure trace is not None
                            trace_path = f"{exp_dir}/{model_identifier_for_paths}/{module_name}/gen_{module_name}_{idx_code}_trace.txt"
                            os.makedirs(os.path.dirname(trace_path), exist_ok=True)
                            with open(trace_path, "w") as trace_file:
                                trace_file.write(str(trace))
                except Exception as e:
                    print(f"[GEN_HDL - API Sequential]: Error processing module {module_name}: {e}")
                    print(f"Payload messages: {api_payload_messages}") # Log messages for debugging
                    # Optionally, save a placeholder or skip
                    _dump(f"Error generating code: {e}", exp_dir, model_identifier_for_paths, module_name, idx_code)


    else:
        raise ValueError("backend must be 'hf' or 'api'")

# util (unchanged) --------------------------------------------------------------
def _dump(code: str, exp_root: str, model: str, module: str, idx: int) -> None:
    path = f"{exp_root}/{model}/{module}/gen_{module}_{idx}.v"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fp:
        fp.write(code)

def slice_dataset(dataset, num_process, idx_process):
    total_size = len(dataset)
    if total_size == 0:
        return dataset # Return empty dataset if it's already empty
    chunk_size = total_size // num_process
    if chunk_size == 0 and total_size > 0 : # Handle cases where num_process > total_size
        chunk_size = 1 


    start_idx = idx_process * chunk_size
    if idx_process == num_process - 1:
        end_idx = total_size
    else:
        end_idx = (idx_process + 1) * chunk_size
    
    if start_idx >= total_size: # if start index is out of bounds, return empty dataset
        return dataset.select([])

    end_idx = min(end_idx, total_size) # Ensure end_idx doesn't exceed total_size

    return dataset.select(range(start_idx, end_idx))
