import os
import torch
import transformers
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.pipelines.pt_utils import KeyDataset
from tqdm import tqdm

from openai import OpenAI

# Imports for lm-deluge batch inference
from lm_deluge import LLMClient, SamplingParams, Conversation as DelugeConversation, Message as DelugeMessage

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
    **provider_kw,
):
    """
    Generate Verilog HDL either with a local Hugging-Face model
    or with an OpenAI hosted model (ChatCompletion API).
    Supports batching for Hugging Face backend and lm-deluge for API backend.
    """
    print(f'[GEN_HDL]: Generating Verilog HDL code with {model}')
    # ─────────────────────────────────────────  Dataset  ────────────────────────
    data_path = os.path.join(data_dir, "jsonl", data_jsonl)
    # Load the initial dataset
    current_ds = load_dataset("json", data_files=data_path, split="train")

    if num_process is not None and idx_process is not None:
        current_ds = slice_dataset(current_ds, num_process, idx_process)

    # Extract all messages and module names from the current dataset slice
    all_messages_in_slice = [item["messages"] for item in current_ds]
    all_module_names_in_slice = [
        item["name"] if "name" in current_ds.column_names else str(item["index"])
        for item in current_ds
    ]

    # These lists will hold the items that actually need to be processed
    modules_to_process_names = []
    # For HF, we will filter the Dataset object directly.
    # For API (sequential), we need the list of message dicts.
    prompts_for_sequential_api = []
    # For API (lm-deluge), we need (system_prompt, user_assistant_messages) tuples.
    prompts_tuples_for_deluge = []
    indices_to_keep_for_hf = [] # To filter current_ds for HF backend

    if resume_generation:
        print(f"[GEN_HDL]: Checking for existing files to resume generation. Total modules in slice: {len(all_module_names_in_slice)}")
        for i, module_name in enumerate(all_module_names_in_slice):
            module_path = f"{exp_dir}/{model}/{module_name}/gen_{module_name}_{idx_code}.v"
            if os.path.exists(module_path) and os.path.getsize(module_path) > 0:
                # File exists and is not empty, skip this module
                continue
            
            # Module needs to be processed
            modules_to_process_names.append(module_name)
            indices_to_keep_for_hf.append(i)
            
            current_prompt_messages = all_messages_in_slice[i]
            prompts_for_sequential_api.append(current_prompt_messages)

            system_prompt_content = None
            user_assistant_messages = []
            for msg in current_prompt_messages:
                if msg["role"] == "system":
                    system_prompt_content = msg["content"]
                elif msg["role"] in ["user", "assistant"]:
                    user_assistant_messages.append(msg)
            
            if system_prompt_content is None:
                print(f"Warning: No system prompt found for module {module_name}. Using a default.")
                system_prompt_content = "You are a helpful Verilog code generator."
            prompts_tuples_for_deluge.append((system_prompt_content, user_assistant_messages))

        if not modules_to_process_names:
            print("[GEN_HDL]: No new modules to process. All already generated. Exiting.")
            return
        
        print(f"[GEN_HDL]: Resuming. Modules to process: {len(modules_to_process_names)}")
        # Filter the dataset for HF backend
        ds_for_hf = current_ds.select(indices_to_keep_for_hf)
    else:
        print(f"[GEN_HDL]: Processing all {len(all_module_names_in_slice)} modules in slice.")
        modules_to_process_names = all_module_names_in_slice
        ds_for_hf = current_ds # Use the whole (or process-sliced) dataset for HF
        prompts_for_sequential_api = all_messages_in_slice

        for current_prompt_messages in all_messages_in_slice:
            system_prompt_content = None
            user_assistant_messages = []
            for msg in current_prompt_messages:
                if msg["role"] == "system":
                    system_prompt_content = msg["content"]
                elif msg["role"] in ["user", "assistant"]:
                    user_assistant_messages.append(msg)
            if system_prompt_content is None:
                print("Warning: No system prompt found. Using a default.")
                system_prompt_content = "You are a helpful Verilog code generator."
            prompts_tuples_for_deluge.append((system_prompt_content, user_assistant_messages))
    
    if not modules_to_process_names: # Should be caught earlier if resume_generation, but as a safeguard
        print("[GEN_HDL]: No modules to process. Exiting.")
        return

    # ───────────────────────────────────────  Back-end - HF  ───────────────────
    if backend == "hf":
        print(f"[GEN_HDL - HF]: Processing {len(modules_to_process_names)} modules with Hugging Face backend.")
        print(f"[GEN_HDL]: Hugging Face Batch Size: {hf_batch_size}")
        if not ds_for_hf: # Check if the dataset for HF is empty after filtering
            print("[GEN_HDL - HF]: No data to process for Hugging Face backend after filtering. Exiting.")
            return

        pretrained_path = (
            f"{cache_dir}/{model}" if "VeriLogos" in model else model
        )
        tok = AutoTokenizer.from_pretrained(
            pretrained_path, padding_side="left",
            trust_remote_code=True, cache_dir=cache_dir
        )
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        net = AutoModelForCausalLM.from_pretrained(
            pretrained_path, torch_dtype="auto",
            device_map="auto", cache_dir=cache_dir
        )

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
            temperature=1.0, top_k=50, top_p=1.0,
            return_full_text=False,
            eos_token_id=tok.eos_token_id, # Explicitly set eos_token_id
            pad_token_id=tok.pad_token_id  # Explicitly set pad_token_id
        )
        print(f"[GEN_HDL - HF]: Starting generation for {len(ds_for_hf_formatted)} prompts with batch size {hf_batch_size}.")
        for i, out in tqdm(enumerate(pipe(KeyDataset(ds_for_hf_formatted, "formatted_messages"), **gen_args)), total=len(ds_for_hf_formatted)):
            module_name = modules_to_process_names[i] # Aligned with ds_for_hf_formatted
            code = out[0]["generated_text"]
            _dump(code, exp_dir, model, module_name, idx_code)

    # ───────────────────────────────────  Back-end - OpenAI  ───────────────────
    elif backend == "api":
        if not modules_to_process_names: # Check if any modules to process for API
            print("[GEN_HDL - API]: No data to process for API backend after filtering. Exiting.")
            return

        if batch_inference is True: # Using lm-deluge
            DELUGE_BATCH_SIZE = 50 # This could be a parameter too
            deluge_client = LLMClient(model_names=[model], max_requests_per_minute=5_000, max_tokens_per_minute=1_000_000,max_concurrent_requests=1_000, sampling_params=SamplingParams(
                temperature=1.0, top_p=1.0, max_new_tokens=4096))
            
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

            print(f"[GEN_HDL]: Processing {len(deluge_conversations)} prompts with lm-deluge.")
            num_total_prompts_deluge = len(deluge_conversations)

            # Batching for lm-deluge
            batched_deluge_prompts = [deluge_conversations[i:i + DELUGE_BATCH_SIZE] for i in range(0, num_total_prompts_deluge, DELUGE_BATCH_SIZE)]
            # Ensure modules_to_process_names is used for batching module names
            batched_module_names_deluge = [modules_to_process_names[i:i + DELUGE_BATCH_SIZE] for i in range(0, num_total_prompts_deluge, DELUGE_BATCH_SIZE)]

            print(f"[GEN_HDL]: Total lm-deluge batches to process: {len(batched_deluge_prompts)}")
            for idx, batch_of_convs in enumerate(batched_deluge_prompts):
                print(f"[GEN_HDL]: Processing lm-deluge batch {idx + 1}/{len(batched_deluge_prompts)} with {len(batch_of_convs)} prompts.")
                current_module_names_batch = batched_module_names_deluge[idx]
                responses = deluge_client.process_prompts_sync(batch_of_convs, show_progress=True)
                
                for j, rsp in enumerate(responses):
                    module_name = current_module_names_batch[j]
                    code = rsp.completion
                    thinking = rsp.thinking
                    if code is None:
                        print(f"[GEN_HDL]: Error generating code for module {module_name}. Response is None.")
                    else:              
                        _dump(code, exp_dir, model, module_name, idx_code)
                        if model in ("deepseek-reasoner", "deepseek-r1") and thinking: # Check if thinking is not None
                            trace_path = f"{exp_dir}/{model}/{module_name}/gen_{module_name}_{idx_code}_trace.txt"
                            os.makedirs(os.path.dirname(trace_path), exist_ok=True)
                            with open(trace_path, "w") as trace_file:
                                trace_file.write(str(thinking)) # Ensure thinking is written as string
        else: # Sequential API calls (no external batching library)
            client = get_client(provider=provider, **provider_kw)
            # Default system prompt, will be prepended if item["messages"] doesn't have one or to standardize
            generic_sys_prompt_content = "You are a helpful Verilog code generator."

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
                if model != "deepseek-reasoner":
                     # Add empty assistant message to prompt for response, if not already ending with one.
                     if not api_payload_messages or api_payload_messages[-1]["role"] != "assistant":
                        api_payload_messages.append({"role": "assistant", "content": ""})
                
                try:
                    rsp = client.chat.completions.create(
                        model=model,
                        messages=api_payload_messages,
                        max_tokens=4096, temperature=1.0, top_p=1.0,
                    )
                    code = rsp.choices[0].message.content
                    _dump(code, exp_dir, model, module_name, idx_code)

                    if model == "deepseek-reasoner" and hasattr(rsp.choices[0].message, 'reasoning_content'):
                        trace = rsp.choices[0].message.reasoning_content
                        if trace: # Ensure trace is not None
                            trace_path = f"{exp_dir}/{model}/{module_name}/gen_{module_name}_{idx_code}_trace.txt"
                            os.makedirs(os.path.dirname(trace_path), exist_ok=True)
                            with open(trace_path, "w") as trace_file:
                                trace_file.write(str(trace))
                except Exception as e:
                    print(f"[GEN_HDL - API Sequential]: Error processing module {module_name}: {e}")
                    print(f"Payload messages: {api_payload_messages}") # Log messages for debugging
                    # Optionally, save a placeholder or skip
                    _dump(f"Error generating code: {e}", exp_dir, model, module_name, idx_code)


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
