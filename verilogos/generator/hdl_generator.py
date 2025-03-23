import os
import torch
import transformers
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.pipelines.pt_utils import KeyDataset
from tqdm import tqdm

def gen_hdl(model, data_jsonl, idx_code, cache_dir, data_dir, exp_dir, num_process, idx_process):
    if 'VeriLogos' in model:
        pretrained_model_name_or_path = f'{cache_dir}/{model}'
    else :
        pretrained_model_name_or_path = model
    print(f'[GEN_HDL]: Generating Verilog HDL code with {model}')

    """
    tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, padding_side="left", trust_remote_code=True, cache_dir=cache_dir)

    """
    model
    """
    model_config = {
        "pretrained_model_name_or_path": pretrained_model_name_or_path,
        "torch_dtype": "auto",
        "device_map": "auto",
        "cache_dir": cache_dir
    }
    m = AutoModelForCausalLM.from_pretrained(**model_config)

    """
    dataset
    """
    data_path = os.path.join(data_dir, 'jsonl', data_jsonl)
    dataset = load_dataset("json", data_files=data_path, split="train")

    if num_process != None and idx_process != None:
        dataset = slice_dataset(dataset, num_process, idx_process)
    dataset = dataset.map(lambda x: {"formatted_messages": tokenizer.apply_chat_template(x["messages"], tokenize=False, add_generation_prompt=True)})

    """
    pipeline
    """
    pipeline_config = {
        "task": "text-generation",
        "model": m,
        "tokenizer": tokenizer,
        "framework": 'pt',
        "use_fast": True,
        "num_workers" : torch.cuda.device_count() * 8,
        'batch_size': 1,
        "device_map": "auto",
        "torch_dtype": "auto"
    }
    pipeline = transformers.pipeline(**pipeline_config)

    """
    generation
    """
    generate_config = {
        'text_inputs': KeyDataset(dataset, "formatted_messages"),
        'max_new_tokens' : 4096,
        'min_new_tokens' : 0,
        'do_sample' : True,
        'use_cache' : True,
        'num_return_sequences': 1,
        'return_full_text': False,
        'temperature': 1.0,
        'top_k': 50,
        'top_p': 1.0
    }

    if 'name' in dataset.column_names:
        modules = [item['name'] for item in dataset]
    else :
        modules = [item['index'] for item in dataset]

    for i, results in enumerate(tqdm(pipeline(**generate_config))):
        module = modules[i]
        module_code = results[0]['generated_text']

        output_path = f'{exp_dir}/{model}/{module}/gen_{module}_{idx_code}.v'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(module_code)

def slice_dataset(dataset, num_process, idx_process):
    total_size = len(dataset)
    chunk_size = total_size // num_process

    start_idx = idx_process * chunk_size
    if idx_process == num_process - 1:
        end_idx = total_size
    else:
        end_idx = (idx_process + 1) * chunk_size

    sliced_dataset = dataset.select(range(start_idx, end_idx))

    return sliced_dataset

