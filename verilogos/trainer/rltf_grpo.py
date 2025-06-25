import os
import torch
import multiprocessing
import wandb
import re
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOTrainer, GRPOConfig
from tqdm import tqdm
from verilogos.utils.io import print_trainable_parameters
from verilogos.utils.parser import parse_module_name, parse_fm
from verilogos.utils.status import Status
from verilogos.utils.syntax import check_syntax
from verilogos.utils.functionality import check_functionality

def build_dataset(dataset, tokenizer):
   dataset = dataset.map(
       lambda x: {
           "formatted_messages": tokenizer.apply_chat_template(x["messages"], tokenize=False, add_generation_prompt=True)
       }
   )

   def tokenize(sample):
       sample["input_ids"] = tokenizer.encode(sample["formatted_messages"])
       sample["query"] = tokenizer.decode(sample["input_ids"])
       sample["prompt"] = sample["query"]
       return sample

   dataset = dataset.map(tokenize, batched=False)
   dataset.set_format(type="torch")
   dataset = dataset.remove_columns(["messages", "formatted_messages"])

   return dataset

def compute_reward(gen_code, ref_code, work_dir):
    os.makedirs(work_dir, exist_ok=True)

    with open(ref_code, 'r') as f:
        ref_content = f.read()
    name = parse_module_name(ref_content)

    if check_syntax(gen_code) == Status.FAIL:
        score = 0.0
    elif check_functionality(gen_code, ref_code, name, work_dir) == Status.FAIL:
        score = 0.5
    else:
        if parse_fm(f'{work_dir}/{name}.fm.log') == 1:
            score = 1.0
        else:
            score = 0.5

    with open(f'{work_dir}/{name}.score', 'w') as f:
        f.write(f'{score}')

    return score

def grpo_reward_function(prompts, completions, **kwargs):
    """
    A wrapper for the custom reward function to be used with GRPOTrainer.
    """
    # Assuming 'name', 'data_dir', and 'exp_dir' are available.
    # 'name' comes from the dataset, others need to be passed.
    # We can use a partial function or a class to hold these states.
    data_dir = kwargs.pop("data_dir")
    exp_dir = kwargs.pop("exp_dir")
    output_model_name = kwargs.pop("output_model_name")
    
    rewards = []
    for i, completion in enumerate(completions):
        # Extract the Verilog code from the completion
        verilog_code = response_to_netlist_str(completion)
        
        # Get the sample's unique identifier
        sample_name = kwargs["name"][i]
        
        # Define paths for generated and reference files
        work_dir = f'{exp_dir}/RLTF/{output_model_name}/{sample_name}'
        gen_path = f'{work_dir}/gen_{sample_name}.v'
        gen_path_raw = f'{work_dir}/gen_{sample_name}_raw.txt'
        ref_path = f'{data_dir}/rltf_code/{sample_name}.v'
        
        os.makedirs(os.path.dirname(gen_path), exist_ok=True)
        with open(gen_path, 'w') as f:
            f.write(verilog_code)
        with open(gen_path_raw, 'w') as f:
            f.write(completion)

        # Compute the reward for the single instance
        reward = compute_reward(gen_path, ref_path, work_dir)
        rewards.append(reward)

    return rewards

def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])

def rltf_grpo(input_model, output_model, data_jsonl, cache_dir, data_dir, exp_dir):
    # Set environment variables for better performance and to avoid warnings
    torch.set_float32_matmul_precision('high')
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'

    multiprocessing.set_start_method('spawn', force=True)

    if 'VeriLogos' in input_model:
        pretrained_model_name_or_path = f'{cache_dir}/{input_model}'
    else :
        pretrained_model_name_or_path = input_model
    print(f'[RLTF]: Running RLTF [{input_model} >> {output_model}]')

    """
    wandb
    """
    wandb.init()

    """
    Tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path,
        padding_side="left",
        trust_remote_code=True
    )

    """
    Dataset
    """
    data_path = os.path.join(data_dir, 'jsonl', data_jsonl)
    dataset = load_dataset("json", data_files=data_path, split="train")
    dataset = build_dataset(dataset, tokenizer)

    print(dataset)
    # print(load_dataset("trl-lib/tldr", split="train"))

    """
    Model
    """
    model_config = {
        "pretrained_model_name_or_path": pretrained_model_name_or_path,
        "torch_dtype": "auto",
        #"device_map": 'auto',
        "cache_dir": cache_dir,
    }
    model = AutoModelForCausalLM.from_pretrained(**model_config)

    """
    GRPO Trainer
    """
    config = GRPOConfig(
        output_dir=os.path.join(output_model, 'GRPO'),
        per_device_train_batch_size=2,
        num_generations=4,
        bf16= True if torch.cuda.is_bf16_supported() else False,
        gradient_checkpointing=True,
        max_completion_length=4096,
        temperature=0.2,
        top_p=0.95,
        top_k=50,
        #log_with='wandb',
        #batch_size=16,
        #mini_batch_size=1,
        #optimize_cuda_cache=True,
        #use_score_norm=True,
        #use_score_scaling=True,
        num_train_epochs = 1,
        remove_unused_columns=False
    )

    # We need to pass extra arguments to our reward function.
    # We can use a lambda or functools.partial for this.
    # from functools import partial
    # reward_fn = partial(
    #     grpo_reward_function,
    #     data_dir=data_dir,
    #     exp_dir=exp_dir,
    #     output_model_name=output_model
    # )

    # Use lambda to generate reward_fn
    reward_fn = lambda prompts, completions, **kwargs: grpo_reward_function(
        prompts, completions,
        data_dir=data_dir,
        exp_dir=exp_dir,
        output_model_name=output_model,
        **kwargs  # Pass any additional kwargs
    )



    grpo_trainer = GRPOTrainer(
        args=config, 
        model=model, 
        processing_class=tokenizer, 
        reward_funcs=reward_fn, 
        train_dataset=dataset, 
    )
    print_trainable_parameters('RLTF_GRPO', grpo_trainer.model)

    """
    GRPO
    """
    generation_kwargs = {
        "min_length": -1,
        "top_k": 100,
        "top_p": 0.95,
        "do_sample": True,
        "max_new_tokens": 4096,
        "pad_token_id": tokenizer.pad_token_id
    }

    print('[RLTF_GRPO]: Starting training...')
    grpo_trainer.train()
    #grpo_trainer.train(generation_kwargs=generation_kwargs)

    tune_path = os.path.join(cache_dir, f'{output_model}')
    grpo_trainer.save_pretrained(tune_path)
    print(f"[RLTF_GRPO]: Training complete. Model saved to {tune_path}")

def response_to_netlist(input_string, gen_path):
    os.makedirs(os.path.dirname(gen_path), exist_ok=True)

    pattern = r"```verilog(.*?)```"

    match = re.search(pattern, input_string, re.DOTALL)

    if match:
        verilog_code = match.group(1).strip()
        with open(gen_path, 'w') as f:
            f.write(verilog_code)
    else:
        with open(gen_path, 'w') as f:
            f.write(input_string)

def response_to_netlist_str(input_string):
    """
    Extracts Verilog code from the input string and returns it as a string.
    If no Verilog code is found, returns the original input string.
    """
    pattern = r"```verilog(.*?)```"
    match = re.search(pattern, input_string, re.DOTALL)

    if match:
        return match.group(1).strip()
    else:
        return input_string


