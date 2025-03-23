import os
import torch
import multiprocessing
import wandb
import re
from datasets import load_dataset
from transformers import AutoTokenizer
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
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

def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])

def rltf(input_model, output_model, data_jsonl, cache_dir, data_dir, exp_dir):
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

    """
    Model
    """
    model_config = {
        "pretrained_model_name_or_path": pretrained_model_name_or_path,
        "torch_dtype": "auto",
        "device_map": 'auto',
        "cache_dir": cache_dir,
    }
    model = AutoModelForCausalLMWithValueHead.from_pretrained(**model_config)
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(**model_config)

    """
    PPO Trainer
    """
    config = PPOConfig(
        log_with='wandb',
        batch_size=16,
        mini_batch_size=1,
        optimize_cuda_cache=True,
        use_score_norm=True,
        use_score_scaling=True,
        ppo_epochs = 4,
        remove_unused_columns=False
    )
    ppo_trainer = PPOTrainer(config=config, model=model, ref_model=ref_model, tokenizer=tokenizer, dataset=dataset, data_collator=collator)
    print_trainable_parameters('RLTF', ppo_trainer.model)

    """
    PPO
    """
    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "max_new_tokens": 2048,
        "pad_token_id": tokenizer.pad_token_id
    }

    for epoch in range(config.ppo_epochs):
        print(f'[RLTF]: Starting epoch: {epoch+1}/{config.ppo_epochs}')
        for batch in tqdm(ppo_trainer.dataloader, "[RLTF]: Batch: "):
            # Generate response
            query_tensors = batch["input_ids"]
            response_tensors = ppo_trainer.generate(
                query_tensors, batch_size=8, return_prompt=False, generate_ref_response=False, **generation_kwargs
            )
            batch["response"] = tokenizer.batch_decode(response_tensors)

            # Save response to netlist
            inputs = [(
                response,
                f'{exp_dir}/RLTF/{output_model}/{index}/{epoch}/gen_{index}.v')
                for index, response in zip(batch["index"], batch["response"])
            ]
            with multiprocessing.Pool(processes=min(config.batch_size, 16)) as pool:
                pool.starmap(response_to_netlist, inputs)

            # Compute reward
            inputs = [(
                f'{exp_dir}/RLTF/{output_model}/{index}/{epoch}/gen_{index}.v',
                f'{data_dir}/rltf_code/{index}.v',
                f'{exp_dir}/RLTF/{output_model}/{index}/{epoch}')
                for index in batch["index"]
            ]
            with multiprocessing.Pool(processes=min(config.batch_size, 16)) as pool:
                rewards = pool.starmap(compute_reward, inputs)
            rewards = [torch.tensor(reward, dtype=torch.float32, device='cuda') for reward in rewards]

            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
            ppo_trainer.log_stats(stats, batch, rewards)

        if epoch != config.ppo_epochs-1:
            ckpt_path = os.path.join(cache_dir, f'{output_model}_ckpt_{epoch}')
            ppo_trainer.save_pretrained(ckpt_path)

    tune_path = os.path.join(cache_dir, f'{output_model}')
    ppo_trainer.save_pretrained(tune_path)

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


