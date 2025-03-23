import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer
from verilogos.utils.io import print_trainable_parameters

def sft(input_model, output_model, data_jsonl, cache_dir, data_dir):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if 'VeriLogos' in input_model:
        pretrained_model_name_or_path = f'{cache_dir}/{input_model}'
    else :
        pretrained_model_name_or_path = input_model
    print(f'[SFT]: Running supervised fine-tuning [{input_model} >> {output_model}]')

    """
    Tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, padding_side='right', trust_remote_code=True, cache_dir=cache_dir)
    tokenizer.truncation_side = 'left'

    if "CodeLlama" in input_model:
        print("Adding special tokens")
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    """
    Dataset
    """
    data_path = os.path.join(data_dir, 'jsonl', data_jsonl)
    dataset = load_dataset("json", data_files=data_path, split="train")

    """
    Model
    """
    model_config = {
        "pretrained_model_name_or_path": pretrained_model_name_or_path,
        "torch_dtype": "auto",
        "device_map": 'auto',
        "cache_dir": cache_dir,
    }
    model = AutoModelForCausalLM.from_pretrained(**model_config)

    if "CodeLlama" in input_model:
        model.resize_token_embeddings(len(tokenizer))

    """
    SFTTrainer
    """
    trainer_config = {
        "model": model,
        "train_dataset": dataset,
        "tokenizer": tokenizer,
        "max_seq_length": 2048,
    }
    trainer = SFTTrainer(**trainer_config)
    print_trainable_parameters('SFT', trainer.model)

    trainer.train()
    trainer.save_model(os.path.join(cache_dir, output_model))

