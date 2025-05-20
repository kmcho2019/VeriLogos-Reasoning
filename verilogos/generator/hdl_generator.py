import os
import torch
import transformers
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.pipelines.pt_utils import KeyDataset
from tqdm import tqdm

from openai import OpenAI

_REGISTRY = {
    "openai":   ("https://api.openai.com/v1",      "OPENAI_API_KEY"),
    "groq":     ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "together": ("https://api.together.xyz/v1",    "TOGETHER_API_KEY"),
    "fireworks":("https://api.fireworks.ai/v1",    "FIREWORKS_API_KEY"),
    "deepseek":("https://api.deepseek.com/v1", "DEEPSEEK_API_KEY"),
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
    model,                 # Hugging-Face repo OR OpenAI model id
    data_jsonl,
    idx_code,
    cache_dir,
    data_dir,
    exp_dir,
    num_process=None,
    idx_process=None,
    backend="hf",          # "hf" (default) | "api"
    provider="openai", # "openai" | "together" | "fireworks" | "deepseek_api"
    api_key=None,        # set here or via $OPENAI_API_KEY
    **provider_kw,
):
    """
    Generate Verilog HDL either with a local Hugging-Face model
    or with an OpenAI hosted model (ChatCompletion API).
    """
    print(f'[GEN_HDL]: Generating Verilog HDL code with {model}')
    # ─────────────────────────────────────────  Dataset  ────────────────────────
    data_path = os.path.join(data_dir, "jsonl", data_jsonl)
    ds = load_dataset("json", data_files=data_path, split="train")
    if num_process is not None and idx_process is not None:
        ds = slice_dataset(ds, num_process, idx_process)

    # decide once so both back-ends share identical prompts
    prompts = [item["messages"] for item in ds]          # list[list[dict]]
    modules   = [
        item["name"] if "name" in ds.column_names else str(item["index"])
        for item in ds
    ]
    # modules = (
    #     [item["name"]  if "name"  in ds.column_names else item["index"]]
    #     for item in ds
    # )

    # ───────────────────────────────────────  Back-end - HF  ───────────────────
    if backend == "hf":
        pretrained_path = (
            f"{cache_dir}/{model}" if "VeriLogos" in model else model
        )
        tok = AutoTokenizer.from_pretrained(
            pretrained_path, padding_side="left",
            trust_remote_code=True, cache_dir=cache_dir
        )
        # Ensure the tokenizer has a pad_token if it's missing
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        net = AutoModelForCausalLM.from_pretrained(
            pretrained_path, torch_dtype="auto",
            device_map="auto", cache_dir=cache_dir
        )

        # Apply chat template to create 'formatted_messages' column
        ds = ds.map(
            lambda x: {"formatted_messages": tok.apply_chat_template(
                x["messages"], tokenize=False, add_generation_prompt=True
            )}
        )

        pipe = transformers.pipeline(
            task="text-generation",
            model=net,
            tokenizer=tok,
            framework="pt",
            batch_size=1,
            num_workers=torch.cuda.device_count() * 8,
            device_map="auto",
            torch_dtype="auto"
        )
        gen_args = dict(
            max_new_tokens=4096, do_sample=True,
            temperature=1.0, top_k=50, top_p=1.0,
            return_full_text=False
        )

        for i, out in tqdm(enumerate(pipe(KeyDataset(ds, "formatted_messages"), **gen_args)), total=len(ds)):
            module_name = modules[i] # Get corresponding module name
            code = out[0]["generated_text"]
            _dump(code, exp_dir, model, module_name, idx_code)

    # ───────────────────────────────────  Back-end - OpenAI  ───────────────────
    elif backend == "api":
        client = get_client(provider=provider, **provider_kw)
        sys_prompt = {"role": "system",
                      "content": "You are a helpful Verilog code generator."}

        for mod, msgs in tqdm(zip(modules, prompts), total=len(modules)):
            rsp = client.chat.completions.create(
                model=model,
                messages=[sys_prompt, *msgs, {"role": "assistant", "content": ""}],
                max_tokens=4096, temperature=1.0, top_p=1.0,
            )
            code = rsp.choices[0].message.content
            _dump(code, exp_dir, model, mod, idx_code)
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
    chunk_size = total_size // num_process

    start_idx = idx_process * chunk_size
    if idx_process == num_process - 1:
        end_idx = total_size
    else:
        end_idx = (idx_process + 1) * chunk_size

    sliced_dataset = dataset.select(range(start_idx, end_idx))

    return sliced_dataset

