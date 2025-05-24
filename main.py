import os
import argparse
from verilogos.trainer.sft import sft
from verilogos.trainer.rltf import rltf
from verilogos.augmentator.hdl_augmentator import augment
from verilogos.generator.hdl_generator import gen_hdl
from verilogos.generator.jsonl_generator import gen_jsonl
from verilogos.evaluator.hdl_evaluator import evaluate

if __name__ == '__main__':
    """
    Argument
    """
    parser = argparse.ArgumentParser(description="Improving LLM-based Verilog Code Generation with Data Augmentation and RL")
    parser.add_argument("mode", choices=['AUG', 'SFT', 'RLTF', 'EVAL', 'GEN_HDL', 'GEN_SFT_JSONL', 'GEN_RLTF_JSONL', 'GEN_SFT_CUSTOM_JSONL', 'GEN_SYNTHETIC_TRACES', 'FINETUNE_VERILOG_COMPLETION'])
    parser.add_argument("-im", "--input_model", type=str)
    parser.add_argument("-om", "--output_model", type=str)
    parser.add_argument("-d", "--data_jsonl", type=str)
    parser.add_argument("-i", "--idx", type=int)
    parser.add_argument("-x", "--suffix", type=str)
    parser.add_argument("-mp", "--multiprocess", type=bool, default=False)
    parser.add_argument("-np", "--num_process", type=int)
    parser.add_argument("-ip", "--idx_process", type=int)
    parser.add_argument("-if", "--input_file", type=str)
    parser.add_argument("-of", "--output_file", type=str)
    parser.add_argument("-it", "--iter", type=int)
    parser.add_argument("-be", "--backend", type=str, default="hf", choices=["hf", "api"], help="Backend for model generation: Hugging Face or OpenAI")
    parser.add_argument("-ap", "--api_provider", type=str, default="openai", choices=["openai", "together", "fireworks", "deepseek", "openrouter", "gemini"], help="External API provider for synthetic trace generation") # New arg
    parser.add_argument("-ak", "--api_key", type=str, default=None, help="API key for the External LLM API provider") # New arg
    parser.add_argument("--reference_code_dir", type=str, help="Directory containing reference Verilog snippets for trace generation") # New arg
    parser.add_argument("--synthetic_traces_file", type=str, help="File to save/load synthetic reasoning traces") # New arg
    parser.add_argument("-jm", "--jsonl_method", type=str, default=None, choices=["module", "sentence", "token", "logic", "evaluation"], help="Method for generating JSONL files (either determines masking method or evaluation mode)") # New arg

    args = parser.parse_args()

    """
    Directories
    """
    cache_dir = os.path.realpath('./cache')
    data_dir = os.path.realpath('./data')
    exp_dir = os.path.realpath('./exp')

    """
    Hyperparameters
    """
    num_code = 20

    """
    Main body
    """
    if args.mode == "AUG":
        augment(0.5, 1, data_dir, exp_dir)
    elif args.mode == "SFT":
        sft(args.input_model, args.output_model, args.data_jsonl, cache_dir, data_dir)
    elif args.mode == "RLTF":
        rltf(args.input_model, args.output_model, args.data_jsonl, cache_dir, data_dir, exp_dir)
    elif args.mode == "EVAL":
        evaluate(args.input_model, num_code, data_dir, exp_dir)
    elif args.mode == "GEN_HDL":
        gen_hdl(args.input_model, args.data_jsonl, args.idx, cache_dir, data_dir, exp_dir, args.num_process, args.idx_process, args.backend, args.api_provider, args.api_key)
    elif args.mode == "GEN_SFT_JSONL":
        gen_jsonl("SFT", args.input_file, args.output_file)
    elif args.mode == "GEN_RLTF_JSONL":
        gen_jsonl("RLTF", args.input_file, args.output_file)
    elif args.mode == "GEN_SFT_CUSTOM_JSONL":
        if args.jsonl_method is None:
            raise ValueError("jsonl_method must be specified for GEN_SFT_CUSTOM_JSONL mode.")
        if args.jsonl_method not in ["module", "sentence", "token", "logic", "evaluation"]:
            raise ValueError("jsonl_method must be one of ['module', 'sentence', 'token', 'logic', 'evaluation'].")
        gen_jsonl("SFT", args.input_file, args.output_file, method=args.jsonl_method, add_name=True)
    elif args.mode == "GEN_SYNTHETIC_TRACES":
        print(f"Mode {args.mode} selected. Placeholder for generating synthetic traces.")
        # Ensure new directory exists
        os.makedirs(os.path.join(exp_dir, "synthetic_traces"), exist_ok=True)
        # generate_and_filter_traces(
        #     reference_code_dir=args.reference_code_dir,
        #     output_traces_file=args.synthetic_traces_file or os.path.join(exp_dir, "synthetic_traces", "filtered_traces.jsonl"),
        #     r1_api_provider=args.r1_api_provider,
        #     r1_api_key=args.r1_api_key,
        #     cache_dir=cache_dir, # For any models loaded by verifier
        #     data_dir=data_dir,   # For golden RTL location by verifier
        #     exp_dir=exp_dir      # For temporary work_dirs by verifier
        # )
        pass
    elif args.mode == "FINETUNE_VERILOG_COMPLETION":
        print(f"Mode {args.mode} selected. Placeholder for fine-tuning on synthetic traces.")
        # finetune_on_traces(
        #     input_model=args.input_model, # This would be your R1-distilled model
        #     output_model=args.output_model or f"{args.input_model}-verilogcomp",
        #     traces_file=args.synthetic_traces_file or os.path.join(exp_dir, "synthetic_traces", "filtered_traces.jsonl"),
        #     cache_dir=cache_dir,
        #     data_dir=data_dir # Though likely not used directly for data loading here, SFT might expect it
        # )
        pass
    else:
        parser.print_help()



