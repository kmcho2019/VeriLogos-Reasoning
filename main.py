import os
import argparse
from verilogos.trainer.sft import sft
from verilogos.trainer.rltf import rltf
from verilogos.trainer.rltf_grpo import rltf_grpo
from verilogos.augmentator.hdl_augmentator import augment, augment_custom
from verilogos.generator.hdl_generator import gen_hdl
from verilogos.generator.jsonl_generator import gen_jsonl, gen_reasoning_jsonl
from verilogos.evaluator.hdl_evaluator import evaluate

if __name__ == '__main__':
    """
    Argument
    """
    parser = argparse.ArgumentParser(description="Improving LLM-based Verilog Code Generation with Data Augmentation and RL")
    parser.add_argument("mode", choices=['AUG', 'AUG_CUSTOM', 'SFT', 'RLTF', 'RLTF_GRPO', 'EVAL', 'GEN_HDL', 'GEN_SFT_JSONL', 'GEN_RLTF_JSONL', 'GEN_SFT_CUSTOM_JSONL', 'GEN_REASONING_JSONL', 'FINETUNE_VERILOG_COMPLETION'])
    parser.add_argument("-im", "--input_model", type=str)
    parser.add_argument("-om", "--output_model", type=str)
    parser.add_argument("-d", "--data_jsonl", type=str)
    parser.add_argument("-i", "--idx", type=int, nargs='+', help="One or more integer indices for file generation (e.g., -i 0 1 2).")
    parser.add_argument("-x", "--suffix", type=str)
    parser.add_argument("-mp", "--multiprocess", type=bool, default=False)
    parser.add_argument("-np", "--num_process", type=int)
    parser.add_argument("-ip", "--idx_process", type=int)
    parser.add_argument("-if", "--input_file", type=str)
    parser.add_argument("-of", "--output_file", type=str)
    parser.add_argument("-it", "--iter", type=int)
    parser.add_argument("-be", "--backend", type=str, default="hf", choices=["hf", "api", "vllm"], help="Backend for model generation: Hugging Face or OpenAI API or vLLM") # New arg
    parser.add_argument("-ap", "--api_provider", type=str, default="openai", choices=["openai", "together", "fireworks", "deepseek", "openrouter", "gemini"], help="External API provider for synthetic trace generation") # New arg
    parser.add_argument("-ak", "--api_key", type=str, default=None, help="API key for the External LLM API provider") # New arg
    parser.add_argument("--reference_code_dir", type=str, help="Directory containing reference Verilog snippets for trace generation") # New arg
    parser.add_argument("--synthetic_traces_file", type=str, help="File to save/load synthetic reasoning traces") # New arg
    parser.add_argument("-jm", "--jsonl_method", type=str, default=None, choices=["module", "sentence", "token", "logic", "evaluation"], help="Method for generating JSONL files (either determines masking method or evaluation mode)") # New arg
    parser.add_argument("-rg", "--resume_generation", action='store_true', help="Resume hdl generation from the last generated module file") # New arg
    parser.add_argument("-bi", "--batch_inference", action='store_true', help="Use batch inference (uses package llm-deluge) for faster HDL generation when using external APIs, might cause high credit use") # New arg
    parser.add_argument("-es", "--eval_source_list", type=str, nargs='+', default=['RTLLM', 'VerilogEval'], help="List of sources to evaluate against (default: ['RTLLM', 'VerilogEval'])")
    parser.add_argument("-ep", "--eval-parse-module-name",
                        action='store_true', # Makes it a boolean flag, default is False
                        help="For EVAL mode, parse the module name directly from Verilog file content instead of deriving it from the filename")
    parser.add_argument("-hb", "--hf_batch_size", type=int, default=1, help="Batch size for Huggingface local inference (GEN_HDL-HF mode) (default: 1), only used if backend is set to 'hf', adjust to maximize performance and avoid OOM errors") # New arg
    parser.add_argument("-ft", "--force_thinking", action='store_true', help="For Huggingface local inference (GEN_HDL-HF mode) force the model to start thinking by ensuring that <think> always appears first in text (Only enable for specific thinking models like DeepSeek-R1-Distill series) (default: off)") # New arg
    parser.add_argument("-tp", "--temperature", type=float, default=1.0, help="Temperature for model generation (default: 1.0)") # New arg
    parser.add_argument("-lw", "--lora_weights", type=str, default=None, help="Path to LoRA adapter weights directory to be loaded for Hugging Face models (GEN_HDL-HF mode) (default: None)") # New arg
    parser.add_argument("-crf","--create_resume_file", type=str, default=None, help="Create a resume file consisting of all unfinished modules in the current directory, useful for resuming generation later, exits after just creating the file") # New arg
    parser.add_argument("-rff", "--resume_from_file", type=str, default=None, help="Resume hdl generation from a specific file, useful if there is a specific list of modules to generate") # New arg
    parser.add_argument("-gl", "--generation_length", type=int, default=4096, help="Set the maximum number of output token generated during hdl generation (GEN_HDL) mode (default: 4096)")
    parser.add_argument("-as", "--augment_source", type=str, default="code", help="Source for augmentation(AUG_CUSTOM): 'code' for default code directory or 'custom' for user-specified directory located within ./data directory, the directory must consist of .v files only no nested directories (default: 'code')") # New arg


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
    elif args.mode == "AUG_CUSTOM": # Custom augmentation mode, able to select directory instead of it being fixed to data/code
        augment_custom(0.5, 1, data_dir, exp_dir, args.augment_source)
    elif args.mode == "SFT":
        sft(args.input_model, args.output_model, args.data_jsonl, cache_dir, data_dir)
    elif args.mode == "RLTF":
        rltf(args.input_model, args.output_model, args.data_jsonl, cache_dir, data_dir, exp_dir)
    elif args.mode == "RLTF_GRPO":
        rltf_grpo(args.input_model, args.output_model, args.data_jsonl, cache_dir, data_dir, exp_dir)
    elif args.mode == "EVAL":
        evaluate(args.input_model, num_code, data_dir, exp_dir, args.eval_source_list, parse_module_name_from_content=args.eval_parse_module_name)
    elif args.mode == "GEN_HDL":
        gen_hdl(args.input_model, args.data_jsonl, args.idx, cache_dir, data_dir, exp_dir, args.num_process, 
                args.idx_process, args.backend, args.api_provider, args.api_key, args.resume_generation, 
                args.batch_inference, args.hf_batch_size, args.force_thinking, args.temperature, args.lora_weights,
                args.create_resume_file, args.resume_from_file, args.generation_length)
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
    elif args.mode == "GEN_REASONING_JSONL":
        # Uses synthetic reasoning traces to fine-tune the model on Verilog completion tasks
        # Generates a jsonl file with synthetic traces that can be used for fine-tuning models for reasoning task distilation
        gen_reasoning_jsonl(os.path.join(data_dir, "jsonl", args.data_jsonl), os.path.join(exp_dir, args.input_model), args.output_file)
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



