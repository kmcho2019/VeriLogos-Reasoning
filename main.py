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
    parser.add_argument("mode", choices=['AUG', 'SFT', 'RLTF', 'EVAL', 'GEN_HDL', 'GEN_SFT_JSONL', 'GEN_RLTF_JSONL'])
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
        gen_hdl(args.input_model, args.data_jsonl, args.idx, cache_dir, data_dir, exp_dir, args.num_process, args.idx_process)
    elif args.mode == "GEN_SFT_JSONL":
        gen_jsonl("SFT", args.input_file, args.output_file)
    elif args.mode == "GEN_RLTF_JSONL":
        gen_jsonl("RLTF", args.input_file, args.output_file)



