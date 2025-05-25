import networkx as nx
import pandas as pd
import random
import os
import pyverilog
import shutil
import multiprocessing as mp
from datasets import Dataset
from transformers import AutoTokenizer
from functools import partial
from pyverilog.ast_code_generator.codegen import ASTCodeGenerator
from pyverilog.vparser.ast import *
from pyverilog.vparser.parser import VerilogCodeParser
from verilogos.utils.parser import parse_module_decl_cont

import json
from pathlib import Path

### Adjust it to fit your model ### 
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-6.7b-instruct")

class CustomASTCodeGenerator(ASTCodeGenerator):
    def __init__(self, related_identifiers):
        super().__init__()
        self.related_identifiers = related_identifiers

    def visit(self, node):
        if isinstance(node, Identifier) and node.name in self.related_identifiers:
            return '[BLANK]'
        else:
            return super().visit(node)

def blank_logic(code):
    codeparser = VerilogCodeParser([code],
                                    preprocess_output=os.path.join('.', os.path.basename(code)),
                                    preprocess_include=None,
                                    preprocess_define=None)

    ast = codeparser.parse()

    if check_hierarchy(ast):
        return None, None

    G = nx.DiGraph()

    track_identifier_relations(ast, G)

    target_identifier = find_root_node(G)

    if target_identifier is None:
        return None, None

    related_identifiers = find_all_related_identifiers(target_identifier, G)
    related_identifiers.add(target_identifier)

    custom_codegen = CustomASTCodeGenerator(related_identifiers)
    masked_verilog_code = custom_codegen.visit(ast)

    return masked_verilog_code, target_identifier

def blank_module(code):
    module_decl, _ = parse_module_decl_cont(code)
    return module_decl + "\n[BLANK]\nendmodule"

def blank_sentence(code):
    sentences = code.split(";")  
    if len(sentences) > 1:
        random_sentence = random.choice(range(len(sentences) - 1))
        sentences[random_sentence] = "\n[BLANK]"
    return ";".join(sentences)

def blank_token(code):
    tokens = tokenizer.tokenize(code)

    if len(tokens) > 0:
        random_token = random.choice(range(len(tokens)))
        print(f"Selected token: {tokens[random_token]}")

        tokens[random_token] = "[BLANK]"

    return tokenizer.convert_tokens_to_string(tokens)

def check_hierarchy(ast):
    moduledef_list = []
    for grand_child in ast.children()[0].children():
        if isinstance(grand_child, ModuleDef):
            moduledef_list.append(grand_child)

    if len(moduledef_list) > 1:
        return True
    else:
        return False

def find_all_identifiers(node):
    identifiers = []
    if isinstance(node, Identifier):
        identifiers.append(node.name)
    elif hasattr(node, 'children'):
        for child in node.children():
            identifiers.extend(find_all_identifiers(child))
    return identifiers

def find_all_related_identifiers(target_signal, G):
    return nx.descendants(G, target_signal)

def find_root_node(G):
    root_nodes = [node for node in G.nodes if G.in_degree(node) == 0]

    if root_nodes:
        return random.choice(root_nodes)
    elif G.nodes:
        return random.choice(list(G.nodes))
    else:
        return None

def process_row(row, system_prompt, mode, method, add_name):
    code = row["code"].strip()

    if method == "module":
        command_1 = "The given Verilog code has the section between the module declaration and 'endmodule' left blank as [BLANK]."
        command_2 = "Fill in the [BLANK] part to complete a syntactically and functionally correct Verilog code."
        blank = blank_module(code)
        user_prompt = f"{command_1}\n\n```verilog\n{blank}\n```\n\n{command_2}"
    elif method == "sentence":
        command_1 = "The given Verilog code has a line left blank as [BLANK]."
        command_2 = "Fill in the [BLANK] part to complete a syntactically and functionally correct Verilog code."
        blank = blank_sentence(code)
        user_prompt = f"{command_1}\n\n```verilog\n{blank}\n```\n\n{command_2}"
    elif method == "token":
        command_1 = "The given Verilog code has a token left blank as [BLANK]."
        command_2 = "Fill in the [BLANK] part to complete a syntactically and functionally correct Verilog code."
        blank = blank_token(code)
        user_prompt = f"{command_1}\n\n```verilog\n{blank}\n```\n\n{command_2}"
    elif method == "logic":
        # blank, target = blank_logic(f'/VeriLogos_V2/data/code/{row.name}.v')
        # blank, target = blank_logic(code)
        blank, target = blank_logic(f'./data/code/{row.name}.v')
        command_1 = f"The given Verilog code has the logic related to the module's output, '{target}', left blank as [BLANK]."
        command_2 = "Consider the module's logic and fill in all the [BLANK] parts to complete a syntactically and functionally correct Verilog code."
        if blank is None:
            print(f"Skipping {row.name}...")
            return None
        user_prompt = f"{command_1}\n\n```verilog\n{blank.strip()}\n```\n\n{command_2}"
    elif method == "evaluation":
        module_decl, _ = parse_module_decl_cont(code)
        description = row["description"]
        command_1 = "The given Verilog code is the module declaration part of the Verilog code that you need to generate."
        command_2 = "You need to write the complete Verilog code between '```verilog' and '```'. Also, the Verilog code must start with 'module' and end with 'endmodule'."
        user_prompt = f"{description}\n\n{command_1}\n\n```verilog\n{module_decl}\n```\n\n{command_2}"

    completion = f"Here is the completed Verilog code.\n\n```verilog\n{code}\n```"

    if mode == "SFT":
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": completion}
        ]
    elif mode == "RLTF":
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    else:
        raise("Error: mode must be either SFT or RLTF.")

    if method == "evaluation":
        return {"messages": messages, "name": row["name"]}
    if mode == "RLTF":
        return {"messages": messages, "index": row.name}
    if add_name is not None:
        return {"messages": messages, "name": row["name"]}
    return {"messages": messages}

def gen_jsonl(mode, input_path, output_path, method="desc", add_name=None, num_workers=64):
    if not input_path.endswith('.csv'):
        raise("Error: input_path must be a CSV file.")

    df = pd.read_csv(input_path)

    with open('./ref/system.prompt', 'r') as f:
        system_prompt = f.read()

    process_row_partial = partial(process_row, system_prompt=system_prompt, mode=mode, method=method, add_name=add_name)

    with mp.Pool(num_workers) as pool:
        data_list = pool.map(process_row_partial, [row for _, row in df.iterrows()])

    data_list = [data for data in data_list if data is not None]

    print(len(data_list))

    dataset = Dataset.from_list(data_list)
    dataset.to_json(output_path)

    print(f"[GEN_{mode}_JSONL]: Generated {output_path} with {input_path}...")

def gen_reasoning_jsonl(input_jsonl_path: str, output_dir_path: str, output_file_path: str):
    """
    Generates a JSONL file for fine-tuning reasoning models for RTL LLM generation.

    Args:
        input_jsonl_path (str): Path to the input JSONL file containing initial prompts. (Example: "./data/jsonl/initial_prompts.jsonl")
        output_dir_path (str): Path to the base directory containing model outputs
                                (scores, Verilog files, trace files). (Example: "./exp/deepseek-r1)
        output_file_path (str): Path to save the generated fine-tuning JSONL file.
    """
    input_p = Path(input_jsonl_path)
    output_base_p = Path(output_dir_path)
    output_jsonl_p = Path(output_file_path)

    if not input_p.exists():
        print(f"Error: Input JSONL file not found: {input_jsonl_path}")
        return
    if not output_base_p.is_dir():
        print(f"Error: Output directory not found: {output_dir_path}")
        return

    processed_count = 0
    skipped_count = 0

    with open(output_jsonl_p, 'w', encoding='utf-8') as outfile:
        with open(input_p, 'r', encoding='utf-8') as infile:
            for line_num, line in enumerate(infile):
                try:
                    problem_data = json.loads(line)
                except json.JSONDecodeError:
                    print(f"Warning: Skipping malformed JSON line {line_num + 1} in {input_jsonl_path}")
                    continue

                module_id_original = problem_data.get("name")
                if module_id_original is None:
                    # If module_id is not present, use the index as a fallback
                    module_id_original = line_num
                    print(f"Warning: No 'name' field found in {line_num}th line, using index {line_num} as module_id.")
                    #print(f"Warning: Skipping line {line_num + 1} due to missing 'name' (module_id).")
                    #continue
                
                module_id_str = str(module_id_original) # Ensure string for path construction
                
                system_prompt = ""
                user_prompt = ""
                if "messages" in problem_data and isinstance(problem_data["messages"], list):
                    for msg in problem_data["messages"]:
                        if msg.get("role") == "system":
                            system_prompt = msg.get("content", "")
                        elif msg.get("role") == "user":
                            user_prompt = msg.get("content", "")
                
                if not system_prompt or not user_prompt:
                    print(f"Warning: Skipping module {module_id_str} due to missing system or user prompt.")
                    skipped_count += 1
                    continue

                module_output_path = output_base_p / module_id_str
                if not module_output_path.is_dir():
                    # print(f"Info: Module directory not found for {module_id_str}, skipping.")
                    skipped_count += 1
                    continue

                found_valid_iteration = False
                best_trace_content = ""
                best_verilog_content = ""

                # Iterate over potential generation_iter_num subdirectories
                iter_num_dirs = [d for d in module_output_path.iterdir() if d.is_dir()]
                
                # Sort iter_num_dirs by name (e.g., "0", "1", "10") to process in order
                # This helps if multiple iterations have score 1.0, we pick the "earliest"
                iter_num_dirs.sort(key=lambda x: x.name)


                for iter_dir in iter_num_dirs:
                    generation_iter_num = iter_dir.name # This is the string like "0", "1", etc.
                    
                    highest_score_for_this_iter = -float('inf')
                    score_files = list(iter_dir.glob("*.score"))

                    if not score_files:
                        # print(f"Debug: No score files in {iter_dir} for module {module_id_str}")
                        continue
                    
                    for score_file in score_files:
                        try:
                            with open(score_file, 'r', encoding='utf-8') as sf:
                                score_str = sf.read().strip()
                                score = float(score_str)
                                if score > highest_score_for_this_iter:
                                    highest_score_for_this_iter = score
                        except ValueError:
                            print(f"Warning: Could not parse score in {score_file} for module {module_id_str}, iter {generation_iter_num}.")
                        except Exception as e:
                            print(f"Warning: Error reading score file {score_file}: {e}")
                    
                    if highest_score_for_this_iter == 1.0:
                        # Found a valid iteration for this module_id
                        # Paths for trace and verilog are in the parent module_output_path
                        trace_file_path = module_output_path / f"gen_{module_id_original}_{generation_iter_num}_trace.txt"
                        verilog_file_path = module_output_path / f"gen_{module_id_original}_{generation_iter_num}.v"
                        
                        if trace_file_path.exists() and verilog_file_path.exists():
                            try:
                                with open(trace_file_path, 'r', encoding='utf-8') as tf:
                                    best_trace_content = tf.read()
                                with open(verilog_file_path, 'r', encoding='utf-8') as vf:
                                    best_verilog_content = vf.read()
                                
                                found_valid_iteration = True
                                # print(f"Debug: Found valid iteration {generation_iter_num} for module {module_id_str}")
                                break # Stop checking other iterations for this module_id
                            except Exception as e:
                                print(f"Warning: Error reading trace/Verilog for module {module_id_str}, iter {generation_iter_num}: {e}")
                                # Continue to check other iterations if this one had file read errors
                        else:
                            print(f"Warning: Score 1.0 found for module {module_id_str}, iter {generation_iter_num}, but trace or Verilog file missing.")
                            print(f"  Trace expected: {trace_file_path}")
                            print(f"  Verilog expected: {verilog_file_path}")
                            # Continue to check other iterations

                if found_valid_iteration:
                    assistant_content = f"<think>\n{best_trace_content.strip()}\n</think>\n{best_verilog_content.strip()}"
                    
                    output_entry = {
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                            {"role": "assistant", "content": assistant_content}
                        ]
                    }
                    # Optionally, you might want to include the module name/id in the output for reference
                    # output_entry["name"] = module_id_original 

                    outfile.write(json.dumps(output_entry) + "\n")
                    processed_count += 1
                else:
                    # print(f"Info: No iteration with score 1.0 found for module {module_id_str}, skipping.")
                    skipped_count += 1
    
    print(f"\n[GEN_REASONING_JSONL]: Processing complete.")
    print(f"[GEN_REASONING_JSONL]: Generated fine-tuning data for {processed_count} modules.")
    print(f"[GEN_REASONING_JSONL]: Skipped {skipped_count} modules.")
    print(f"[GEN_REASONING_JSONL]: Output saved to: {output_jsonl_p.resolve()}")


def track_identifier_relations(node, G):
    if isinstance(node, (BlockingSubstitution, NonblockingSubstitution, Assign)):
        if isinstance(node.left.var, Identifier):
            left = node.left.var.name
            G.add_node(left)
            if isinstance(node.right, Identifier):
                right = node.right.var.name
                G.add_edge(left, right)
            else:
                right_identifiers = find_all_identifiers(node.right)
                for right in right_identifiers:
                    G.add_edge(left, right)
        else:
            left = find_all_identifiers(node.left)
            for l in left:
                G.add_node(l)
                if isinstance(node.right, Identifier):
                    right = node.right.var.name
                    G.add_edge(l, right)
                else:
                    right_identifiers = find_all_identifiers(node.right)
                    for right in right_identifiers:
                        G.add_edge(l, right)

    elif hasattr(node, 'children'):
        for child in node.children():
            track_identifier_relations(child, G)
