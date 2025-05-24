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
