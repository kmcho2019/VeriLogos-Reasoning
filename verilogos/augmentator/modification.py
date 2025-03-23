import os
import random
import copy
from tqdm import tqdm
from pyverilog.vparser.parser import VerilogCodeParser
from pyverilog.ast_code_generator.codegen import ASTCodeGenerator
from pyverilog.vparser.ast import *
from collections import deque
from multiprocessing import Pool, cpu_count

# Define operator lists
bo_list = (Times, Divide, Mod, Plus, Minus)
so_list = (Sll, Srl, Sla, Sra)
co_list = (LessThan, GreaterThan, LessEq, GreaterEq)
eqo_list = (Eq, NotEq, Eql, NotEql)
go_list = (And, Xor, Xnor, Or, Land, Lor)
unary_list = (Uplus, Uminus, Ulnot, Unot, Uand, Unand, Uor, Unor, Uxor, Uxnor)
sub_list = (Substitution, BlockingSubstitution, NonblockingSubstitution)
case_list = (CaseStatement, CasexStatement)

# Randomize node function
def randomize_node(node, node_classes):
    node_classes = copy.deepcopy(node_classes)
    node_type = type(node)
    if node_type in node_classes:
        node_classes.remove(node_type)
    return random.choice(node_classes)

# Get max depth of the AST tree
def get_max_depth(node):
    if not hasattr(node, 'children') or not node.children():
        return 1
    max_child_depth = 0
    for child in node.children():
        max_child_depth = max(max_child_depth, get_max_depth(child))
    return max_child_depth + 1

def change_node(node, prob):
    bank = []
    for child_node in node.children():
        if isinstance(child_node, bo_list + so_list + co_list + eqo_list + go_list + unary_list + sub_list + case_list):
            for key, value in node.__dict__.items():
                if isinstance(node, Width):
                    continue
                ## If Statement and For Statement Exception Handling
                if isinstance(node, IfStatement) and key is 'cond':
                    continue

                if isinstance(node, ForStatement):
                    if key is 'pre' or key is 'cond' or key is 'post':
                        continue

                if child_node is value:
                    if random.random() < prob:
                        if isinstance(child_node, unary_list):
                            bank.append((node, key, randomize_node(child_node, list(unary_list))(child_node.right, child_node.lineno)))
                        elif isinstance(child_node, sub_list):
                            bank.append((node, key, randomize_node(child_node, list(sub_list))(child_node.left, child_node.right, child_node.ldelay, child_node.rdelay, child_node.lineno)))
                        elif isinstance(child_node, case_list):
                            bank.append((node, key, randomize_node(child_node, list(case_list))(child_node.comp, child_node.caselist, child_node.lineno)))
                        else:
                            bank.append((node, key, randomize_node(child_node, list(bo_list))(child_node.left, child_node.right, child_node.lineno)))
                elif isinstance(value, tuple) and child_node in value:
                    if random.random() < prob:
                        value_list = list(value)
                        value_list.remove(child_node)
                        if isinstance(child_node, unary_list):
                            generated_node = randomize_node(child_node, list(unary_list))(child_node.right, child_node.lineno)
                        elif isinstance(child_node, sub_list):
                            generated_node = randomize_node(child_node, list(sub_list))(child_node.left, child_node.right, child_node.ldelay, child_node.rdelay, child_node.lineno)
                        elif isinstance(child_node, case_list):
                            generated_node = randomize_node(child_node, list(case_list))(child_node.comp, child_node.caselist, child_node.lineno)
                        else:
                            generated_node = randomize_node(child_node, list(bo_list))(child_node.left, child_node.right, child_node.lineno)
                        value_list.append(generated_node)
                        value_tuple = tuple(value_list)
                        bank.append((node, key, value_tuple))
    for node, key, value in bank:
        setattr(node, key, value)

# Process the AST at a specific depth
def visit_node_change(root_node, target_depth, prob=0.5):
    queue = deque([(root_node, 0)])
    while queue:
        node, current_depth = queue.popleft()
        if current_depth == target_depth:
            change_node(node, prob)
        elif hasattr(node, 'children'):
            for child in node.children():
                queue.append((child, current_depth + 1))
    return "Completed processing at target depth."

# Function to process a single Verilog file
def process_verilog_file(file_info):
    file_path, save_dir, prob = file_info
    try:
        codeparser = VerilogCodeParser([file_path],
                                       preprocess_output=os.path.join('.', os.path.basename(file_path)),
                                       preprocess_include=None,
                                       preprocess_define=None)
        ast = codeparser.parse()
        depth = get_max_depth(ast)
        for i in range(0, depth + 1):
            visit_node_change(ast, i, prob)
        codegen = ASTCodeGenerator()
        generated_code = codegen.visit(ast)
        file_name = os.path.basename(file_path)
        output_file_path = os.path.join(save_dir, file_name)
        with open(output_file_path, 'w') as f:
            f.write(generated_code)
        return f"Processed file saved to: {output_file_path}", None
    except Exception as e:
        return None, f"Error processing file {file_path}: {str(e)}"

# Function to process all Verilog files in a folder using multiprocessing
def process_all_verilog_files_in_folder(input_dir, save_dir, prob=0.5):
    verilog_files = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.v')])
    file_info_list = [(file_path, save_dir, prob) for file_path in verilog_files]
    
    errors = []  # To accumulate error messages
    
    # Use multiprocessing pool to process files in parallel
    with Pool(processes=cpu_count()) as pool:
        for result, error in tqdm(pool.imap_unordered(process_verilog_file, file_info_list), total=len(file_info_list)):
            if result:
                print(result)
            if error:
                errors.append(error)
    
    # Write all errors to 'errors.txt' at the end of processing
    if errors:
        error_log_path = os.path.join(save_dir, 'errors.txt')
        with open(error_log_path, 'w') as log_file:
            for error in errors:
                log_file.write(error + "\n")
        print(f"Errors logged to: {error_log_path}")


