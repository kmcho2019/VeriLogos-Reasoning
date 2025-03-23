import random
import pandas as pd
import os
import sys
from pyverilog.vparser.parser import VerilogCodeParser
from pyverilog.ast_code_generator.codegen import ASTCodeGenerator
from pyverilog.vparser.ast import *
from collections import deque
from multiprocessing import Pool, cpu_count

bo_list = (Power, Times, Divide, Mod, Plus, Minus) ## basic operator
so_list = (Sll, Srl, Sla, Sra) ## shift operator
co_list = (LessThan, GreaterThan, LessEq, GreaterEq) ## comparison operator
eqo_list = (Eq, NotEq, Eql, NotEql) ## equality operator
go_list = (And, Xor, Xnor, Or, Land, Lor) ## gate operator
ab_list = (Always, AlwaysFF, AlwaysComb, AlwaysLatch) ## always operator
sub_list = (BlockingSubstitution, NonblockingSubstitution) ## substitution operator
case_list = (CaseStatement, CasexStatement, Case, Assign) ## case + assign
statement_list = (ForeverStatement, EventStatement) 
unary_list = (Uplus, Uminus, Ulnot, Unot, Uand, Unand, Uor, Unor, Uxor, Uxnor) ## unary operator
cond_list = (IfStatement, Cond)

def get_max_depth(node):
    if not hasattr(node, 'children') or not node.children():
        return 1
    max_child_depth = 0
    for child in node.children():
        max_child_depth = max(max_child_depth, get_max_depth(child))
    return max_child_depth + 1

def visit_node_delete(root_node, target_depth, prob_cond, prob_case, parameter_usage, input_usage, output_usage, wire_usage, reg_usage):    
    queue = deque([(root_node, 0)])
    while queue:
        node, current_depth = queue.popleft()
        if current_depth == target_depth:
            delete_node(node, prob_cond, prob_case, parameter_usage, input_usage, output_usage, wire_usage, reg_usage)
        elif hasattr(node, 'children'):
            for child in node.children():
                queue.append((child, current_depth + 1))        
    return "finish"

def delete_node(node, prob_cond, prob_case, parameter_usage, input_usage, output_usage, wire_usage, reg_usage):
    nodes_to_remove = []
    for child_node in node.children():
        try:
            if isinstance(child_node, tuple(cond_list + case_list + sub_list)):
                identifier_count = collect_identifiers_with_count(child_node)
                if(check_identifier_usage(identifier_count, input_usage, output_usage)):
                    continue
                else :
                    for key, value in node.__dict__.items():
                        if child_node is value:
                            if isinstance(child_node, tuple(cond_list)):
                                prob = prob_cond
                            elif isinstance(child_node, tuple(case_list)):
                                prob = prob_case
                            elif isinstance(child_node, tuple(sub_list)):
                                prob = prob_case
                            else:
                                continue
                            if random.random() < prob:
                                nodes_to_remove.append((node, key, child_node))
                                child_identifiers = collect_identifiers_with_count(child_node)
                                for identifier, count in child_identifiers.items():
                                    if identifier in parameter_usage:
                                        parameter_usage[identifier] = max(0, parameter_usage[identifier] - count)
                                    if identifier in input_usage:
                                        input_usage[identifier] = max(0, input_usage[identifier] - count)
                                    if identifier in output_usage:
                                        output_usage[identifier] = max(0, output_usage[identifier] - count)
                                    if identifier in wire_usage:
                                        wire_usage[identifier] = max(0, wire_usage[identifier] - count)
                                    if identifier in reg_usage:
                                        reg_usage[identifier] = max(0, reg_usage[identifier] - count)
                        elif isinstance(value, tuple) and child_node in value:
                            if isinstance(child_node, tuple(cond_list)):
                                prob = prob_cond
                            elif isinstance(child_node, tuple(case_list)):
                                prob = prob_case
                            elif isinstance(child_node, tuple(sub_list)):
                                prob = prob_case
                            else:
                                continue
                            if random.random() < prob:
                                value_list = list(value)
                                value_list.remove(child_node)
                                value_tuple = tuple(value_list)
                                nodes_to_remove.append((node, key, value_tuple))
                                child_identifiers = collect_identifiers_with_count(child_node)
                                for identifier, count in child_identifiers.items():
                                    if identifier in parameter_usage:
                                        parameter_usage[identifier] = max(0, parameter_usage[identifier] - count)
                                    if identifier in input_usage:
                                        input_usage[identifier] = max(0, input_usage[identifier] - count)
                                    if identifier in output_usage:
                                        output_usage[identifier] = max(0, output_usage[identifier] - count)
                                    if identifier in wire_usage:
                                        wire_usage[identifier] = max(0, wire_usage[identifier] - count)
                                    if identifier in reg_usage:
                                        reg_usage[identifier] = max(0, reg_usage[identifier] - count)
        except Exception as e:
            print(f"Error during node processing: {str(e)} at node: {child_node}") 
    for node, key, value in nodes_to_remove:
        setattr(node, key, value)  
    return "finish" 

def collect_identifiers_with_count(node):
    identifier_count = {}

    def traverse(child_node):
        if isinstance(child_node, Identifier):
            if child_node.name in identifier_count:
                identifier_count[child_node.name] += 1
            else:
                identifier_count[child_node.name] = 1
        for sub_child in child_node.children():
            traverse(sub_child)

    traverse(node)
    return identifier_count

def check_identifier_usage(identifier_count, input_usage, output_usage):
    input_usage_copy = input_usage.copy()  
    output_usage_copy = output_usage.copy()  

    for identifier, count in identifier_count.items():
        if identifier in input_usage_copy:
            input_usage_copy[identifier] = max(0, input_usage_copy[identifier] - count)  

    for identifier, count in identifier_count.items():
        if identifier in output_usage_copy:
            output_usage_copy[identifier] = max(0, output_usage_copy[identifier] - count)  

    input_all_zero = all(count == 0 for count in input_usage_copy.values())
    output_all_zero = all(count == 0 for count in output_usage_copy.values())

    return input_all_zero or output_all_zero

def collect_names(ast):
    names_dict = {
        Parameter: [],
        Input: [],
        Output: [],
        Wire: [],
        Reg: []
    }

    def find_node(node):
        for node_type, name_list in names_dict.items():
            if isinstance(node, node_type):
                name_list.append(node.name)

        for child in node.children():
            find_node(child)

    find_node(ast)

    return names_dict[Parameter], names_dict[Input], names_dict[Output], names_dict[Wire], names_dict[Reg]

def count_identifier_usages(ast, parameter_names, input_names, output_names, wire_names, reg_names):
    parameter_usage = {name: 0 for name in parameter_names}
    input_usage = {name: 0 for name in input_names}
    output_usage = {name: 0 for name in output_names}
    wire_usage = {name: 0 for name in wire_names}
    reg_usage = {name: 0 for name in reg_names}

    def traverse_node(node):
        if isinstance(node, Identifier):
            if node.name in parameter_usage:
                parameter_usage[node.name] += 1
            if node.name in input_usage:
                input_usage[node.name] += 1
            if node.name in output_usage:
                output_usage[node.name] += 1
            if node.name in wire_usage:
                wire_usage[node.name] += 1
            if node.name in reg_usage:
                reg_usage[node.name] += 1

        for child in node.children():
            traverse_node(child)

    traverse_node(ast)

    return parameter_usage, input_usage, output_usage, wire_usage, reg_usage

def find_missing_elements(parameter_usage, input_usage, output_usage, wire_usage, reg_usage):
    unused_parameters = [name for name, count in parameter_usage.items() if count == 0]
    unused_inputs = [name for name, count in input_usage.items() if count == 0]
    unused_outputs = [name for name, count in output_usage.items() if count == 0]
    unused_wires = [name for name, count in wire_usage.items() if count == 0]
    unused_regs = [name for name, count in reg_usage.items() if count == 0]

    return unused_parameters, unused_inputs, unused_outputs, unused_wires, unused_regs

def find_and_remove_parameters(root_node, target_depth, missing_parameter, missing_input, missing_output, missing_wire, missing_reg):
    queue = deque([(root_node, 0)])
    while queue:
        node, current_depth = queue.popleft()
        if current_depth == target_depth:
            nodes_to_remove = []
            for child_node in node.children():
                for key, value in node.__dict__.items():
                    if isinstance(value, tuple):
                        value_list = list(value)
                        value_list = [child for child in value_list if not (isinstance(child, Parameter) and child.name in missing_parameter)]
                        value_list = [child for child in value_list if not(isinstance(child, Input) and child.name in missing_input)]
                        value_list = [child for child in value_list if not (isinstance(child, Output) and child.name in missing_output)]
                        value_list = [child for child in value_list if not (isinstance(child, Wire) and child.name in missing_wire)]
                        value_list = [child for child in value_list if not (isinstance(child, Reg) and child.name in missing_reg)]                
                        value_list = [child for child in value_list if not (isinstance(child, Port) and child.name in missing_input)]
                        value_list = [child for child in value_list if not (isinstance(child, Port) and child.name in missing_output)]
                        value_list = [child for child in value_list if not (isinstance(child, Ioport) and ((isinstance(child.first, Input) and child.first.name in missing_input) or (isinstance(child.first, Output) and child.first.name in missing_output)))]
                        value_tuple = tuple(value_list)
                        nodes_to_remove.append((node, key, value_tuple))
            for node, key, value in nodes_to_remove:
                setattr(node, key, value)            
        elif hasattr(node, 'children'):
            for child in node.children():
                queue.append((child, current_depth + 1)) 
    return "finish" 

def process_verilog_code(file_path, prob_cond, prob_case):
    codeparser = VerilogCodeParser([file_path],
                                  preprocess_output=os.path.join('.', os.path.basename(file_path)),
                                  preprocess_include=None,
                                  preprocess_define=None)
    ast = codeparser.parse()
    depth = get_max_depth(ast)
    for i in range(0, depth+1):
        parameter_names, input_names, output_names, wire_names, reg_names = collect_names(ast)
        parameter_usage, input_usage, output_usage, wire_usage, reg_usage = count_identifier_usages(ast, parameter_names, input_names, output_names, wire_names, reg_names)
        visit_node_delete(ast, i, prob_cond, prob_case, parameter_usage, input_usage, output_usage, wire_usage, reg_usage)
    missing_parameter, missing_input, missing_output, missing_wire, missing_reg = find_missing_elements(parameter_usage, input_usage, output_usage, wire_usage, reg_usage)
    for i in range(0, depth+1):
        missing_parameter, missing_input, missing_output, missing_wire, missing_reg = find_missing_elements(parameter_usage, input_usage, output_usage, wire_usage, reg_usage)
        find_and_remove_parameters(ast,i, missing_parameter, missing_input, missing_output, missing_wire, missing_reg)
    codegen = ASTCodeGenerator()
    generated_code = codegen.visit(ast)

    return generated_code

def process_file(args):
    file_path, prob_cond, prob_case = args
    file_name = os.path.basename(file_path)
    print(f"Processing file: {file_name}")
    augmented_code= process_verilog_code(file_path, prob_cond, prob_case)
    return file_name, augmented_code
    
def process_csv_file(infile_path, num_samples, prob_cond, prob_case):
    verilog_files = [os.path.join(infile_path, f) for f in os.listdir(infile_path) if f.endswith('.v')]
    if num_samples == 0:
      sample_modules = verilog_files
    else :
      sample_modules = random.sample(verilog_files, num_samples)
    
    augmented_df = pd.DataFrame(columns=['num', 'module_aug'])

    with Pool(cpu_count()) as pool:
        args = [(file_path, prob_cond, prob_case) for file_path in sample_modules]
        results = pool.map(process_file, args)  

    for file_name, code in results:
        augmented_df = augmented_df.append({'num': file_name, 'module_aug': code}, ignore_index=True)

    output_file_path = 'module_aug.csv'
    
    augmented_df.to_csv(output_file_path, index=False)

    print(f"Processed Verilog codes have been saved to {output_file_path}")



