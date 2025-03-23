import os
import copy
import pandas as pd
from pyverilog.vparser.parser import VerilogCodeParser
from pyverilog.ast_code_generator.codegen import ASTCodeGenerator
from pyverilog.vparser.ast import *
from modification import visit_node_change, get_max_depth
from insertion import visit_node_insertion
from deletion import collect_names, count_identifier_usages, find_missing_elements, find_and_remove_parameters, visit_node_delete
from multiprocessing import Pool
from tqdm import tqdm

def augment_code(code_info):
    code_path, prob, num_aug = code_info

    codes, errors = [], []

    try:
        codeparser = VerilogCodeParser([code_path],
                                       preprocess_output=os.path.join('.', os.path.basename(code_path)),
                                       preprocess_include=None,
                                       preprocess_define=None)
        
        ast_original = codeparser.parse()
        depth = get_max_depth(ast_original)

        for i in range(num_aug):
            ast = copy.deepcopy(ast_original)

            for i in range(0, depth + 1):
                parameter_names, input_names, output_names, wire_names, reg_names = collect_names(ast)
                parameter_usage, input_usage, output_usage, wire_usage, reg_usage = count_identifier_usages(ast, parameter_names, input_names, output_names, wire_names, reg_names)
                visit_node_delete(ast, i, prob, prob, parameter_usage, input_usage, output_usage, wire_usage, reg_usage)
            missing_parameter, missing_input, missing_output, missing_wire, missing_reg = find_missing_elements(parameter_usage, input_usage, output_usage, wire_usage, reg_usage)

            for i in range(0, depth + 1):
                missing_parameter, missing_input, missing_output, missing_wire, missing_reg = find_missing_elements(parameter_usage, input_usage, output_usage, wire_usage, reg_usage)
                find_and_remove_parameters(ast,i, missing_parameter, missing_input, missing_output, missing_wire, missing_reg)

            for i in range(0, depth + 1):
                visit_node_change(ast, i, prob)
                visit_node_insertion(ast, i, prob)

            codegen = ASTCodeGenerator()
            code = codegen.visit(ast)
            codes.append((os.path.basename(code_path), code))

    except Exception as e:
        errors.append(f"Error processing file {code_path}: {str(e)}")

    return codes, errors

def augment(prob, num_aug, data_dir, exp_dir):
    print(f'[AUG]: Augmenting Verilog files in {data_dir} with prob={prob} and num_aug={num_aug}.')

    code_paths = sorted([os.path.join(f'{data_dir}/code', f) for f in os.listdir(f'{data_dir}/code') if f.endswith('.v')])
    code_infos = [(code_path, prob, num_aug) for code_path in code_paths]

    results, errors = [], []

    with Pool(processes=64) as pool:
        for result, error in tqdm(pool.imap_unordered(augment_code, code_infos), total=len(code_infos)):
            if result:
                results.extend(result)
            if error:
                errors.extend(error)

    augmented_df = pd.DataFrame(columns=['source', 'code'])
    for source, code in results:
        new_row = pd.DataFrame([{'source': source, 'code': code}])
        augmented_df = pd.concat([augmented_df, new_row], ignore_index=True)

    augmented_df = augmented_df.sort_values(by='source')

    output_file_path = os.path.join(exp_dir, 'augmentation', 'results.csv')
    augmented_df.to_csv(output_file_path, index=False)
    print(f'[AUG]: Augmentation results saved to: {output_file_path}')

    if errors:
        error_log_path = os.path.join(exp_dir, 'augmentation', 'errors.log')
        with open(error_log_path, 'w') as log_file:
            for error in errors:
                log_file.write(error + "\n")
        print(f'[AUG]: Augmentation errors saved to: {error_log_path}')

