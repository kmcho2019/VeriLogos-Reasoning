import os
import pandas as pd
import re
from math import comb
from multiprocessing import Pool
from verilogos.utils.parser import parse_fm, parse_module_decl_cont
from verilogos.utils.status import Status
from verilogos.utils.syntax import check_syntax
from verilogos.utils.functionality import check_functionality

def process_verilog_code(input_string):
    # Try to filter out content before </think> tag to exclude reasoning traces
    think_tag_end = "</think>"
    think_tag_index = input_string.find(think_tag_end)
    if think_tag_index != -1:
        # If </think> tag is found, only consider content after it
        processed_string = input_string[think_tag_index + len(think_tag_end):]
    else:
        # If </think> tag is not found, use the entire input string
        processed_string = input_string

    pattern = r"```(.*?)```"
    match = re.search(pattern, processed_string, re.DOTALL)

    if match:
        code = match.group(1).strip()
    else:
        code = processed_string  

    pattern2 = r'\bmodule\b.*?endmodule'
    match2 = re.search(pattern2, code, re.DOTALL)

    if match2:
        return match2.group(0)  
    else:
        return code  
        
def calculate_pass_at_k(scores, k, mode='functionality'):
    n = len(scores)
    if mode == 'syntax':
        c = sum(1 for score in scores if score != -1)
    else:
        c = sum(1 for score in scores if score == 1)

    if c == 0:
        return 0.0
    if n - c < k:
        return 1.0
    pass_at_k = 1 - (comb(n - c, k) / comb(n, k))

    return pass_at_k

def evaluate(model, num_code, data_dir, exp_dir, source_list = ['RTLLM', 'VerilogEval'], parse_module_name_from_content=False):
    print(f'[EVAL]: Evaluating {model} with {source_list}...')

    tasks = []
    for source in source_list:
        for module in os.listdir(f'{data_dir}/{source}'):
            for i in range(num_code):
                work_dir = f'{exp_dir}/{model}/{module}/{i}'

                gen_rtl = f'{exp_dir}/{model}/{module}/gen_{module}_{i}.v'
                if not os.path.isfile(gen_rtl):
                    continue

                tasks.append((model, module, source, i, data_dir, exp_dir, work_dir, parse_module_name_from_content))

    with Pool(30) as pool:
        pool.map(evaluate_code_wrapper, tasks)

    write_report(model, data_dir, exp_dir, source_list, num_code)

def evaluate_code(model, module, module_source, module_idx, data_dir, exp_dir, work_dir, parse_module_name_from_content):
    os.makedirs(work_dir, exist_ok=True)

    gen_rtl = f'{exp_dir}/{model}/{module}/gen_{module}_{module_idx}.v'
    gol_rtl = f'{data_dir}/{module_source}/{module}/{module}.v'

    with open(gen_rtl, 'r') as f:
        gen_content = f.read()

    gen_content = process_verilog_code(gen_content)
    gen_rtl = gen_rtl.replace('.v', '_mod.v')
    with open(gen_rtl, 'w') as f:
        f.write(gen_content)

    # If the module name does not match the file name, we need to parse it
    # and use it for functionality checking
    if parse_module_name_from_content:
        match = re.search(r"^\s*module\s+([a-zA-Z_][a-zA-Z0-9_]*)", gen_content, re.MULTILINE)
        if match:
            module = match.group(1)
        else:
            print(f"Warning: Could not parse module name from content in {gen_rtl}. Using original module name {module}.")

    if check_syntax(gen_rtl) == Status.FAIL:
        score = -1
    elif check_functionality(gen_rtl, gol_rtl, module, work_dir) == Status.FAIL:
        score = 0
    else :
        score = parse_fm(f'{work_dir}/{module}.fm.log')

    with open(f'{work_dir}/{module}.score', 'w') as f:
        f.write(f'{score}')

def evaluate_code_wrapper(args):
    model, module, source, i, data_dir, exp_dir, work_dir, parse_module_name_from_content = args
    evaluate_code(model, module, source, i, data_dir, exp_dir, work_dir, parse_module_name_from_content)

def write_report(model, data_dir, exp_dir, source_list, num_code):
    module_results = []

    for source in source_list:
        module_dirs = [d for d in os.listdir(f'{data_dir}/{source}') if os.path.isdir(os.path.join(f'{data_dir}/{source}', d))]

        for module in module_dirs:
            if not os.path.isdir(f'{exp_dir}/{model}/{module}'):
                continue
            
            scores = []
            for i in range(num_code):
                score_file = f'{exp_dir}/{model}/{module}/{i}/{module}.score'
                if os.path.isfile(score_file):
                    with open(score_file, 'r') as f:
                        score = float(f.read().strip())
                        scores.append(score)

            pass_at_1_syntax = calculate_pass_at_k(scores, 1, mode='syntax')
            pass_at_5_syntax = calculate_pass_at_k(scores, 5, mode='syntax')
            pass_at_10_syntax = calculate_pass_at_k(scores, 10, mode='syntax')

            pass_at_1_functionality = calculate_pass_at_k(scores, 1, mode='functionality')
            pass_at_5_functionality = calculate_pass_at_k(scores, 5, mode='functionality')
            pass_at_10_functionality = calculate_pass_at_k(scores, 10, mode='functionality')

            module_results.append({
                'module': module,
                'source': source,
                'pass@1_syntax': pass_at_1_syntax,
                'pass@5_syntax': pass_at_5_syntax,
                'pass@10_syntax': pass_at_10_syntax,
                'pass@1_functionality': pass_at_1_functionality,
                'pass@5_functionality': pass_at_5_functionality,
                'pass@10_functionality': pass_at_10_functionality
            })

        source_df = pd.DataFrame([result for result in module_results if result['source'] == source])
        mean_pass_at_1_syntax = source_df['pass@1_syntax'].mean()
        mean_pass_at_5_syntax = source_df['pass@5_syntax'].mean()
        mean_pass_at_10_syntax = source_df['pass@10_syntax'].mean()

        mean_pass_at_1_functionality = source_df['pass@1_functionality'].mean()
        mean_pass_at_5_functionality = source_df['pass@5_functionality'].mean()
        mean_pass_at_10_functionality = source_df['pass@10_functionality'].mean()

        module_results.append({
            'module': 'mean',
            'source': source,
            'pass@1_syntax': mean_pass_at_1_syntax,
            'pass@5_syntax': mean_pass_at_5_syntax,
            'pass@10_syntax': mean_pass_at_10_syntax,
            'pass@1_functionality': mean_pass_at_1_functionality,
            'pass@5_functionality': mean_pass_at_5_functionality,
            'pass@10_functionality': mean_pass_at_10_functionality
        })

    df = pd.DataFrame(module_results)

    mean_pass_at_1_syntax = df['pass@1_syntax'].mean()
    mean_pass_at_5_syntax = df['pass@5_syntax'].mean()
    mean_pass_at_10_syntax = df['pass@10_syntax'].mean()

    mean_pass_at_1_functionality = df['pass@1_functionality'].mean()
    mean_pass_at_5_functionality = df['pass@5_functionality'].mean()
    mean_pass_at_10_functionality = df['pass@10_functionality'].mean()

    df.loc['mean'] = {
        'module': 'mean',
        'source': 'all',
        'pass@1_syntax': mean_pass_at_1_syntax,
        'pass@5_syntax': mean_pass_at_5_syntax,
        'pass@10_syntax': mean_pass_at_10_syntax,
        'pass@1_functionality': mean_pass_at_1_functionality,
        'pass@5_functionality': mean_pass_at_5_functionality,
        'pass@10_functionality': mean_pass_at_10_functionality
    }

    df.to_csv(f'{exp_dir}/{model}/evaluation.csv', index=False)
