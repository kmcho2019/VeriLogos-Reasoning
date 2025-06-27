import os
import copy
import pandas as pd
from pyverilog.vparser.parser import VerilogCodeParser
from pyverilog.ast_code_generator.codegen import ASTCodeGenerator
from pyverilog.vparser.ast import *
from .modification import visit_node_change, get_max_depth
from .insertion import visit_node_insertion
from .deletion import collect_names, count_identifier_usages, find_missing_elements, find_and_remove_parameters, visit_node_delete
from multiprocessing import Pool
from tqdm import tqdm
import random

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

# This function generates a controlled set of variants for a given Verilog file.
# It systematically creates variants with specific combinations of deletions, changes, and insertions in the AST.
def augment_code_variants(code_info):
    """
    Generates a controlled set of variants for a given Verilog file.

    This function systematically creates variants with specific combinations of
    deletions, changes, and insertions in the AST.

    Args:
        code_info (tuple): A tuple containing the code path and other info.
                           We only use the path.
            code_path (str): The path to the Verilog file to be augmented.
            prob (float): The probability of applying an augmentation. (Not used as we control the number of variants per type)
            variants_per_type (int): The number of augmented versions to generate per file.


    Returns:
        tuple: A tuple containing a list of generated (filename, code) pairs
               and a list of any errors encountered.
    """
    code_path, _, variants_per_type = code_info
    VERSIONS_PER_TYPE = variants_per_type 
    codes, errors = [], []
    # Use a set to store generated code to ensure all variants are unique
    generated_codes = set()

    try:
        # Initialize the Verilog parser
        codeparser = VerilogCodeParser([code_path],
                                       preprocess_output=os.path.join('.', os.path.basename(code_path)),
                                       preprocess_include=None,
                                       preprocess_define=None)
        
        ast_original = codeparser.parse()
        codegen = ASTCodeGenerator()
        original_code = codegen.visit(ast_original)
        generated_codes.add(original_code)

        # Get the max depth of the AST to guide modifications
        max_depth = get_max_depth(ast_original)
        if max_depth == 0:
            errors.append(f"Warning: Could not process AST for {code_path}, max_depth is 0.")
            return [], errors

        # Define the 7 types of augmentations as a list of tuples:
        # (apply_delete, apply_change, apply_insert)
        variant_configs = [
            (True, False, False),  # 1. Deletion only
            (False, True, False),  # 2. Change only
            (False, False, True),  # 3. Insertion only
            (True, True, False),   # 4. Deletion & Change
            (True, False, True),   # 5. Deletion & Insertion
            (False, True, True),   # 6. Change & Insertion
            (True, True, True),    # 7. All three
        ]

        # Loop through each configuration to generate variants
        for do_delete, do_change, do_insert in variant_configs:
            created_count = 0
            attempts = 0
            # Set a max number of attempts to prevent infinite loops
            MAX_ATTEMPTS_PER_TYPE = 100 

            while created_count < VERSIONS_PER_TYPE and attempts < MAX_ATTEMPTS_PER_TYPE:
                attempts += 1
                ast = copy.deepcopy(ast_original)

                # Use a high probability to encourage at least one modification
                high_prob = 1.0

                # --- 1. Deletion Logic ---
                if do_delete:
                    # This block preserves your original, multi-step deletion process
                    parameter_names, input_names, output_names, wire_names, reg_names = collect_names(ast)
                    parameter_usage, input_usage, output_usage, wire_usage, reg_usage = count_identifier_usages(ast, parameter_names, input_names, output_names, wire_names, reg_names)
                    
                    # Target a random depth to limit the scope of changes
                    random_depth = random.randint(0, max_depth)
                    visit_node_delete(ast, random_depth, high_prob, high_prob, parameter_usage, input_usage, output_usage, wire_usage, reg_usage)
                    
                    # Cleanup logic related to the deletion
                    missing_parameter, missing_input, missing_output, missing_wire, missing_reg = find_missing_elements(parameter_usage, input_usage, output_usage, wire_usage, reg_usage)
                    for i in range(0, max_depth + 1):
                        find_and_remove_parameters(ast, i, missing_parameter, missing_input, missing_output, missing_wire, missing_reg)

                # --- 2. Change Logic ---
                if do_change:
                    random_depth = random.randint(0, max_depth)
                    visit_node_change(ast, random_depth, high_prob)

                # --- 3. Insertion Logic ---
                if do_insert:
                    random_depth = random.randint(0, max_depth)
                    visit_node_insertion(ast, random_depth, high_prob)

                # Generate the new code from the modified AST
                new_code = codegen.visit(ast)

                # Add the new code only if it's different and has not been generated before
                if new_code not in generated_codes:
                    generated_codes.add(new_code)
                    # Create new name for the code variant ({original_filename}_{variant_type}_{variant_number}.{original_extension})
                    variant_type = f"{'Del' if do_delete else ''}{'Chg' if do_change else ''}{'Ins' if do_insert else ''}"
                    new_code_filename =  f"{os.path.splitext(os.path.basename(code_path))[0]}_{variant_type}_{created_count}{os.path.splitext(code_path)[1]}"
                    codes.append((new_code_filename, new_code))
                    created_count += 1
        
        total_variants = len(variant_configs) * VERSIONS_PER_TYPE
        if len(codes) < total_variants:
            errors.append(f"Warning: Could not generate all {total_variants} variants for {code_path}. "
                          f"Successfully generated {len(codes)} unique variants.")

    except Exception as e:
        errors.append(f"Fatal error processing file {code_path}: {str(e)}")

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

def worker_wrapper(code_info):
    """
    A wrapper around augment_code to associate its output with its input.
    """
    # Call the original function
    codes_result, errors_result = augment_code(code_info)
    # Return the original input path along with the results
    input_path = code_info[0]
    return input_path, codes_result, errors_result


def augment_custom(prob, num_aug, data_dir, exp_dir, augment_source='code', max_retries=20):
    """
    Augments Verilog files with a retry mechanism for failed instances.

    Args:
        prob (float): The probability of applying an augmentation.
        num_aug (int): The number of augmented versions to generate per file.
        data_dir (str): The directory containing the source code.
        exp_dir (str): The directory to save experiment results and logs.
        augment_source (str, optional): The sub-directory containing .v files. Defaults to 'code'.
        max_retries (int, optional): The maximum number of times to retry failed files. Defaults to 3.
    """
    print(f'[AUG_CUSTOM]: Augmenting Verilog files in {data_dir} with prob={prob}, num_aug={num_aug}, and max_retries={max_retries}.')


    code_paths = sorted([os.path.join(f'{data_dir}/{augment_source}', f) for f in os.listdir(f'{data_dir}/{augment_source}') if f.endswith('.v')])
    
    # List of files that need to be processed. Initially, it's all of them.
    files_to_process = [(path, prob, num_aug) for path in code_paths]
    
    successful_results = []
    permanent_errors = []
    
    retry_attempt = 0
    while files_to_process and retry_attempt <= max_retries:
        if retry_attempt > 0:
            print(f"\n[AUG_CUSTOM]: Retrying {len(files_to_process)} failed files... (Attempt {retry_attempt}/{max_retries})")
            #time.sleep(2) # Small delay before retrying

        failed_in_this_run = []
        
        with Pool(processes=64) as pool:
            # Use tqdm to show progress for the current batch
            progress_bar = tqdm(pool.imap_unordered(worker_wrapper, files_to_process), total=len(files_to_process), desc=f"Attempt {retry_attempt}")
            
            for code_path, result, error in progress_bar:
                if error:
                    # Find the original code_info to add it for the next retry attempt
                    original_info = next((item for item in files_to_process if item[0] == code_path), None)
                    if original_info:
                        failed_in_this_run.append((original_info, error))
                else:
                    successful_results.extend(result)
        
        # Prepare for the next iteration
        if failed_in_this_run:
            # Update the list of files to process for the next retry
            files_to_process = [info for info, err in failed_in_this_run]
            
            # If this is the last attempt, move remaining failures to permanent errors
            if retry_attempt == max_retries:
                for info, err_list in failed_in_this_run:
                    permanent_errors.extend(err_list)
                print(f"\n[AUG_CUSTOM]: Max retries reached. {len(permanent_errors)} files failed permanently.")
                break # Exit the while loop
        else:
            # If nothing failed, we are done.
            print("\n[AUG_CUSTOM]: All files processed successfully!")
            files_to_process = [] # Empty the list to exit the while loop

        retry_attempt += 1

    # --- Save results ---
    output_dir = os.path.join(exp_dir, f'{augment_source}_augmentation')
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n[AUG_CUSTOM]: Writing {len(successful_results)} successful augmentations to CSV...")
    augmented_df = pd.DataFrame(successful_results, columns=['source', 'code'])
    augmented_df = augmented_df.sort_values(by='source').reset_index(drop=True)

    output_file_path = os.path.join(output_dir, 'results.csv')
    augmented_df.to_csv(output_file_path, index=False)
    print(f'[AUG_CUSTOM]: Augmentation results saved to: {output_file_path}')

    # --- Save permanent errors if any ---
    if permanent_errors:
        error_log_path = os.path.join(output_dir, 'errors.log')
        print(f'[AUG_CUSTOM]: Saving {len(permanent_errors)} permanent errors to: {error_log_path}')
        with open(error_log_path, 'w') as log_file:
            for error in permanent_errors:
                log_file.write(error + "\n")


def worker_wrapper_variants(code_info):
    """
    A wrapper around augment_code to associate its output with its input.
    """
    # Call the original function
    codes_result, errors_result = augment_code_variants(code_info)
    # Return the original input path along with the results
    input_path = code_info[0] # input_path string
    return input_path, codes_result, errors_result


def augment_custom_variants(num_aug_per_variant, data_dir, exp_dir, augment_source='code', max_retries=20):
    """
    Augments Verilog files with a retry mechanism for failed instances.
    Uses a controlled set of variants for each file.
    This function systematically creates variants with specific combinations of deletions, changes, and insertions in the AST.

    Args:
        num_aug_per_variant (int): The number of augmented versions to generate per variant.
        data_dir (str): The directory containing the source code.
        exp_dir (str): The directory to save experiment results and logs.
        augment_source (str, optional): The sub-directory containing .v files. Defaults to 'code'.
        max_retries (int, optional): The maximum number of times to retry failed files. Defaults to 3.
    """
    print(f'[AUG_CUSTOM_VARIANTS]: Augmenting Verilog files in {data_dir} with num_aug_per_variant={num_aug_per_variant}, and max_retries={max_retries}.')


    code_paths = sorted([os.path.join(f'{data_dir}/{augment_source}', f) for f in os.listdir(f'{data_dir}/{augment_source}') if f.endswith('.v')])

    NUM_VARIANTS_PER_TYPE = num_aug_per_variant  # Number of variants to generate for each type of modification
    
    # List of files that need to be processed. Initially, it's all of them.
    files_to_process = [(path, 1.0, NUM_VARIANTS_PER_TYPE) for path in code_paths]
    
    successful_results = []
    permanent_errors = []
    
    retry_attempt = 0
    while files_to_process and retry_attempt <= max_retries:
        if retry_attempt > 0:
            print(f"\n[AUG_CUSTOM_VARIANTS]: Retrying {len(files_to_process)} failed files... (Attempt {retry_attempt}/{max_retries})")
            #time.sleep(2) # Small delay before retrying

        failed_in_this_run = []
        
        with Pool(processes=64) as pool:
            # Use tqdm to show progress for the current batch
            progress_bar = tqdm(pool.imap_unordered(worker_wrapper_variants, files_to_process), total=len(files_to_process), desc=f"Attempt {retry_attempt}")
            
            for code_path, result, error in progress_bar:
                if error:
                    # Find the original code_info to add it for the next retry attempt
                    original_info = next((item for item in files_to_process if item[0] == code_path), None)
                    if original_info:
                        failed_in_this_run.append((original_info, error))
                else:
                    successful_results.extend(result)
        
        # Prepare for the next iteration
        if failed_in_this_run:
            # Update the list of files to process for the next retry
            files_to_process = [info for info, err in failed_in_this_run]
            
            # If this is the last attempt, move remaining failures to permanent errors
            if retry_attempt == max_retries:
                for info, err_list in failed_in_this_run:
                    permanent_errors.extend(err_list)
                print(f"\n[AUG_CUSTOM_VARIANTS]: Max retries reached. {len(permanent_errors)} files failed permanently.")
                break # Exit the while loop
        else:
            # If nothing failed, we are done.
            print("\n[AUG_CUSTOM_VARIANTS]: All files processed successfully!")
            files_to_process = [] # Empty the list to exit the while loop

        retry_attempt += 1

    # --- Save results ---
    output_dir = os.path.join(exp_dir, f'{augment_source}_augmentation_variants')
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n[AUG_CUSTOM_VARIANTS]: Writing {len(successful_results)} successful augmentations to CSV...")
    augmented_df = pd.DataFrame(successful_results, columns=['source', 'code'])
    augmented_df = augmented_df.sort_values(by='source').reset_index(drop=True)

    output_file_path = os.path.join(output_dir, 'results.csv')
    augmented_df.to_csv(output_file_path, index=False)
    print(f'[AUG_CUSTOM_VARIANTS]: Augmentation results saved to: {output_file_path}')

    # --- Save permanent errors if any ---
    if permanent_errors:
        error_log_path = os.path.join(output_dir, 'errors.log')
        print(f'[AUG_CUSTOM_VARIANTS]: Saving {len(permanent_errors)} permanent errors to: {error_log_path}')
        with open(error_log_path, 'w') as log_file:
            for error in permanent_errors:
                log_file.write(error + "\n")
    # --- Save the variants to a separate directory ---
    # With each original file haveing its own directory containing the variants.
    print(f'[AUG_CUSTOM_VARIANTS]: Saving variants to {output_dir}/variants')
    variants_dir = os.path.join(output_dir, 'variants')
    os.makedirs(variants_dir, exist_ok=True)
    # The source has format of {original_filename}_{variant_type}_{variant_number}.{original_extension}
    # where variant_type is a combination of Del, Chg, Ins for deletion, change, and insertion respectively.
    # There are 7 types of variants and each type has a number of variants defined by NUM_VARIANTS_PER_TYPE.
    # Create directories for each original file and save the variants in them.

    # Split the string of the file names to extract the original file name and extension
    for source, code in successful_results:
        # Extract the original file name and extension
        source_split = source.rsplit('_', 2)
        original_file_name = source_split[0]  # The part before the last two underscores
        original_extension = (source_split[-1]).split('.')[-1]
        # Create the directory for the original file
        original_dir = os.path.join(variants_dir, original_file_name)
        os.makedirs(original_dir, exist_ok=True)
        # Save the code in the original directory
        with open(os.path.join(original_dir, source), 'w') as f:
            f.write(code)
    print(f'[AUG_CUSTOM_VARIANTS]: Variants saved to: {variants_dir}')
    # --- End of saving variants ---
    print(f'[AUG_CUSTOM_VARIANTS]: Augmentation complete. {len(successful_results)} variants generated, {len(permanent_errors)} errors encountered.')