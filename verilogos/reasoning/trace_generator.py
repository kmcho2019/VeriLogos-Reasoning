import os
import json
import glob
from tqdm import tqdm

from verilogos.reasoning.api_client import R1APIClient
from verilogos.generator.jsonl_generator import blank_logic, blank_module # For masking
# For verification - you'll need to adapt or use parts of hdl_evaluator
from verilogos.utils.syntax import check_syntax
from verilogos.utils.functionality import check_functionality, gen_fm_script # Assuming gen_fm_script is useful directly
from verilogos.utils.parser import parse_fm, parse_module_name, parse_module_decl_cont
from verilogos.utils.status import Status


def mask_verilog_code(filepath: str, method: str = "logic"):
    """
    Loads Verilog code from a file and applies a masking strategy.
    Returns the original code and the masked code.
    """
    with open(filepath, 'r') as f:
        original_code = f.read()

    masked_code = None
    target_identifier = None # For 'logic' method

    if method == "logic":
        # blank_logic expects a file path. For simplicity, we pass it.
        # It also writes a preprocessed file, which might need cleanup or redirection.
        # Consider adapting blank_logic to accept code as a string if needed.
        # For now, assuming blank_logic can work with the filepath directly
        # and we handle the temp file it creates or it's benign.
        # It returns (masked_code_str, target_identifier_str)
        # The current jsonl_generator.blank_logic uses a hardcoded path for preprocess_output.
        # This might need to be made configurable or handled.
        # For now, we are calling it with the original filepath, it might create a temp file.

        # Simplified: directly call blank_logic.
        # This part needs careful integration with how blank_logic works regarding file I/O
        # and temporary file creation.
        # For now, let's assume a conceptual call:
        # masked_code, target_identifier = blank_logic(filepath) # This is a simplification

        # More robust: Re-implement or adapt masking here if blank_logic is too tied to jsonl_generator's specific file handling
        # For now, we'll use a placeholder for masking.
        # You should integrate the AST-based masking from jsonl_generator.py here.
        # For example, you might need to call pyverilog parser, then CustomASTCodeGenerator

        # Placeholder masking:
        module_decl, _ = parse_module_decl_cont(original_code)
        if module_decl:
             masked_code = module_decl + "\n  // Logic related to outputs is [BLANK]\nendmodule"
        else:
            masked_code = original_code.replace("always @", "// [BLANK] always @") # very naive
        print(f"Applied placeholder masking to {filepath}")

    elif method == "module":
        masked_code = blank_module(original_code)
    else:
        raise ValueError(f"Unsupported masking method: {method}")

    if masked_code is None:
        print(f"Warning: Masking failed for {filepath}. Skipping.")
        return original_code, None

    return original_code, masked_code

def verify_synthetic_trace(generated_verilog: str, original_verilog: str, module_name_override: str, work_dir_base: str, data_dir_for_ref: str) -> bool:
    """
    Verifies the R1-generated Verilog against the original using existing utilities.
    Returns True if functionally equivalent, False otherwise.
    """
    # Create a unique working directory for this verification
    # Ensure module_name_override is filesystem-safe
    safe_module_name = "".join(c if c.isalnum() else "_" for c in module_name_override)
    work_dir = os.path.join(work_dir_base, f"verify_{safe_module_name}_{os.urandom(4).hex()}")
    os.makedirs(work_dir, exist_ok=True)

    gen_rtl_path = os.path.join(work_dir, f"{safe_module_name}_gen.v")
    gol_rtl_path = os.path.join(work_dir, f"{safe_module_name}_gol.v") # Changed to work_dir to avoid multiple writes to data_dir

    with open(gen_rtl_path, 'w') as f:
        # The R1 output might contain explanations before/after the code block.
        # We need to extract just the Verilog code.
        # This is a common pattern, but the exact extraction depends on R1 API output format.
        match = parse_module_decl_cont(generated_verilog) # Attempt to get module ... endmodule
        if match and match[0] and match[1]:
             code_to_write = match[0] + match[1] # if module decl and content are separate
        else: # Fallback: try to find ```verilog ... ```
            code_match = re.search(r"```verilog\s*([\s\S]*?)\s*```", generated_verilog, re.DOTALL)
            if code_match:
                code_to_write = code_match.group(1).strip()
            else: # If no block, assume the whole thing is code, or it's bad
                code_to_write = generated_verilog.strip() 
        f.write(code_to_write)

    with open(gol_rtl_path, 'w') as f:
        f.write(original_verilog)

    # 1. Syntax Check
    if check_syntax(gen_rtl_path) == Status.FAIL:
        print(f"Syntax check FAIL for {gen_rtl_path}")
        # shutil.rmtree(work_dir) # Optional: cleanup
        return False
    print(f"Syntax check PASS for {gen_rtl_path}")

    # 2. Functionality Check
    # The check_functionality needs the true module name.
    # We parse it from the golden RTL.
    parsed_module_name = parse_module_name(original_verilog)
    if not parsed_module_name:
        print(f"Could not parse module name from golden RTL: {gol_rtl_path}")
        # shutil.rmtree(work_dir)
        return False

    # Note: check_functionality uses fm_shell. Ensure it's in PATH and license is available.
    # It also reads a reference fm_script from './ref/ref.fm'
    # Ensure this path is correct relative to where the script runs.
    if check_functionality(gen_rtl_path, gol_rtl_path, parsed_module_name, work_dir) == Status.FAIL:
        print(f"Functionality check FAIL (initial) for {gen_rtl_path} against {gol_rtl_path}")
        # shutil.rmtree(work_dir)
        return False

    fm_log_path = f'{work_dir}/{parsed_module_name}.fm.log'
    if not os.path.exists(fm_log_path):
        print(f"Functionality log not found: {fm_log_path}")
        # shutil.rmtree(work_dir) # Optional: cleanup
        return False

    equivalence_score = parse_fm(fm_log_path)
    if equivalence_score == 1.0:
        print(f"Functionality check PASS (equivalent) for {gen_rtl_path}")
        # shutil.rmtree(work_dir) # Optional: cleanup
        return True
    else:
        print(f"Functionality check FAIL (score: {equivalence_score}) for {gen_rtl_path}")
        # shutil.rmtree(work_dir) # Optional: cleanup
        return False

def generate_and_filter_traces(reference_code_dir: str, output_traces_file: str,
                               r1_api_provider: str, r1_api_key: str,
                               cache_dir: str, data_dir:str, exp_dir: str,
                               r1_model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"): # Or API specific name
    """
    Main function to generate synthetic traces, verify, and save them.
    """
    if not r1_api_provider or not r1_api_key:
        raise ValueError("R1 API provider and API key must be specified.")
    if not reference_code_dir:
        raise ValueError("Reference code directory must be specified.")

    api_client = R1APIClient(provider=r1_api_provider, api_key=r1_api_key)

    verilog_files = glob.glob(os.path.join(reference_code_dir, "*.v")) + \
                    glob.glob(os.path.join(reference_code_dir, "*.sv"))

    if not verilog_files:
        print(f"No Verilog files found in {reference_code_dir}")
        return

    print(f"Found {len(verilog_files)} Verilog files for trace generation.")

    filtered_traces = []
    # Base directory for verification temporary files
    verification_work_dir_base = os.path.join(exp_dir, "verification_traces")
    os.makedirs(verification_work_dir_base, exist_ok=True)

    for filepath in tqdm(verilog_files, desc="Generating and Filtering Traces"):
        module_file_name = os.path.basename(filepath)
        try:
            original_code, masked_code = mask_verilog_code(filepath, method="logic") # Or other methods
            if masked_code is None:
                continue

            # Generate trace using R1 API
            # Use the model name appropriate for the API provider (e.g., "deepseek-chat" for DeepSeek API)
            # For Together/Fireworks, it might be the HuggingFace name if they host it directly.
            # This should be configurable. For now, using the passed r1_model_name.
            r1_generated_output = api_client.generate_trace(masked_code, model_name=r1_model_name) 

            if not r1_generated_output:
                print(f"R1 API failed to generate output for {module_file_name}")
                continue

            # Verify the R1 generated output
            # The module name for verification should ideally be parsed from original_code.
            # Pass a unique name for the file being processed.
            if verify_synthetic_trace(r1_generated_output, original_code, 
                                      module_name_override=module_file_name.replace('.v',''), 
                                      work_dir_base=verification_work_dir_base,
                                      data_dir_for_ref=data_dir): # data_dir is where golden RTLs are normally found by evaluator

                # Extract just the code part from r1_generated_output for saving
                # This logic should be robust, similar to verify_synthetic_trace
                code_match = re.search(r"```verilog\s*([\s\S]*?)\s*```", r1_generated_output, re.DOTALL)
                if code_match:
                    extracted_completion = code_match.group(1).strip()
                else: # Fallback if no ```verilog block
                    # This might indicate a poor generation or a need for better extraction
                    # For now, let's assume we need to be more careful or the API provides cleaner output
                    # A simple heuristic: take text after a common phrase like "Here is the code:"
                    # Or, if verify_synthetic_trace already writes the clean version, we could re-read that.
                    # For simplicity here, let's assume r1_generated_output *is* the code, or needs simple cleaning.
                    # It's better if verify_synthetic_trace returns the cleaned code on success.
                    extracted_completion = r1_generated_output.strip() # Needs improvement

                trace_data = {
                    "reference_file": module_file_name,
                    "masked_code": masked_code,
                    "original_code": original_code, # For reference, might remove for final dataset
                    "r1_raw_output": r1_generated_output, # Full API output for debugging
                    "verified_completion": extracted_completion, # The part to use for fine-tuning
                    # Format for SFT:
                    "messages": [
                        {"role": "system", "content": "You are a Verilog expert. Complete the given masked Verilog code."},
                        {"role": "user", "content": f"Complete the [BLANK] sections in the following Verilog code:\n\n```verilog\n{masked_code}\n```"},
                        {"role": "assistant", "content": f"```verilog\n{extracted_completion}\n```"}
                    ]
                }
                filtered_traces.append(trace_data)
                print(f"Verified and added trace for {module_file_name}")
            else:
                print(f"Verification failed for R1 output of {module_file_name}")

        except Exception as e:
            print(f"Error processing {module_file_name}: {e}")
            import traceback
            traceback.print_exc()

    # Save the filtered traces
    with open(output_traces_file, 'w') as f:
        for trace in filtered_traces:
            f.write(json.dumps(trace) + "\n")
    print(f"Saved {len(filtered_traces)} filtered synthetic traces to {output_traces_file}")

# Add a basic main for testing if needed
if __name__ == '__main__':
    # This is an example. You'll call this from main.py
    # Create dummy reference files first in './data/reference_verilog/'
    # e.g., ./data/reference_verilog/adder.v
    # module adder(input a, input b, output s);
    # assign s = a ^ b;
    # endmodule

    # os.makedirs('./data/reference_verilog', exist_ok=True)
    # with open('./data/reference_verilog/test_module1.v', 'w') as f:
    #    f.write("module test_module1(input clk, input rst, output reg out_val);\n initial begin out_val = 0; end \n always @(posedge clk) begin if(rst) out_val <= 1'b0; else out_val <= !out_val; end \nendmodule")

    # print("Starting trace generation test...")
    # generate_and_filter_traces(
    #     reference_code_dir='./data/reference_verilog/', # Create this dir and add some .v files
    #     output_traces_file='./exp/synthetic_traces/test_filtered_traces.jsonl',
    #     r1_api_provider='deepseek_api',  # or 'together', 'fireworks'
    #     r1_api_key='YOUR_API_KEY_HERE', # Replace with your actual key
    #     cache_dir='./cache',
    #     data_dir='./data',
    #     exp_dir='./exp',
    #     r1_model_name='deepseek-chat' # for deepseek_api; for others, check their model names
    # )
    pass