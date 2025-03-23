import re

def parse_fm(fm_log):
    num_pass, num_fail = 0, 0

    with open(fm_log, 'r') as f:
        for line in f:
            if "Passing (equivalent)" in line:
                sp = line.split()
                num_pass = float(sp[6])
            elif "Failing (not equivalent)" in line:
                sp = line.split()
                num_fail = float(sp[7])

    if num_pass == 0 and num_fail == 0:
        score = 0
    else:
        score = num_pass / (num_pass + num_fail)

    return score

def parse_module_decl_cont(verilog_code):
    module_pattern = re.compile(r'(module\s+.*?);(.*?)(endmodule)', re.DOTALL)
    match = module_pattern.search(verilog_code)
    
    if match:
        module_declaration = match.group(1) + ';'
        module_content = match.group(2) + 'endmodule'
        return module_declaration, module_content
    else:
        return None, None
  
def parse_module_name(verilog_code):
    match = re.search(r'\bmodule\s+(\w+)', verilog_code)
    if match:
        return match.group(1)
    return None