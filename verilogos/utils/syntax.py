import subprocess as sp
from verilogos.utils.status import Status

def check_syntax(gen_rtl):
    command = f'iverilog {gen_rtl}'
    process = sp.run(command, shell=True, stdout=sp.PIPE, stderr=sp.PIPE)

    if process.returncode == 0:
        return Status.PASS
    else:
        return Status.FAIL