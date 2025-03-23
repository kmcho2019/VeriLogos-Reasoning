import os
import subprocess as sp
from verilogos.utils.status import Status

def check_functionality(gen_rtl, gol_rtl, module, work_dir):
    fm_script = gen_fm_script(gen_rtl, gol_rtl, module, work_dir)
    fm_log = f'{work_dir}/{module}.fm.log'

    command = f'fm_shell -work_path {work_dir} -file {fm_script} | tee {fm_log}'
    process = sp.run(command, shell=True, stdout=sp.PIPE, stderr=sp.PIPE)

    if process.returncode == 0 :
        return Status.PASS
    else:
        return Status.FAIL

def gen_fm_script(gen_rtl, gol_rtl, module, work_dir):
    fm_ref = "./ref/ref.fm"
    fm_gen = f'{work_dir}/{module}.fm'

    with open(fm_ref, 'r') as infile:
        with open(fm_gen, 'w') as outfile:
            text = infile.read()
            text = text.replace("__GOLD_RTL__", os.path.realpath(gol_rtl))
            text = text.replace("__GATE_RTL__", os.path.realpath(gen_rtl))
            text = text.replace("__MODULE_NAME__", module)
            outfile.write(text)

    return fm_gen