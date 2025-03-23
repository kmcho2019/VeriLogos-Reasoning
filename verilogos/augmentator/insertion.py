import os
import random
import multiprocessing
from pyverilog.vparser.parser import VerilogCodeParser
from pyverilog.ast_code_generator.codegen import ASTCodeGenerator
from pyverilog.vparser.ast import *
from collections import deque

bo_list = (Times, Divide, Mod, Plus, Minus)  # basic operator 
so_list = (Sll, Srl, Sla, Sra)  # shift operator
co_list = (LessThan, GreaterThan, LessEq, GreaterEq)  # comparison operator
eqo_list = (Eq, NotEq, Eql, NotEql)  # equality operator
go_list = (And, Xor, Xnor, Or, Land, Lor)  # gate operator
unary_list = (Uplus, Uminus, Ulnot, Unot, Uand, Unand, Uor, Unor, Uxor, Uxnor)  # unary operator
sub_list = (Substitution, BlockingSubstitution, NonblockingSubstitution)  # substitution operator
case_list = (CaseStatement, CasexStatement)  

unary_ops = unary_list
binary_ops = bo_list + so_list + co_list + eqo_list + go_list
identifier_counter = 0

class InsertException(Exception):
    pass

def binary_to_int(binary_str):
    if "b" in binary_str:
        binary_value = binary_str.split("b")[1]
        return int(binary_value, 2)  
    else:
        raise InsertException(f"Invalid binary string: {binary_str}")

def calc_operator(node, moduledef):
    if isinstance(node, Operator):
        if node.__class__.__name__ == 'Cond':
            condition_value = calc_operator(node.cond, moduledef)
            true_value = calc_operator(node.true_value, moduledef)
            false_value = calc_operator(node.false_value, moduledef)
            return true_value if condition_value else false_value
        
        left_value = calc_operator(node.left, moduledef) if hasattr(node, 'left') else None
        right_value = calc_operator(node.right, moduledef) if hasattr(node, 'right') else None
        
        operator_map = {
            'Times': lambda: left_value * right_value,
            'Divide': lambda: left_value // right_value,
            'Mod': lambda: left_value % right_value,
            'Plus': lambda: left_value + right_value,
            'Minus': lambda: left_value - right_value,
            'LessThan': lambda: left_value < right_value,
            'GreaterThan': lambda: left_value > right_value,
            'LessEq': lambda: left_value <= right_value,
            'GreaterEq': lambda: left_value >= right_value,
            'Eq': lambda: left_value == right_value,
            'NotEq': lambda: left_value != right_value,
            'And': lambda: left_value & right_value,
            'Or': lambda: left_value | right_value,
            'Xor': lambda: left_value ^ right_value,
            'Xnor': lambda: ~(left_value ^ right_value),
            'Land': lambda: left_value and right_value,
            'Lor': lambda: left_value or right_value,
            'Uplus': lambda: +left_value,
            'Uminus': lambda: -left_value,
            'Ulnot': lambda: not left_value,
            'Unot': lambda: ~left_value,
            'Uand': lambda: int(all(int(bit) for bit in bin(left_value)[2:])),
            'Unand': lambda: int(not all(int(bit) for bit in bin(left_value)[2:])),
            'Uor': lambda: int(any(int(bit) for bit in bin(left_value)[2:])),
            'Unor': lambda: int(not any(int(bit) for bit in bin(left_value)[2:])),
            'Uxor': lambda: int(bin(left_value).count('1') % 2),
            'Uxnor': lambda: int(bin(left_value).count('1') % 2 == 0)
        }
        
        if node.__class__.__name__ in operator_map:
            return operator_map[node.__class__.__name__]()
        else:
            raise InsertException(f"Unknown operator {node.__class__.__name__}")

    elif isinstance(node, IntConst):
        return int(node.value)
    
    elif isinstance(node, Identifier):
        param = get_param(node.name, moduledef)
        if param is None:
            raise InsertException(f"Parameter {node.name} not found in module definition")
        return binary_to_int(param) if 'b' in str(param) else int(param)
    
    else:
        raise InsertException(f"Unknown node type {node.__class__.__name__}")

def is_function(var, moduledef):
    if not hasattr(var, 'name'):
        return False
    
    for item in moduledef.items:
        if isinstance(item, Function):
            if item.name == var.name:
                return True
            
    return False

def is_child_of_generate_statement(node, moduledef):
    def traverse(node):
        if isinstance(node, GenerateStatement):
            if node_contains(node, target_node):
                return True
        if hasattr(node, 'children'):
            for child in node.children():
                if traverse(child):
                    return True
        return False

    def node_contains(parent, target):
        if parent == target:
            return True
        if hasattr(parent, 'children'):
            for child in parent.children():
                if node_contains(child, target):
                    return True
        return False

    target_node = node
    for item in moduledef.items:
        if traverse(item):
            return True
    return False

def get_max_depth(node):
    if not hasattr(node, 'children') or not node.children():
        return 1
    return max(get_max_depth(child) for child in node.children()) + 1

def get_param(name, moduledef):
    for param in moduledef.paramlist.params:
        for decl_child in param.children():
            if isinstance(decl_child, Parameter):
                if decl_child.name == name:
                    if isinstance(decl_child.value.var, Operator):
                        return calc_operator(decl_child.value.var, moduledef)
                    elif isinstance(decl_child.value.var, IntConst):
                        return decl_child.value.var.value
                    else:
                        raise InsertException("Wrong type of parameter.")

    for item in moduledef.items:
        if isinstance(item, Decl):
            for decl_child in item.children():
                if isinstance(decl_child, Parameter):     
                    if decl_child.name == name:
                        if isinstance(decl_child.value.var, Operator):
                            return calc_operator(decl_child.value.var, moduledef)
                        elif isinstance(decl_child.value.var, IntConst):
                            return decl_child.value.var.value
                        else:
                            raise InsertException("Wrong type of parameter.")
                        
def get_bitwidth(lvalue_node, moduledef):
    def calculate_width(child):
        if child.width:
            if isinstance(child.width.msb, Operator):
                msb = calc_operator(child.width.msb, moduledef)
            elif isinstance(child.width.msb, Identifier):
                msb = int(get_param(child.width.msb.name, moduledef))
            else:
                msb = int(child.width.msb.value)

            if isinstance(child.width.lsb, Operator):
                raise ValueError("LSB is an operator.")
            elif isinstance(child.width.lsb, Identifier):
                lsb = int(get_param(child.width.lsb.name, moduledef))
            else: 
                lsb = int(child.width.lsb.value)

            return abs(msb - lsb) + 1, child.width
        else:
            return 1, None

    for item in moduledef.items:
        if isinstance(item, Decl):
            for child in item.children():
                if child.name == lvalue_node.var.name:
                    return calculate_width(child)

    for port in moduledef.portlist.children():
        ioport = port.children()
        if len(ioport) == 0:
            return 1, None
        else:
            if ioport[0].name == lvalue_node.var.name:
                return calculate_width(ioport[0])

    return 1, None

def insert_node(node, moduledef_list):
    global identifier_counter

    try:
        if len(moduledef_list) > 1:
            raise InsertException("Multiple ModuleDefs found.")
        else :
            moduledef = moduledef_list[0]

        if is_child_of_generate_statement(node, moduledef):
            raise InsertException("Node is child of GenerateStatement.")

        has_lvalue = any(isinstance(child, Lvalue) for child in node.children())
        has_rvalue = any(isinstance(child, Rvalue) for child in node.children())

        if has_lvalue and has_rvalue:
            lvalue_node, rvalue_node = None, None
            for sibling in node.children():
                if isinstance(sibling, Lvalue):
                    lvalue_node = sibling
                elif isinstance(sibling, Rvalue):
                    rvalue_node = sibling

            if is_function(lvalue_node.var, moduledef):
                raise InsertException("Lvalue is a function.")
            
            if not isinstance(lvalue_node.var, Identifier):
                raise InsertException("Lvalue is not an Identifier.")

            for rvalue_child in rvalue_node.children():
                if isinstance(rvalue_child, IntConst):
                    if random.random() < 0.5:
                        bitwidth, _ = get_bitwidth(lvalue_node, moduledef)
                        new_value = random.randint(0, 2**bitwidth - 1)
                        rvalue_child.value = str(new_value)
                        print(f"Changed IntConst to random value: {new_value}")
                    else:
                        _, width = get_bitwidth(lvalue_node, moduledef)
                        if random.random() < len(unary_ops) / (len(unary_ops) + len(binary_ops)):
                            replace_with_unary_op(rvalue_node, moduledef, width)
                        else:
                            replace_with_binary_op(rvalue_node, moduledef, width, use_existing=False)  
                elif isinstance(rvalue_child, Identifier):
                    _, width = get_bitwidth(lvalue_node, moduledef)
                    if random.random() < len(unary_ops) / (len(unary_ops) + len(binary_ops)):
                        unary_op = random.choice(unary_ops)
                        new_unary = unary_op(right=rvalue_child)
                        rvalue_node.var = new_unary
                        print(f"Changed identifier to unary operator: {unary_op}")
                    else:
                        replace_with_binary_op(rvalue_node, moduledef, width, use_existing=True, existing_identifier=rvalue_child)  
                else:
                    raise InsertException("Unexpected rvalue type.")

    except InsertException as e:
        print(f"Unexpected error during insert_node: {e}")
        return 


def replace_with_unary_op(rvalue_node, moduledef, width):
    global identifier_counter

    new_name = f'aug_w_{identifier_counter}'

    new_identifier = Identifier(name=new_name)
    identifier_counter += 1

    new_input = Input(name=new_name, width=width)
    new_wire = Wire(name=new_name, width=width)
    new_ioport = Ioport(first=new_input, second=new_wire)
    moduledef.portlist.ports += (new_ioport,)

    unary_op = random.choice(unary_ops)
    new_unary = unary_op(right=new_identifier)
    rvalue_node.var = new_unary

    print(f"Changed IntConst to unary operator: {unary_op}")

def replace_with_binary_op(rvalue_node, moduledef, width, use_existing=False, existing_identifier=None):
    global identifier_counter
    
    if use_existing and existing_identifier is not None:
        new_identifier_1 = existing_identifier
    else:
        new_name_1 = f'aug_w_{identifier_counter}'
        
        new_identifier_1 = Identifier(name=new_name_1)
        identifier_counter += 1
        
        new_input_1 = Input(name=new_name_1, width=width)
        new_wire_1 = Wire(name=new_name_1, width=width)
        new_ioport_1 = Ioport(first=new_input_1, second=new_wire_1)
        moduledef.portlist.ports += (new_ioport_1,)
    
    new_name_2 = f'aug_w_{identifier_counter}'
    
    new_identifier_2 = Identifier(name=new_name_2)
    identifier_counter += 1

    new_input_2 = Input(name=new_name_2, width=width)
    new_wire_2 = Wire(name=new_name_2, width=width)
    new_ioport_2 = Ioport(first=new_input_2, second=new_wire_2)
    moduledef.portlist.ports += (new_ioport_2,)
    
    binary_op = random.choice(binary_ops)
    new_binary = binary_op(left=new_identifier_1, right=new_identifier_2)
    rvalue_node.var = new_binary

    if use_existing and existing_identifier is not None:
        print(f"Changed identifier to binary operator: {binary_op} using existing Identifier.")
    else:
        print(f"Changed IntConst to binary operator: {binary_op}")

def visit_node_insertion(root_node, target_depth, prob=0.5):    
    queue = deque([(root_node, 0)])

    moduledef_list = [] 
    for grand_child in root_node.children()[0].children():
        if isinstance(grand_child, ModuleDef):
            moduledef_list.append(grand_child)
    if len(moduledef_list) == 0  :
        raise ValueError("ModuleDef not found.")

    while queue:
        node, current_depth = queue.popleft()

        if current_depth == target_depth:
            insert_node(node, moduledef_list)

        elif hasattr(node, 'children'):
            for child in node.children():
                queue.append((child, current_depth + 1))

    return "Completed processing at target depth."

def process_verilog_file(file_path, save_dir, prob=0.5):
    codeparser = VerilogCodeParser([file_path],
                                   preprocess_output=os.path.join('.', os.path.basename(file_path)))
    ast = codeparser.parse()
    depth = get_max_depth(ast)

    for i in range(0, depth+1):
        visit_node_insertion(ast, i, prob)

    codegen = ASTCodeGenerator()
    generated_code = codegen.visit(ast)
    file_name = os.path.basename(file_path)
    output_file_path = os.path.join(save_dir, file_name)

    with open(output_file_path, 'w') as f:
        f.write(generated_code)

    print(f"Processed file saved to: {output_file_path}")

def process_all_verilog_files_in_folder(input_dir, save_dir, prob=0.5, multiprocess=True):
    verilog_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.v')]
    verilog_files.sort()
    
    if multiprocess:
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            pool.starmap(process_verilog_file, [(file_path, save_dir, prob) for file_path in verilog_files])
    else: 
        for file_path in verilog_files:
            file_name = os.path.basename(file_path)
            output_file_path = os.path.join(save_dir, file_name)
            if os.path.exists(output_file_path): continue
            
            process_verilog_file(file_path, save_dir, prob)

