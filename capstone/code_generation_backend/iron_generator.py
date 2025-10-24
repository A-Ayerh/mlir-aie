import networkx as nx
from collections import namedtuple
import numpy as np
from ml_dtypes import bfloat16
import os

def generateIronCode(dataFlowGraph: nx.MultiDiGraph, filepath: str, external_functions: list = None, data_size: int = 128):
    external_functions = external_functions or []

    # Buffers for code generation sections
    imports_buffer = []
    types_buffer = []
    objectFifo_buffer = []
    split_join_buffer = []
    external_buffer = []
    corefn_buffer = []
    worker_buffer = []
    runtime_buffer = []
    main_buffer = []

    # Maps for tracking nodes and connections
    tile_map = {}           # Maps nodes to tile variables
    fifo_map = {}           # Maps edges to FIFO variables
    fifo_name_map = {}      # Maps FIFO variables to generated names
    split_outputs = {}      # Tracks split/join operations
    externalfn_instances = {}  # Tracks external function instances
    node_indices = {}       # Maps nodes to unique indices by type

    # Placement named tuple for tile coordinates
    Placement = namedtuple('Placement', ['column', 'row'])

    #----------Node Indexing---------------
    for node_type in ['shim', 'mem', 'compute']:
        nodes = [n for n, d in dataFlowGraph.nodes(data=True) if d.get('type') == node_type]
        for i, node in enumerate(nodes):
            node_indices[node] = f"{node_type}_{i}"

    #----------Input and Output Sources-------------
    inputs = set()
    outputs = set()
    for prod, cons, key in dataFlowGraph.edges(keys=True):
        data = dataFlowGraph.edges[prod, cons, key]
        name = data.get('name', '')
        source_part = name.split('_')[3][0] if len(name.split('_')) > 3 else ''
        if dataFlowGraph.nodes[prod].get('type') == 'shim':
            inputs.add(source_part)
        elif dataFlowGraph.nodes[cons].get('type') == 'shim':
            outputs.add(source_part)
    sorted_inputs = sorted(inputs)
    sorted_outputs = sorted(outputs)
    argument_names = [f"input{source}" for source in sorted_inputs] + [f"output{source}" for source in sorted_outputs]

    #----------Chunk Size Calculation---------------
    mem_nodes = [node for node in dataFlowGraph.nodes() if dataFlowGraph.nodes[node].get('type') == 'mem']
    num_mem_nodes = len(mem_nodes)
    chunk_size = data_size // num_mem_nodes
    worker_chunk_size = chunk_size // 2

    #----------Imports and Function Header----------
    imports_buffer = [
        "from aie.iron import Program, Runtime, Worker, ObjectFifo",
        "from aie.iron.placers import SequentialPlacer",
        "from aie.iron import ExternalFunction, jit",
        "from aie.iron.dataflow import ObjectFifoLink",
        "from aie.iron.device import Tile",
        "import numpy as np",
        "import aie.iron as iron",
        "from ml_dtypes import bfloat16",
        "from aie.helpers.taplib import TensorAccessPattern",
        "",
        "@iron.jit(is_placed=False)",
        f"def design({', '.join(argument_names)}):"
    ]

    #----------Data Type Definitions----------------
    types_buffer = [
        f"    data_a_ty = np.ndarray[(inputA.numel(),), np.dtype[bfloat16]]",
        f"    data_b_ty = np.ndarray[(inputB.numel(),), np.dtype[bfloat16]]",
        f"    data_d_ty = np.ndarray[(outputD.numel(),), np.dtype[bfloat16]]",
        "",
        f"    chunk_a = np.ndarray[({chunk_size},), np.dtype[bfloat16]]",
        f"    chunk_b = np.ndarray[({chunk_size},), np.dtype[bfloat16]]",
        f"    chunk_d = np.ndarray[({chunk_size},), np.dtype[bfloat16]]",
        "",
        f"    chunk_a_worker = np.ndarray[({worker_chunk_size},), np.dtype[bfloat16]]",
        f"    chunk_b_worker = np.ndarray[({worker_chunk_size},), np.dtype[bfloat16]]",
        f"    chunk_d_worker = np.ndarray[({worker_chunk_size},), np.dtype[bfloat16]]",
    ]

    #----------ObjectFifo Definitions (shim <-> mem)---------
    for prod, cons, key in dataFlowGraph.edges(keys=True):
        data = dataFlowGraph.edges[prod, cons, key]
        name = data.get('name', '')
        source_part = name.split('_')[3][0] if len(name.split('_')) > 3 else ''
        prod_idx = node_indices.get(prod, prod)
        cons_idx = node_indices.get(cons, cons)
        if dataFlowGraph.nodes[prod].get('type') == 'shim' and dataFlowGraph.nodes[cons].get('type') == 'mem':
            fifo_variable = f"fifo_{prod_idx}_to_{cons_idx}_{source_part}"
            fifo_map[(prod, cons, key)] = fifo_variable
            fifo_name_map[fifo_variable] = f"fifo_{prod_idx}_to_{cons_idx}_{source_part}"
            fifo_type = f"chunk_{source_part.lower()}"
            objectFifo_buffer.append(
                f"    {fifo_variable} = ObjectFifo(obj_type={fifo_type}, depth=2, name=\"{fifo_name_map[fifo_variable]}\")"
            )

    #----------Split Operations (mem -> compute)------------
    for node in mem_nodes:
        col_idx = dataFlowGraph.nodes[node]['placement'].column
        tile_var = f"Tile({col_idx}, 1)"
        tile_map[node] = tile_var
        in_edges = list(dataFlowGraph.in_edges(node, keys=True))
        for pred, _, key in in_edges:
            if dataFlowGraph.nodes[pred].get('type') == 'shim':
                fifo_var = fifo_map[(pred, node, key)]
                source_part = fifo_name_map[fifo_var].split('_')[-1]
                out_edges = [e for e in dataFlowGraph.out_edges(node, keys=True) if dataFlowGraph.edges[e]['name'].split('_')[3][0] == source_part]
                num_workers = len(out_edges)
                split_names = [f"split_{node_indices[node]}_{source_part}_{i}" for i in range(num_workers)]
                offsets = [worker_chunk_size * i for i in range(num_workers)]
                split_var = f"split_{node_indices[node]}_{source_part}"
                split_join_buffer.append(
                    f"    {split_var} = {fifo_var}.cons().split("
                    f"        obj_types=[chunk_{source_part.lower()}_worker,chunk_{source_part.lower()}_worker],"
                    f"        offsets={offsets},"
                    f"        names={split_names},"
                    f"        placement={tile_var}"
                    f"    )"
                )
                split_outputs[(node, source_part)] = {
                    'var': split_var,
                    'num_workers': num_workers,
                    'consumers': [(e[1], i) for i, e in enumerate(out_edges)]
                }

    #----------ObjectFifo Definitions (compute -> compute)---------
    for prod, cons, key in dataFlowGraph.edges(keys=True):
        data = dataFlowGraph.edges[prod, cons, key]
        if dataFlowGraph.nodes[prod].get('type') == 'compute' and dataFlowGraph.nodes[cons].get('type') == 'compute':
            prod_idx = node_indices.get(prod, prod)
            cons_idx = node_indices.get(cons, cons)
            fifo_variable = f"fifo_{prod_idx}_to_{cons_idx}"
            fifo_map[(prod, cons, key)] = fifo_variable
            fifo_name_map[fifo_variable] = fifo_variable
            objectFifo_buffer.append(
                f"    {fifo_variable} = ObjectFifo(obj_type=chunk_d_worker, depth=2, name=\"{fifo_variable}\")"
            )

    #----------ObjectFifo Definitions (mem -> shim)---------
    for prod, cons, key in dataFlowGraph.edges(keys=True):
        data = dataFlowGraph.edges[prod, cons, key]
        source_part = data.get('name', '').split('_')[3][0] if len(data.get('name', '').split('_')) > 3 else ''
        if dataFlowGraph.nodes[prod].get('type') == 'mem' and dataFlowGraph.nodes[cons].get('type') == 'shim':
            prod_idx = node_indices.get(prod, prod)
            cons_idx = node_indices.get(cons, cons)
            fifo_variable = f"fifo_{prod_idx}_to_{cons_idx}_{source_part}"
            fifo_map[(prod, cons, key)] = fifo_variable
            fifo_name_map[fifo_variable] = fifo_variable
            objectFifo_buffer.append(
                f"    {fifo_variable} = ObjectFifo(obj_type=chunk_d, depth=2, name=\"{fifo_variable}\")"
            )

    #----------Join Operations (compute -> mem)------------
    for node in mem_nodes:
        col_idx = dataFlowGraph.nodes[node]['placement'].column
        tile_var = f"Tile({col_idx}, 1)"
        out_edges = list(dataFlowGraph.out_edges(node, keys=True))
        for _, succ, key in out_edges:
            if dataFlowGraph.nodes[succ].get('type') == 'shim':
                fifo_var = fifo_map[(node, succ, key)]
                source_part = fifo_name_map[fifo_var].split('_')[-1]
                in_edges = [e for e in dataFlowGraph.in_edges(node, keys=True) if dataFlowGraph.edges[e]['name'].split('_')[3][0] == source_part]
                num_workers = len(in_edges)
                join_names = [f"join_{node_indices[node]}_{source_part}_{i}" for i in range(num_workers)]
                offsets = [worker_chunk_size * i for i in range(num_workers)]
                join_var = f"join_{node_indices[node]}_{source_part}"
                split_join_buffer.append(
                    f"    {join_var} = {fifo_var}.prod().join("
                    f"        obj_types=[chunk_d_worker,chunk_d_worker],"
                    f"        names={join_names},"
                    f"        placement={tile_var},"
                    f"        offsets={offsets},"
                    f"    )"
                )
                split_outputs[(node, source_part, 'join')] = {
                    'var': join_var,
                    'num_workers': num_workers,
                    'producers': [(e[0], i) for i, e in enumerate(in_edges)]
                }

    #----------External Function Definitions---------
    for fn in external_functions:
        external_variable = f"external_{fn['name'].replace('_', '')}"
        if fn['name'] == 'eltwise_add_bf16_scalar':
            arg_types = '[chunk_a_worker, chunk_b_worker, chunk_d_worker]'
            source_file = "/scratch/btsorens/mlir-aie/aie_kernels/aie2/add.cc"
        elif fn['name'] == 'bf16_relu':
            arg_types = '[chunk_d_worker, chunk_d_worker]'
            source_file = "/scratch/btsorens/mlir-aie/aie_kernels/aie2/relu.cc"
        else:
            arg_types = '[chunk_d_worker, chunk_d_worker]'
            source_file = fn['source_file']
        external_buffer.append(
            f"    {external_variable} = ExternalFunction("
            f"        name=\"{fn['name']}\","
            f"        source_file=\"{source_file}\","
            f"        arg_types={arg_types},"
            f"        include_dirs={fn['include_dirs']}"
            f"    )"
        )
        externalfn_instances[fn['name']] = external_variable

    #----------Core Function Definitions------------
    corefn_buffer = [
        "    # Define kernels here...",
        "    def eltwise_add(element_wise_add, inputA, inputB, outputC):",
        "        elementA = inputA.acquire(1)",
        "        elementB = inputB.acquire(1)",
        "        elementC = outputC.acquire(1)",
        "        element_wise_add(elementA, elementB, elementC)",
        "        inputA.release(1)",
        "        inputB.release(1)",
        "        outputC.release(1)",
        "",
        "    def relu(relu_activation, inputC, outputD):",
        "        elementC = inputC.acquire(1)",
        "        elementD = outputD.acquire(1)",
        "        relu_activation(elementC, elementD)",
        "        inputC.release(1)",
        "        outputD.release(1)",
    ]

    #----------Worker Definitions-------------------
    worker_buffer.append("    Workers = []")
    for node in dataFlowGraph.nodes():
        if dataFlowGraph.nodes[node].get('type') == 'compute':
            attributes = dataFlowGraph.nodes[node]
            core_fn = attributes.get('core_fn')
            col = attributes['placement'].column
            row = attributes['placement'].row
            worker_name = f"worker_{node_indices[node]}"
            in_edges = list(dataFlowGraph.in_edges(node, keys=True))
            out_edges = list(dataFlowGraph.out_edges(node, keys=True))

            if core_fn == 'eltwise_add_bf16_scalar':
                fn_name = "eltwise_add"
                fn_args = [externalfn_instances[core_fn]]
                for pred, _, key in in_edges:
                    if dataFlowGraph.nodes[pred].get('type') == 'mem':
                        source_part = dataFlowGraph.edges[pred, node, key]['name'].split('_')[3][0]
                        split_info = split_outputs.get((pred, source_part), {})
                        consumer_entry = next((e for e in split_info.get('consumers', []) if e[0] == node), None)
                        if consumer_entry:
                            fn_args.append(f"{split_info['var']}[{consumer_entry[1]}].cons()")
                fn_args.append(f"{fifo_map[out_edges[0]]}.prod()")
                worker_buffer.append(
                    f"    {worker_name} = Worker(core_fn={fn_name},fn_args=[{', '.join(fn_args)}],placement=Tile({col},{row}))"
                )
                worker_buffer.append(f"    Workers.append({worker_name})")
            elif core_fn == 'bf16_relu':
                fn_name = "relu"
                fn_args = [externalfn_instances[core_fn], f"{fifo_map[in_edges[0]]}.cons()"]
                for _, succ, key in out_edges:
                    if dataFlowGraph.nodes[succ].get('type') == 'mem':
                        source_part = dataFlowGraph.edges[node, succ, key]['name'].split('_')[3][0]
                        join_info = split_outputs.get((succ, source_part, 'join'), {})
                        producer_entry = next((e for e in join_info.get('producers', []) if e[0] == node), None)
                        if producer_entry:
                            fn_args.append(f"{join_info['var']}[{producer_entry[1]}].prod()")
                worker_buffer.append(
                    f"    {worker_name} = Worker(core_fn={fn_name},fn_args=[{', '.join(fn_args)}],placement=Tile({col},{row}))"
                )
                worker_buffer.append(f"    Workers.append({worker_name})")

    #----------Runtime Configuration----------------
    runtime_buffer = [
        "    # Define runtime here:",
        "    rt = Runtime()",
        f"    with rt.sequence(chunk_a, chunk_b, chunk_d) as (A, B, D):",
        "        rt.start(*Workers)"
    ]
    for col in range(num_mem_nodes):
        for source in sorted_inputs:
            fifo_name = next(
                f for (p, c, k), f in fifo_map.items()
                if node_indices.get(p, p) == f"shim_{col}"
                and node_indices.get(c, c) == f"mem_{col}"
                and fifo_name_map[f].endswith(f"_{source}")
            )
            runtime_buffer.append(
                f"        rt.fill(placement=Tile({col},0), in_fifo={fifo_name}.prod(), source={source}, tap=TensorAccessPattern(tensor_dims=[(input{source}.numel()),],offset=(((input{source}.numel())//{num_mem_nodes})*{col}), sizes=[((input{source}.numel())//{num_mem_nodes})//((input{source}.numel())//{num_mem_nodes*2}), ((input{source}.numel())//{num_mem_nodes*2})], strides=[((input{source}.numel())//{num_mem_nodes*2}), 1],))"
            )
        fifo_name = next(
            f for (p, c, k), f in fifo_map.items()
            if node_indices.get(p, p) == f"mem_{col}"
            and node_indices.get(c, c) == f"shim_{col}"
            and fifo_name_map[f].endswith('_D')
        )
        runtime_buffer.append(
            f"        rt.drain(placement=Tile({col},0), out_fifo={fifo_name}.cons(), dest=D, wait=True, tap=TensorAccessPattern(tensor_dims=[(outputD.numel()),],offset=(((outputD.numel())//{num_mem_nodes})*{col}), sizes=[((outputD.numel())//{num_mem_nodes})//((outputD.numel())//{num_mem_nodes*2}), ((outputD.numel())//{num_mem_nodes*2})], strides=[((outputD.numel())//{num_mem_nodes*2}), 1],))"
        )
    runtime_buffer.extend([
        "    my_program = Program(iron.get_current_device(), rt)",
        "    my_program = my_program.resolve_program(SequentialPlacer())",
        "    return my_program",
        "",
        "def main():",
        "    datatype = bfloat16",
        f"    data_size = {data_size}",
        f"    inputA = iron.arange(data_size, dtype=datatype, device=\"npu\")",
        f"    inputB = iron.arange(data_size, dtype=datatype, device=\"npu\")",
        f"    outputD = iron.zeros(data_size, dtype=datatype, device=\"npu\")",
        f"    design(inputA, inputB, outputD)",
        "    print(outputD)",
        "    # Validation check",
        "    inputA_data = np.arange(data_size, dtype=np.float32)  # Fallback since np.asarray may fail for IRON tensors",
        "    inputB_data = np.arange(data_size, dtype=np.float32)  # Fallback since np.asarray may fail for IRON tensors",
        "    expected = np.maximum(0, inputA_data + inputB_data)",
        "    actual = np.asarray(outputD, dtype=np.float32)",
        "    print('Element-by-element comparison:')",
        "    for i in range(data_size):",
        "        print(f'Expected = ReLU({inputA_data[i]:.1f} + {inputB_data[i]:.1f}) = {expected[i]:.1f} : Received = {actual[i]:.1f}')",
        "    tolerance = 1e-3  # Tolerance for bfloat16 comparison",
        "    mismatches = np.where(~np.isclose(actual, expected, rtol=tolerance))[0]",
        "    if len(mismatches) == 0:",
        f"        print('Validation passed: Output matches expected for all {data_size} elements')",
        "    else:",
        "        print(f'Validation failed: {len(mismatches)} mismatches found')",
        "        for idx in mismatches[:5]:  # Print up to 5 mismatches",
        "            print(f'Index {idx}: actual={actual[idx]:.1f}, expected={expected[idx]:.1f}')",
        "        if len(mismatches) > 5:",
        "            print(f'... and {len(mismatches) - 5} more mismatches')",
        "",
        'if __name__ == "__main__":',
        "    main()"
    ])

    #----------Write to File-----------------------
    with open(filepath, 'w') as python_file:
        python_file.write('\n'.join(imports_buffer) + '\n')
        python_file.write('\n'.join(types_buffer) + '\n')
        python_file.write("    # Object fifos goes here...\n")
        python_file.write('\n'.join(objectFifo_buffer) + '\n')
        python_file.write("    # Split/Join operations:\n")
        python_file.write('\n'.join(split_join_buffer) + '\n')
        python_file.write("    # External kernels:\n")
        python_file.write('\n'.join(external_buffer) + '\n')
        python_file.write("    # Core functions:\n")
        python_file.write('\n'.join(corefn_buffer) + '\n')
        python_file.write("    # Workers:\n")
        python_file.write('\n'.join(worker_buffer) + '\n')
        python_file.write("    # Runtime configuration:\n")
        python_file.write('\n'.join(runtime_buffer) + '\n')

    return filepath

def main():
    Placement = namedtuple('Placement', ['column', 'row'])

    # Define external functions with hardcoded kernel locations
    external_functions = [
        {
            'name': 'eltwise_add_bf16_scalar',
            'source_file': '/scratch/btsorens/mlir-aie/aie_kernels/aie2/add.cc',
            'include_dirs': ['/scratch/btsorens/mlir-aie/aie_kernels']
        },
        {
            'name': 'bf16_relu',
            'source_file': '/scratch/btsorens/mlir-aie/aie_kernels/aie2/relu.cc',
            'include_dirs': ['/scratch/btsorens/mlir-aie/aie_kernels']
        }
    ]

    #---------Graph Definition-----------
    dataFlowGraph = nx.MultiDiGraph()
    for col in range(4):
        shim = f"shim_{col}"
        mem = f"mem_{col}"
        dataFlowGraph.add_node(shim, placement=Placement(col, 0), type='shim')
        dataFlowGraph.add_node(mem, placement=Placement(col, 1), type='mem')
        dataFlowGraph.add_edge(shim, mem, depth=2, name=f"SHIM_L3_L2_A{col*2+1}A{col*2+2}_col{col}")
        dataFlowGraph.add_edge(shim, mem, depth=2, name=f"SHIM_L3_L2_B{col*2+1}B{col*2+2}_col{col}")
        dataFlowGraph.add_edge(mem, shim, depth=2, name=f"SHIM_L2_L3_D{col*2+1}D{col*2+2}_col{col}")

        for i in range(2):
            add_row = 5 if i == 0 else 3
            add = f"add_{col*2+i+1}"
            dataFlowGraph.add_node(add, placement=Placement(col, add_row), while_true=False, stack_size=1024,
                                 allocation_scheme='heap', trace=False, trace_events=None,
                                 core_fn='eltwise_add_bf16_scalar', type='compute')
            dataFlowGraph.add_edge(mem, add, depth=2, name=f"MEM_L2_L1_A{col*2+i+1}_col{col}")
            dataFlowGraph.add_edge(mem, add, depth=2, name=f"MEM_L2_L1_B{col*2+i+1}_col{col}")

            relu_row = 4 if i == 0 else 2
            relu = f"relu_{col*2+i+1}"
            dataFlowGraph.add_node(relu, placement=Placement(col, relu_row), while_true=False, stack_size=1024,
                                 allocation_scheme='heap', trace=False, trace_events=None,
                                 core_fn='bf16_relu', type='compute')
            dataFlowGraph.add_edge(add, relu, depth=2, name=f"L1_L1_add_relu_{col*2+i+1}")
            dataFlowGraph.add_edge(relu, mem, depth=2, name=f"MEM_L1_L2_D{col*2+i+1}_col{col}")

    # Generate the IRON code
    generated_file = generateIronCode(dataFlowGraph, 'generated_design.py', external_functions, data_size=128)
    print(f"Generated IRON code at: {generated_file}")

if __name__ == "__main__":
    main()