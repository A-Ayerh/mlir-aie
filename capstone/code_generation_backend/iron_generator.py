import networkx as nx
from collections import defaultdict
import re
import numpy as np
from ml_dtypes import bfloat16

# Generator function that will take the dataFlowGraph and the new python file as arguments, 
# will iterate through the graph nodes and edges, and write the corresponding IRON code to 
# the file

#Current limitations:
# Assumes only one shim node for all external I/O
#   - Edges to/from this shim determine all inputs/outputs
# Mem tiles will always split/join based on the number of outgoing/incoming edges
#   - Always splits to compute tiles in its column based on data required
#   - Can't do mem to mem, or mem to non-local-column compute tiles
# Edge names are hard coded to determine sources and data types
#   - changing names breaks source detection, data_types, and column extraction
# Data sources are hard coded (inputA, inputB)
# Local in-file core_fns are hard coded (add and relu)
#   - However, external functions are flexible by passing the external_functions list
# External functions assume default arg_types with flexible number of arguments
# Runtime data size and type is fixed to bfloat16, data type change would require code change
# Hard coded TensorAccessPatterns in runtime code
# Limited error checking and default values
# 
def generateIronCode(dataFlowGraph: nx.MultiDiGraph, filepath: str, external_functions:list = None, data_size: int = 2048):
    
    #Initialize external_functions as empty list if None
    external_functions = external_functions or []

    # create buffers for code insertions
    corefn_buffer = []
    worker_buffer = []
    tile_buffer = []
    external_buffer = []
    objectFifo_buffer = []
    runtime_buffer = []
    split_join_buffer = []
    internal_fifo_buffer = []
    

    # Maps for variables (to be able to reference later)
    tile_map = {}
    worker_map = {}
    corefn_map = {}
    fifo_map = {}
    externalfn_map = {}
    fifo_name_map = {}
    split_outputs = {}
    externalfn_instances = {}

    #----------Input and Output Sources-------------
    # Collect external inputs and outputs of program through shim tiles
    # Analyzes edges connected to shim to determine input and output sources
    inputs = set()
    outputs = set()
    for prod, cons, key in dataFlowGraph.edges(keys=True):
        data = dataFlowGraph.edges[prod, cons, key]
        name = data.get('name')
        source_part = name.split('_')[3][0] if len(name.split('_')) > 3 else name
        if dataFlowGraph.nodes[prod].get('type') == 'shim': 
            inputs.add(source_part)
        elif dataFlowGraph.nodes[cons].get('type') == 'shim':  
            outputs.add(source_part)
    sorted_inputs = sorted(inputs)
    sorted_outputs = sorted(outputs)
    # Create argument names for generated function (inputA, etc)
    argument_names = [f"input{source}" for source in sorted_inputs] + [f"output{source}" for source in sorted_outputs]
    sequential_variables = sorted_inputs + sorted_outputs
    first_input = argument_names[0] if argument_names else None # hard coded used first input as reference for data size

    #--------------ObjectFifo Grouping------------------
    # Group input/output FIFOs by source for runtime data movement
    input_fifos = defaultdict(list)
    output_fifos = defaultdict(list)
    for prod, cons, key in dataFlowGraph.edges(keys=True):
        data = dataFlowGraph.edges[prod, cons, key]
        name = data.get('name')
        # hard coded extract source part from name
        source_part = name.split('_')[3][0] if len(name.split('_')) > 3 else name
        fifo_variable = f"of_from_{prod}_to_{cons}_{key}"
        fifo_map[(prod, cons, key)] = fifo_variable
        fifo_name_map[fifo_variable] = name
        if dataFlowGraph.nodes[prod].get('type') == 'shim': 
            input_fifos[source_part].append(fifo_variable)
        elif dataFlowGraph.nodes[cons].get('type') == 'shim': 
            output_fifos[source_part].append(fifo_variable)

    # Get number of mem nodes (columns to parallelize across)
    mem_nodes = [node for node in dataFlowGraph.nodes() if dataFlowGraph.nodes[node].get('type') == 'mem']
    num_mem_nodes = len(mem_nodes)

    # Calculate dynamic chunk sizes per column
    MAX_REPEAT_COUNT = 255
    max_chunk_size = MAX_REPEAT_COUNT + 1  # 256 max
    chunk_sizes = {}
    col_data_sizes = {}

    for node in mem_nodes:
        col_idx = dataFlowGraph.nodes[node]['placement'].column
        col_data_size = data_size // num_mem_nodes
        
        # Count workers per column for this mem node
        out_edges = list(dataFlowGraph.out_edges(node, keys=True))
        out_edges = list(dataFlowGraph.out_edges(node, keys=True))
        worker_count = sum(1 for _, cons, key in out_edges 
                        if (dataFlowGraph.nodes[cons].get('type') == 'compute' and
                            dataFlowGraph.edges[node, cons, key]['name'].startswith('MEM_L2_L1')))
        
        chunk_size = min(max_chunk_size, col_data_size // max(worker_count, 1))

        chunk_size = (chunk_size // 16) * 16  # Align to 16-byte boundary for efficiency
        if chunk_size == 0:
            chunk_size = 16  # Minimum workable size
        
        chunk_sizes[col_idx] = chunk_size
        col_data_sizes[col_idx] = col_data_size

    #----------Tile Definitions-------------------
    # Iterate through graph nodes to generate tile definitions
    for node in dataFlowGraph.nodes():
        # Extract node data for placement and config
        attributes = dataFlowGraph.nodes[node]
        placement = attributes.get('placement')
        column = placement.column
        row = placement.row

        # write tile generated code to tile_buffer
        tile_variable = f"tile_{column}_{row}"
        tile_buffer.append(f"{tile_variable} = Tile({column}, {row})")
        tile_map[node] = tile_variable

    #-----------ObjectFifo Definitions-------------
    #Generate ObjectFifo defintions for streaming between tiles
    for prod, cons, key in dataFlowGraph.edges(keys=True):
        data = dataFlowGraph.edges[prod, cons, key]
        depth = data.get('depth')
        name = data.get('name', f"of_{prod}_to{cons}")
        #Hard coded extraction of source for data type reference
        source_part = name.split('_')[3][0] if len(name.split('_')) > 3 else name
        fifo_variable = fifo_map[(prod, cons, key)]
        fifo_name_map[fifo_variable] = name

        # Only create base FIFOs for shim<->mem connections
        if dataFlowGraph.nodes[prod].get('type') == 'shim' or dataFlowGraph.nodes[cons].get('type') == 'shim':
            if source_part.lower() == 'a':
                fifo_type = "data_a_ty"
            elif source_part.lower() == 'b':
                fifo_type = "data_b_ty"
            elif source_part.lower() == 'd':
                fifo_type = "data_d_ty"
            else:
                fifo_type = "data_ty"
            # Assign placement to shim tile
            shim_tile = tile_map[prod] if dataFlowGraph.nodes[prod].get('type') == 'shim' else tile_map[cons]
            objectFifo_buffer.append(f"{fifo_variable} = ObjectFifo({fifo_type}, depth={depth}, name='{name}')")
    
    #------------Memory Node split and join-------------
    # Split and join on mem nodes (L2 to L1) and (L1 to L2)
    for node in dataFlowGraph.nodes():
        if dataFlowGraph.nodes[node].get('type') == 'mem':
            attributes = dataFlowGraph.nodes[node]
            col_idx = attributes['placement'].column
            tile_var = tile_map[node]
            in_edges = list(dataFlowGraph.in_edges(node, keys=True))
            out_edges = list(dataFlowGraph.out_edges(node, keys=True))
            col_data_size = col_data_sizes[col_idx]
            chunk_size = chunk_sizes[col_idx]  # Use per-column chunk size
            col_offset = col_idx * col_data_size

            # Process input edges (from shim through memory tiles to compute tiles)
            for pred, _, key in in_edges:
                if dataFlowGraph.nodes[pred].get('type') == 'shim':  # FIXED: Check node type
                    fifo_var = fifo_map[(pred, node, key)]
                    name = fifo_name_map[fifo_var]
                    source_part = name.split('_')[3][0] if len(name.split('_')) > 3 else name
                    
                    # Count workers that need this input data
                    num_workers = sum(1 for _, cons, k in out_edges 
                                    if (dataFlowGraph.nodes[cons].get('type') == 'compute' and
                                        dataFlowGraph.edges[node, cons, k]['name'].startswith('MEM_L2_L1') and
                                        len(dataFlowGraph.edges[node, cons, k]['name'].split('_')) > 3 and
                                        dataFlowGraph.edges[node, cons, k]['name'].split('_')[3][0] == source_part))
                    
                    if num_workers > 0:
                        offsets = [i * chunk_size for i in range(num_workers)]
                        split_var_base = f"split_{node}_{source_part}"
                        
                        # Create ONE split operation for all sub-FIFOs
                        offsets_str = ', '.join(map(str, offsets))
                        split_join_buffer.append(
                            f"{split_var_base} = {fifo_var}.cons().split("
                            f"offsets=[{offsets_str}], "
                            f"obj_types=[chunk_ty] * {num_workers}, "
                            f"depths=[2] * {num_workers}, "
                            f"names=[f'{split_var_base}_{{i}}' for i in range({num_workers})], "
                            f"placement={tile_var})"
                        )

                        # Track which compute nodes connect to which split index
                        split_consumers = []
                        consumer_idx = 0
                        for _, cons, k in out_edges:
                            edge_name = dataFlowGraph.edges[node, cons, k]['name']
                            if (edge_name.startswith('MEM_L2_L1') and 
                                len(edge_name.split('_')) > 3 and 
                                edge_name.split('_')[3][0] == source_part):
                                split_consumers.append((cons, consumer_idx))
                                consumer_idx += 1
                        
                        # Store for worker connections
                        split_outputs[(node, source_part)] = {
                            'var': split_var_base,
                            'num_workers': num_workers,
                            'offsets': offsets,
                            'chunk_size': chunk_size,
                            'consumers': split_consumers
                        }

            # Process output edges (joining data from compute tiles through mem tile to shim tile)
            for _, succ, key in out_edges:
                if dataFlowGraph.nodes[succ].get('type') == 'shim':  # FIXED: Check node type
                    fifo_var = fifo_map[(node, succ, key)]
                    name = fifo_name_map[fifo_var]
                    source_part = name.split('_')[3][0] if len(name.split('_')) > 3 else name
                    
                    num_workers = sum(1 for prod, _, k in in_edges 
                                    if (dataFlowGraph.nodes[prod].get('type') == 'compute' and  # FIXED: Check node type
                                        dataFlowGraph.edges[prod, node, k]['name'].startswith('MEM_L1_L2') and
                                        len(dataFlowGraph.edges[prod, node, k]['name'].split('_')) > 3 and
                                        dataFlowGraph.edges[prod, node, k]['name'].split('_')[3][0] == source_part))
                    
                    if num_workers > 0:
                        offsets = [col_offset + i * chunk_size for i in range(num_workers)]
                        join_var_base = f"join_{node}_{source_part}"
                        
                        # Create ONE join operation for all sub-FIFOs
                        offsets_str = ', '.join(map(str, offsets))
                        split_join_buffer.append(
                            f"{join_var_base} = {fifo_var}.prod().join("
                            f"offsets=[{offsets_str}], "
                            f"obj_types=[chunk_ty] * {num_workers}, "
                            f"depths=[2] * {num_workers}, "
                            f"names=[f'{join_var_base}_{{i}}' for i in range({num_workers})], "
                            f"placement={tile_var})"
                        )

                        # Track which compute nodes contribute to which join index - FIXED: Move inside if block
                        join_producers = []
                        producer_idx = 0
                        for prod, _, k in in_edges:
                            edge_name = dataFlowGraph.edges[prod, node, k]['name']
                            if (edge_name.startswith('MEM_L1_L2') and 
                                len(edge_name.split('_')) > 3 and 
                                edge_name.split('_')[3][0] == source_part):
                                join_producers.append((prod, producer_idx))
                                producer_idx += 1
                        
                        # Store for worker connections
                        split_outputs[(node, source_part, 'join')] = {
                            'var': join_var_base,
                            'num_workers': num_workers,
                            'offsets': offsets,
                            'chunk_size': chunk_size,
                            'producers': join_producers
                        }

    # ------------Internal ObjectFifo Definitions (compute->compute and compute->mem only)-------------
    # Generate internal FIFOs after splits are defined, but skip mem->compute (handled by split refs)
    for prod, cons, key in dataFlowGraph.edges(keys=True):
        data = dataFlowGraph.edges[prod, cons, key]
        depth = data.get('depth', 2)
        name = data.get('name')
        fifo_variable = fifo_map[(prod, cons, key)]
        
        # Skip shim<->mem and mem->compute (handled by splits/joins)
        if ((dataFlowGraph.nodes[prod].get('type') == 'shim' and cons in mem_nodes) or 
            (dataFlowGraph.nodes[cons].get('type') == 'shim' and prod in mem_nodes) or
            (dataFlowGraph.nodes[prod].get('type') == 'mem' and 
            dataFlowGraph.nodes[cons].get('type') == 'compute')):
            continue
        
        # Only create for compute->compute and compute->mem connections
        if dataFlowGraph.nodes[prod].get('type') == 'compute':
            unique_suffix = f"{prod}_{cons}_{key}"
            fifo_type = "chunk_ty"
            internal_fifo_buffer.append(
                f"{fifo_variable} = ObjectFifo({fifo_type}, depth={depth}, name='{name}_{unique_suffix}')"
            )

    #--------------------Compute Nodes-------------
    # Process compute nodes
    # Generate external function and core function code
    for node in dataFlowGraph.nodes():
        attributes = dataFlowGraph.nodes[node]
        tile_type = attributes.get('type')
        if tile_type == 'compute':
            core_fn = attributes.get('core_fn')
            while_true = attributes.get('while_true', False)
            stack_size = attributes.get('stack_size', 1024)
            allocation_scheme = attributes.get('allocation_scheme', 'heap')
            trace = attributes.get('trace', False)
            trace_events = attributes.get('trace_events', None)

            external_function_def = next((fn for fn in external_functions if fn['name'] == core_fn), None)
            in_edges = list(dataFlowGraph.in_edges(node, keys=True))
            out_edges = list(dataFlowGraph.out_edges(node, keys=True))

            function_name = f"core_fn_{node.replace('_', '')}"
            function_parameters = []
            in_acquires = []
            in_releases = []
            out_acquires = []
            out_releases = []
            call_args = []
            worker_args = []

            #Iterates through incoming edges to generate parameters and acquire/release
            for i, (pred, _, key) in enumerate(in_edges):
                of_variable = f"of_in{i+1}"
                function_parameters.append(of_variable)
                fifo_var = fifo_map[(pred, node, key)]
                element = f"elem_in{i+1}"
                in_acquires.append(f"    {element} = {of_variable}.acquire(1)")
                in_releases.append(f"    {of_variable}.release(1)")
                call_args.append(element)
                
                # Handle split FIFO connections from memory
                if dataFlowGraph.nodes[pred].get('type') == 'mem':
                    source_part = dataFlowGraph.edges[pred, node, key]['name'].split('_')[3][0]
                    split_key = (pred, source_part)
                    if split_key in split_outputs:
                        split_info = split_outputs[split_key]
                        consumer_entry = next((entry for entry in split_info['consumers'] if entry[0] == node), None)
                        if consumer_entry:
                            split_idx = consumer_entry[1]
                            # Reference the split sub-FIFO using index notation
                            worker_args.append(f"{split_info['var']}[{split_idx}].cons()")
                            continue
                    worker_args.append(f"{fifo_var}.cons()")
                else:
                    worker_args.append(f"{fifo_var}.cons()")

            #Iterates through outgoing edges to generate parameters and acquire/release
            for j, (_, succ, key) in enumerate(out_edges):
                of_variable = f"of_out{j+1}"
                function_parameters.append(of_variable)
                fifo_var = fifo_map[(node, succ, key)]
                elem = f"elem_out{j+1}"
                out_acquires.append(f"    {elem} = {of_variable}.acquire(1)")
                out_releases.append(f"    {of_variable}.release(1)")
                call_args.append(elem)
                
                # Handle connections to memory joins
                if dataFlowGraph.nodes[succ].get('type') == 'mem':
                    source_part = dataFlowGraph.edges[node, succ, key]['name'].split('_')[3][0]
                    join_key = (succ, source_part, 'join')
                    if join_key in split_outputs:
                        join_info = split_outputs[join_key]
                        producer_entry = next((entry for entry in join_info['producers'] if entry[0] == node), None)
                        if producer_entry:
                            join_idx = producer_entry[1]
                            worker_args.append(f"{join_info['var']}[{join_idx}].prod()")
                            continue
                    worker_args.append(f"{fifo_var}.prod()")
                else:
                    worker_args.append(f"{fifo_var}.prod()")

            #----------------External Functions----------------
            #Generate external function if core_fn is external
            chunk_size = 512
            element_type = bfloat16
            data_ty = np.ndarray[(data_size,), np.dtype[element_type]]
            chunk_ty = np.ndarray[(chunk_size,), np.dtype[element_type]]
            col_ty = np.ndarray[(col_data_size,), np.dtype[element_type]]

            if external_function_def:
                if core_fn not in externalfn_instances:
                    source_file = external_function_def['source_file']
                    include_dirs = external_function_def['include_dirs']
                    
                    argument_types = external_function_def.get('arg_types')
                    if not argument_types:
                        if core_fn == 'eltwise_add_bf16_vector':
                            argument_types = "[chunk_ty, chunk_ty, chunk_ty]"
                        elif core_fn == 'bf16_relu':
                            argument_types = "[chunk_ty, chunk_ty]"
                        else:
                            num_arguments = len(in_edges) + len(out_edges)
                            argument_types = f"[chunk_ty] * {num_arguments}"

                    external_variable = f"external_{core_fn.replace('_', '')}"
                    external_buffer.append(
                        f"{external_variable} = ExternalFunction("
                        f"name=\"{core_fn}\", "
                        f"source_file=\"{source_file}\", "
                        f"arg_types={argument_types}, "
                        f"include_dirs={include_dirs}"
                        f")"
                    )
                    externalfn_instances[core_fn] = external_variable

                external_variable = externalfn_instances[core_fn]
                externalfn_map[node] = external_variable
                function_parameters.append(external_variable)
                worker_args.append(external_variable)

                # Generate core function with proper external function call
                corefn_buffer.append(f"def {function_name}({', '.join(function_parameters)}):")
                for line in in_acquires + out_acquires:
                    corefn_buffer.append(f"    {line}")

                # Handle function-specific argument passing
                if core_fn == 'eltwise_add_bf16_vector':
                    # Expects: inputA, inputB, output (3 arguments)
                    if len(call_args) >= 3:
                        call_args_for_external = call_args[:3]  # inputA, inputB, output
                    else:
                        call_args_for_external = call_args
                elif core_fn == 'bf16_relu':
                    # Expects: input, output (2 arguments)
                    if len(call_args) >= 2:
                        call_args_for_external = call_args[:2]  # input, output
                    else:
                        call_args_for_external = call_args
                else:
                    # For other functions, use all call_args
                    call_args_for_external = call_args
                
                corefn_buffer.append(f"        {external_variable}({', '.join(call_args_for_external)})")
                for line in in_releases + out_releases:
                    corefn_buffer.append(f"    {line}")

            #-------------Internal Core Functions---------------------
            # hard coded for now to include internal python functions eltwise_add_bf16 and bf16_relu
            # will need a solution for taking python functions from nodes
            else: 
                corefn_buffer.append(f"def {function_name}({', '.join(function_parameters)}):")
                for line in in_acquires + out_acquires:
                    corefn_buffer.append(f"{line}")

                if core_fn == 'eltwise_add_bf16_vector':
                    # Assumes 2 inputs (A, B) and 1 output (C)
                    # Get shape from first input MemRef
                    corefn_buffer.append(f"    n = {call_args[0]}.shape[0]  # Get length from MemRef shape")
                    corefn_buffer.append(f"    print(f'DEBUG: {function_name} processing {{n}} elements')")
                    corefn_buffer.append(f"    for i in range(n):")
                    corefn_buffer.append(f"        {call_args[2]}[i] = {call_args[0]}[i] + {call_args[1]}[i]")
                    # Debug first element
                    #orefn_buffer.append(f"    if n > 0:")
                    corefn_buffer.append(f"        print(f'DEBUG: {function_name} first: {{ {call_args[0]}[0] }} + {{ {call_args[1]}[0] }} = {{ {call_args[2]}[0] }}')")
                elif core_fn == 'bf16_relu':
                    corefn_buffer.append(f"    n = {call_args[0]}.shape[0]")
                    corefn_buffer.append(f"    print(f'DEBUG: {function_name} processing {{n}} elements')")
                    corefn_buffer.append(f"    zero = 0.0")
                    corefn_buffer.append(f"    for i in range(n):")
                    # Use max instead of conditional
                    corefn_buffer.append(f"        {call_args[1]}[i] = {call_args[0]}[i] if {call_args[0]}[i] > zero else zero")
                else:
                    raise ValueError(f"Unknown core_fn: {core_fn}")
                
                for line in in_releases + out_releases:
                    corefn_buffer.append(line)

            #----------------Workers-------------------------
            # write worker generated code to worker_buffer
            # define workers to execute core functions on specific tiles
            worker_variable = f"worker_{node.replace('_', '')}"
            worker_buffer.append(
                f"{worker_variable} = Worker({function_name}, [{', '.join(worker_args)}], "
                f"placement={tile_map[node]}, while_true={while_true}, stack_size={stack_size}, "
                f"allocation_scheme='{allocation_scheme}', trace={trace}, trace_events={trace_events})")
            worker_map[node] = worker_variable
    
    #------------------------Runtime Config---------------------
    # extract appropriate info need for runtime environment code and write to runtime_buffer 
    # starting workers, manage input, output data movement
    runtime_buffer.append("rt = Runtime()")
    seq_types = ', '.join([f"data_{source.lower()}_ty" for source in sequential_variables])
    runtime_buffer.append(f"with rt.sequence({seq_types}) as ({','.join(sequential_variables)}):")
    runtime_buffer.append(f"   Workers = [" + ', '.join([f'{w}' for w in worker_map.values()]) + "]")
    runtime_buffer.append("   rt.start(*Workers)")
        
    npu_chunk_size = chunk_sizes[0]
    
    # Generate data movement for input FIFOs (rt.fills)
    for source, fifos in input_fifos.items():
        for fifo_var in fifos:
            name = fifo_name_map[fifo_var]
            col_match = re.search(r'col(\d+)', name)
            if col_match:
                col = int(col_match.group(1))
                col_offset = col * col_data_sizes[col]
                col_chunk_size = chunk_sizes[col]
                num_chunks = col_data_sizes[col] // col_chunk_size
                
                for chunk_idx in range(num_chunks):
                    chunk_offset = col_offset + chunk_idx * col_chunk_size
                    runtime_buffer.append(
                        f"   rt.fill({fifo_var}.prod(), {source}, "
                        f"tap=TensorAccessPattern(tensor_dims=[{data_size},], "
                        f"offset={chunk_offset}, sizes=[{col_chunk_size},], strides=[1,]))"
                    )
            
    # Generate data movement for output FIFOs (rt.drains)
    for source, fifos in output_fifos.items():
        for fifo_var in fifos:
            name = fifo_name_map[fifo_var]
            col_match = re.search(r'col(\d+)', name)
            if col_match:
                col = int(col_match.group(1))
                col_offset = col * col_data_sizes[col]
                col_chunk_size = chunk_sizes[col]
                num_chunks = col_data_sizes[col] // col_chunk_size
                
                for chunk_idx in range(num_chunks):
                    chunk_offset = col_offset + chunk_idx * col_chunk_size
                    runtime_buffer.append(
                        f"   rt.drain({fifo_var}.cons(), {source}, "
                        f"wait=True, tap=TensorAccessPattern(tensor_dims=[{data_size},], "
                        f"offset={chunk_offset}, sizes=[{col_chunk_size},], strides=[1,]))"
                    )
            
    #--------------Write to generated File---------------
    # open file and write all generated code to the output file
    with open(filepath, 'w') as python_file:

        # write imports and header code to file
        python_file.write(
            """import aie.iron as iron
from aie.iron import ExternalFunction, jit
from aie.iron import ObjectFifo, Worker, Runtime, Program
from aie.iron.placers import SequentialPlacer
from aie.iron.device import Tile
import numpy as np
from ml_dtypes import bfloat16
from aie.helpers.taplib import TensorAccessPattern
            
            
@iron.jit(is_placed=False)
def generated_design(""" + ", ".join(argument_names) + """):\n""")
        python_file.write(
f"""
    element_type = bfloat16
    data_size = {first_input}.numel() if {first_input} else 0
    num_mem_nodes = {num_mem_nodes}
    col_data_size = data_size // num_mem_nodes\n""")
        min_chunk_size = min(chunk_sizes.values()) if chunk_sizes else (data_size // num_mem_nodes // 2)
        
        python_file.write(
f"""
    chunk_size = {min_chunk_size}\n""")
        
        python_file.write("""
    max_chunk_size = 256
    data_ty = np.ndarray[(data_size,), np.dtype[element_type]]
    chunk_ty = np.ndarray[(max_chunk_size,), np.dtype[element_type]]
    col_ty = np.ndarray[(col_data_size,), np.dtype[element_type]]
    
    # Input/output specific types
    data_a_ty = np.ndarray[(data_size,), np.dtype[element_type]]
    data_b_ty = np.ndarray[(data_size,), np.dtype[element_type]]
    data_d_ty = np.ndarray[(data_size,), np.dtype[element_type]]
""")
        

        # write tile_buffer to file
        python_file.write("    # Define tiles for compute and shim nodes\n")
        for line in tile_buffer:
            python_file.write("    " + line + '\n')

        # write base objectFifo_buffer to file
        python_file.write("\n    # Define base object FIFOs for shim <-> memory connections\n")
        for line in objectFifo_buffer:
            python_file.write("    " + line + '\n')

        # write split_join_buffer FIRST (before internal FIFOs)
        python_file.write("\n    # Split/Join operations on memory tiles\n")
        for line in split_join_buffer:
            python_file.write("    " + line + '\n')

        # write internal_fifo_buffer to file
        python_file.write("\n    # Define internal object FIFOs (compute->compute, compute->mem)\n")
        for line in internal_fifo_buffer:
            python_file.write("    " + line + '\n')

        # write externalfn_buffer to file
        if external_buffer:
            python_file.write("\n    # Define external C/C++ kernel functions\n")
            for line in external_buffer:
                python_file.write("    " + line + '\n')

        # write corefn_buffer to file
        python_file.write("\n    # Define core functions for each compute node\n")
        for line in corefn_buffer:
            python_file.write("    " + line + '\n')

        # write worker_buffer to file
        python_file.write("\n    # Define workers to execute core functions on tiles\n")
        for line in worker_buffer:
            python_file.write("    " + line + '\n')

        # write runtime date movement to file
        python_file.write("\n    # Define runtime sequence for starting workers and moving data\n")
        for line in runtime_buffer:
            python_file.write("    " + line + '\n')

        # create and return Program and finish file
        python_file.write(
            f"""    my_program = Program(iron.get_current_device(), rt)
    my_program = my_program.resolve_program(SequentialPlacer())
    return my_program

def main():
    datatype = bfloat16
    data_size = {data_size}
""")
        
        # write input and output sources to file
        for source in sorted_inputs:
            python_file.write(f"    input{source} = iron.rand(data_size, dtype=datatype, device=\"npu\")\n")
        for source in sorted_outputs:
            python_file.write(f"    output{source} = iron.zeros(data_size, dtype=datatype, device=\"npu\")\n")
        python_file.write(f"    program = generated_design({', '.join(argument_names)})\n")
        python_file.write("    program()\n")
        
        for source in sorted_outputs:
            python_file.write(f"    print(iron.to_numpy(output{source}))\n")

        python_file.write(
            """if __name__ == "__main__":
    main()
"""
        )
        # close file and return
        return filepath
    
#---------main function---------------
def main():
    import networkx as nx
    from collections import namedtuple
    Placement = namedtuple('Placement', ['column', 'row'])

    # Define external functions in a list
    external_functions = [
    {
        'name': 'eltwise_add_bf16_vector',
        'source_file': './add.cc',  # Relative to code_generation_backend folder
        'include_dirs': ['./'],      # Current directory for headers
        #'arg_types': '[chunk_ty, chunk_ty, chunk_ty, bfloat16]'  # inputA, inputB, output, size
    },
    {
        'name': 'bf16_relu',
        'source_file': './relu.cc',  # Points to relu.cc in same folder
        'include_dirs': ['./'],       # Current directory
        #'arg_types': '[chunk_ty, chunk_ty, int]'  # input, output, size
    }
]

    #---------Graph variant 1-----------
    # Create dataFlowGraph for add relu
    dataFlowGraph_aaa = nx.MultiDiGraph()

    # Add 4 shim nodes at row 0, columns 0-3
    for col in range(4):
        shim = f"shim_col{col}"
        dataFlowGraph_aaa.add_node(shim, placement=Placement(col, 0), type='shim')

    # Four columns with 1 memory tile and 4 compute nodes
    for col in range(4):
        #Memory node at row 1, column col
        mem = f"mem_col{col}"
        dataFlowGraph_aaa.add_node(mem, placement=Placement(col, 1), type='mem')

        # Shim node for this column
        shim = f"shim_col{col}"

        # Add input edges from shim to memory (for inputs A and B)
        a_name = f"SHIM_L3_L2_A{col*2+1}A{col*2+2}_col{col}"
        b_name = f"SHIM_L3_L2_B{col*2+1}B{col*2+2}_col{col}"
        dataFlowGraph_aaa.add_edge(shim, mem, depth=2, name=a_name)
        dataFlowGraph_aaa.add_edge(shim, mem, depth=2, name=b_name)

        # Define memory tile to compute tile edge names
        a_name_mem = f"MEM_L2_L1_A{col*2+1}A{col*2+2}_col{col}"
        b_name_mem = f"MEM_L2_L1_B{col*2+1}B{col*2+2}_col{col}"
        d_name_mem = f"MEM_L1_L2_D{col*2+1}D{col*2+2}_col{col}"

        # Per column 2 add workers and 2 relu workers
        for i in range(2):
            #Add row 4 and 5 compute tiles (add)
            add_row = 4 if i == 0 else 5
            add = f"A{col*2 + i +1}_B{col*2 + i +1}_worker"
            dataFlowGraph_aaa.add_node(add, placement=Placement(col, add_row), while_true=False, stack_size=1024,
                                   allocation_scheme='heap', trace=False, trace_events=None,
                                   core_fn='eltwise_add_bf16_vector', type='compute')

            #Edges from mem to add node
            dataFlowGraph_aaa.add_edge(mem, add, depth=2, name=a_name_mem)
            dataFlowGraph_aaa.add_edge(mem, add, depth=2, name=b_name_mem)

            #Add rows 2 and 3 compute tiles(relu)
            relu_row = 2 if i == 0 else 3
            relu = f"C{col*2 + i +1}_worker"
            dataFlowGraph_aaa.add_node(relu, placement=Placement(col, relu_row), while_true=False, stack_size=1024,
                                   allocation_scheme='heap', trace=False, trace_events=None,
                                   core_fn='bf16_relu', type='compute')

            # Edge from add to relu
            dataFlowGraph_aaa.add_edge(add, relu, depth=2, name="L1_L1_elwiseadd_relu")

            #Edge from relu to mem
            dataFlowGraph_aaa.add_edge(relu, mem, depth=2, name=d_name_mem)

        #Output from mem to shim
        d_name = f"SHIM_L2_L3_D{col*2 +1}D{col*2 +2}_col{col}"
        dataFlowGraph_aaa.add_edge(mem, shim, depth=2, name=d_name)  # Connect back to the same shim

    # Generate the code
    generated_file = generateIronCode(dataFlowGraph_aaa, 'add_relu_generated_design.py', external_functions, data_size=1024)
    print(f"Generated IRON code at: {generated_file}")

    # Generate the code
    generated_file = generateIronCode(dataFlowGraph_aaa, 'generated_design.py', external_functions=None, data_size=1024)
    print(f"Generated IRON code at: {generated_file}")

if __name__ == "__main__":
    main()