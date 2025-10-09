import networkx as nx
from collections import defaultdict

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
def generateIronCode(dataFlowGraph: nx.MultiDiGraph, filepath: str, external_functions:list = None, data_size: int = 8192):
    
    #Initialize external_functions as empty list if None
    external_functions = external_functions or []

    # create buffers for code insertions
    corefn_buffer = []
    worker_buffer = []
    tile_buffer = []
    external_buffer = []
    objectFifo_buffer = []
    runtime_buffer = []

    # Maps for variables (to be able to reference later)
    tile_map = {}
    worker_map = {}
    corefn_map = {}
    fifo_map = {}
    externalfn_map = {}
    fifo_name_map = {}

    #----------Input and Output Sources-------------
    # Collect external inputs and outputs of program through shim tiles
    # Analyzes edges connected to shim to determine input and output sources
    inputs = set()
    outputs = set()
    for prod, cons, key in dataFlowGraph.edges(keys=True):
        data = dataFlowGraph.edges[prod, cons, key]
        name = data.get('name')
        source_part = name.split('_')[3][0] if len(name.split('_')) > 3 else name
        if prod == 'shim':          # from shim is an input
            inputs.add(source_part)
        elif cons == 'shim':        # to shim is an output
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
        if prod == 'shim':
            input_fifos[source_part].append(fifo_variable)
        elif cons == 'shim':
            output_fifos[source_part].append(fifo_variable)


    # Get number of mem nodes (columns to parallelize across)
    mem_nodes = [node for node in dataFlowGraph.nodes() if dataFlowGraph.nodes[node].get('type') == 'mem']
    num_mem_nodes = len(mem_nodes)

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
        #Handle data types
        if source_part not in sorted_inputs + sorted_outputs:
            data_ty = 'data_ty'
        else:
            data_ty = f"data_{source_part.lower()}_ty"
        fifo_variable = fifo_map[(prod, cons, key)]
        fifo_name_map[fifo_variable] = name

        # Skip if edge involves mem (could be split or join (handled later))
        if prod == 'mem' or cons == 'mem' or (dataFlowGraph.nodes[prod].get('type') == 'mem' and dataFlowGraph.nodes[cons].get('type') == 'compute') or (dataFlowGraph.nodes[prod].get('type') == 'compute' and dataFlowGraph.nodes[cons].get('type') == 'mem'):
            continue

        # attach objectFifos for shim to objectFifo buffer
        if prod == 'shim' or cons == 'shim':
            objectFifo_buffer.append(f"{fifo_variable} = ObjectFifo({data_ty}, depth={depth}, name='{name}')")
        else:
            objectFifo_buffer.append(f"{fifo_variable} = ObjectFifo({data_ty}, {tile_map[prod]}, {tile_map[cons]}, depth={depth}, name='{name}')")

    #------------Memory Node split and join-------------
    # Split and join on mem nodes (L2 to L1) and (L1 to L2)
    for node in dataFlowGraph.nodes():
        attributes = dataFlowGraph.nodes[node]
        if attributes.get('type') == 'mem':
            in_edges = list(dataFlowGraph.in_edges(node, keys=True))
            out_edges = list(dataFlowGraph.out_edges(node, keys=True))
            col = attributes['placement'].column + 1

            # Process input edges (from shim through memory tiles to compute tiles)
            for pred, _, key in in_edges:
                if pred == 'shim':
                    fifo_var = fifo_map[(pred, node, key)]
                    name = fifo_name_map[fifo_var]
                    col_str = name.split('_')[-1]
                    col = int(col_str[3:])
                    source_part = name.split('_')[3][0] if len(name.split('_')) > 3 else name
                    data_ty = f"data_{source_part.lower()}_ty"
                    
                    # Count splits based on output edges with matching source names 
                    out_source_parts = [ dataFlowGraph.edges[node, cons, k]['name'].split('_')[3][0] if len(dataFlowGraph.edges[node, cons, k]['name'].split('_')) > 3 else '' for _, cons, k in out_edges ]
                    num_splits = len([1 for os in out_source_parts if os == source_part])
                    
                    # Handle split size and offset
                    chunk_size = data_size / (num_mem_nodes * num_splits)
                    offset_base = (col - 1) * num_splits * chunk_size
                    offsets = [offset_base + chunk_size * (i + 1) for i in range(num_splits)]
                    split_var = f"split_of_from_{node}_to_worker_{source_part}_key"
                    
                    #Generate split operation
                    objectFifo_buffer.append(
                        f"{split_var} = {fifo_var}.cons().split(offsets=[{', '.join(map(str, offsets))}], obj_type={data_ty}, depth=2, name='{split_var}', placement={tile_map[node]})")
                    
                    #Update fifo mappings for output edges
                    for _, cons, k in out_edges:
                        out_name = dataFlowGraph.edges[node, cons, k]['name']
                        out_source_part = out_name.split('_')[3][0] if len(out_name.split('_')) > 3 else ''
                        if out_name.startswith('MEM_L2_L1') and source_part == out_source_part:
                            fifo_map[(node, cons, k)] = split_var
                            fifo_name_map[split_var] = out_name
            
            # Process output edges (joining data from compute tiles through mem tile to shim tile)
            for _, succ, key in out_edges:
                if succ == 'shim':
                    fifo_var = fifo_map[(node, cons, key)]
                    name = fifo_name_map[fifo_var]
                    col_str = name.split('_')[-1]
                    col = int(col_str[3:])
                    source_part = name.split('_')[3][0] if len(name.split('_')) > 3 else name
                    data_ty = f"data_{source_part.lower()}_ty"
                    
                    #Count joins based on input edges with matching sources
                    in_source_parts = [ dataFlowGraph.edges[prod, node, k]['name'].split('_')[3][0] if len(dataFlowGraph.edges[prod, node, k]['name'].split('_')) > 3 else '' for prod, _, k in in_edges ]
                    num_joins = len([1 for is_ in in_source_parts if is_ == source_part])

                    # Handle join size and offset
                    chunk_size = data_size / (num_mem_nodes * num_joins)
                    offset_base = (col - 1) * num_joins * chunk_size
                    offsets = [offset_base + chunk_size * (i + 1) for i in range(num_joins)]
                    join_var = f"join_of_from_worker_{node}_to_{source_part}_key"
                    
                    #Generate join operation
                    objectFifo_buffer.append(
                        f"{join_var} = {fifo_var}.prod().join(obj_type={data_ty}, depth=2, name='{join_var}', placement={tile_map[node]})")
                    
                    # Update fifo mappings for input edges
                    for prod, _, k in in_edges:
                        in_name = dataFlowGraph.edges[prod, node, k]['name']
                        in_source_part = in_name.split('_')[3][0] if len(in_name.split('_')) > 3 else ''
                        if in_name.startswith('MEM_L1_L2') and source_part == in_source_part:
                            fifo_map[(prod, node, k)] = join_var
                            fifo_name_map[join_var] = in_name


    #--------------------Compute Nodes-------------
    # Process compute nodes
    # Generate external function and core function code
    for node in dataFlowGraph.nodes():
        attributes = dataFlowGraph.nodes[node]
        tile_type = attributes.get('type')
        if( tile_type == 'compute'):
            #Compute node attributes
            core_fn = attributes.get('core_fn')
            while_true = attributes.get('while_true')
            stack_size = attributes.get('stack_size')
            allocation_scheme = attributes.get('allocation_scheme')
            trace = attributes.get('trace')
            trace_events = attributes.get('trace_events')

            #check if core_fn is external function in list
            external_function_def = next((fn for fn in external_functions if fn['name'] == core_fn), None)
            in_edges = list(dataFlowGraph.in_edges(node, keys=True))
            out_edges = list(dataFlowGraph.out_edges(node, keys=True))

            #Core function definition buffers
            function_name = f"core_fn_{node}"
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
                worker_args.append(f"{fifo_var}.prod()")

            #----------------External Functions----------------
            #Generate external function if core_fn is external
            if external_function_def:
                source_file = external_function_def['source_file']
                include_dirs = external_function_def['include_dirs']
                num_arguments = len(in_edges) + len(out_edges)
                argument_types = external_function_def.get('arg_types', f"[data_ty] * {num_arguments}")

                #generate ExternalFunction object
                external_variable = f"external_{node}"
                external_buffer.append(
                    f"{external_variable} = ExternalFunction(\n"
                    f"    name=\"{core_fn}\",\n"
                    f"    source_file=\"{source_file}\",\n"
                    f"    arg_types={argument_types},\n"
                    f"    include_dirs={include_dirs}\n"
                    f")"
                )

                #Connect external function to map and worker arguments
                externalfn_map[node] = external_variable
                function_parameters.append(external_variable)
                worker_args.append(external_variable)

                #Generate core function with external function call
                corefn_buffer.append(f"def {function_name}({', '.join(function_parameters)}):")
                for line in in_acquires + out_acquires:
                    corefn_buffer.append(f"{line}")
                corefn_buffer.append(f"    {external_variable}({', '.join(call_args)})")
                for line in in_releases + out_releases:
                    corefn_buffer.append(f"{line}")

            #-------------Internal Core Functions---------------------
            # hard coded for now to include internal python functions eltwise_add_bf16 and bf16_relu
            # will need a solution for taking python functions from nodes
            else: 
                corefn_buffer.append(f"def {function_name}({', '.join(function_parameters)}):")
                for line in in_acquires + out_acquires:
                    corefn_buffer.append(f"{line}")
                if core_fn == 'eltwise_add_bf16_scalar':
                    corefn_buffer.append(f"    for i in range(len({call_args[0]})):")
                    corefn_buffer.append(f"        {call_args[2]}[i] = {call_args[0]}[i] + {call_args[1]}[i]")
                elif core_fn == 'bf16_relu':
                    corefn_buffer.append(f"    for i in range(len({call_args[0]})):")
                    corefn_buffer.append(f"        {call_args[1]}[i] = max(0, {call_args[0]}[i])")
                else:
                    raise ValueError(f"Unknown core_fn: {core_fn}")
                for line in in_releases + out_releases:
                    corefn_buffer.append(f"{line}")

            #----------------Workers-------------------------
            # write worker generated code to worker_buffer
            # define workers to execute core functions on specific tiles
            worker_variable = f"worker_{node}"
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
    runtime_buffer.append(f"   Workers = [" + ', '.join(worker_map.values()) + "]")
    runtime_buffer.append("   rt.start(*Workers)")
    
    # Generate data movement for input FIFOs (rt.fills)
    for source, fifos in input_fifos.items():
        for fifo_var in fifos:
            name = fifo_name_map[fifo_var]
            col_str = name.split('_')[-1]
            col = int(col_str[3:])
            offset = (data_size / 4) * col
            runtime_buffer.append(
                f"   rt.fill(in_fifo={fifo_var}.prod(), in_data={source}, tap=TensorAccessPattern(tensor_dims=[1,1024], offset={offset}, sizes=[1024, (data_size/4)/1024], strides=[1,1024]))")
    
    # Generate data movement for output FIFOs (rt.drains)
    for source, fifos in output_fifos.items():
        for fifo_var in fifos:
            name = fifo_name_map[fifo_var]
            col_str = name.split('_')[-1]
            col = int(col_str[3:])
            offset = (data_size / 4) * col
            runtime_buffer.append(
                f"   rt.drain(out_fifo={fifo_var}.cons(), out_data={source}, tap=TensorAccessPattern(tensor_dims=[1,1024], offset={offset}, sizes=[1024, (data_size/4)/1024], strides=[1, 1024]))")

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
def generated_design(""" + ", ".join(argument_names) + """):

    element_type = bfloat16
    data_size = """ + (f"{first_input}.numel()" if first_input else "0") + """
    data_ty = np.ndarray[(data_size,), np.dtype[element_type]]
    data_a_ty = np.ndarray[(data_size,), np.dtype[element_type]]
    data_b_ty = np.ndarray[(data_size,), np.dtype[element_type]]
    data_d_ty = np.ndarray[(data_size,), np.dtype[element_type]]
""")
        

        # write tile_buffer to file
        python_file.write("    # Define tiles for compute and shim nodes\n")
        for line in tile_buffer:
            python_file.write("    " + line + '\n')

        # write objectFifo_buffer to file
        python_file.write("\n    # Define object FIFOs for data streaming between tiles\n")
        for line in objectFifo_buffer:
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
            if source == 'A':
                python_file.write(f"    input{source} = iron.rand(data_size, dtype=datatype, device=\"npu\")\n")
            elif source == 'B':
                python_file.write(f"    input{source} = iron.arange(data_size, dtype=datatype, device=\"npu\", step=-1)\n")
        for source in sorted_outputs:
            python_file.write(f"    output{source} = iron.zeros(data_size, dtype=datatype, device=\"npu\")\n")
        python_file.write(f"    generated_design({', '.join(argument_names)})\n")
        for source in sorted_outputs:
            python_file.write(f"    print(output{source})\n")

        # call main function
        python_file.write("""if __name__ == "__main__":
    main()
""")
        # close file and return
        return filepath
    
#-------------Example main that creates an example graph and calls generateIronCode
def main():
    #
    import networkx as nx
    from collections import namedtuple
    Placement = namedtuple('Placement', ['column', 'row'])

    # Define external functions in a list
    external_functions = [
        {
            'name': 'eltwise_add_bf16_scalar',
            'source_file': '../../../aie_kernels/aie2/add.cc',
            'include_dirs': ['/scratch/andrewa/mlir-aie/aie_kernels/'],
            'arg_types': '[data_ty] * 3'  
        },
        {
            'name': 'bf16_relu',
            'source_file': '../../../aie_kernels/aie2/relu.cc',
            'include_dirs': ['/scratch/andrewa/mlir-aie/aie_kernels/'],
            'arg_types': '[data_ty] * 2'
        },
        {
            'name': 'matmul_bf16',
            'source_file': '../../../aie_kernels/aie2/matmul.cc',
            'include_dirs': ['/scratch/andrewa/mlir-aie/aie_kernels/'],
            'arg_types': '[data_ty] * 3' 
        }
    ]

    #---------Graph variant 1-----------
    # Create dataFlowGraph for add relu
    dataFlowGraph_aaa = nx.MultiDiGraph()

    # Add shim node at (0,0)
    dataFlowGraph_aaa.add_node('shim', placement=Placement(0, 0), type='shim')

    # Four columns with 1 memory tile and 4 compute nodes
    for col in range(4):
        #Memory node at (0,1), (1,1), (2,1), (3,1)
        mem = f"mem_col{col+1}"
        dataFlowGraph_aaa.add_node(mem, placement=Placement(col, 1), type='mem')


        # Add input edges from shim to memory (for inputs A and B)
        a_name = f"SHIM_L3_L2_A{col*2+1}A{col*2+2}_col{col+1}"
        b_name = f"SHIM_L3_L2_B{col*2+1}B{col*2+2}_col{col+1}"
        dataFlowGraph_aaa.add_edge('shim', mem, depth=2, name=a_name)
        dataFlowGraph_aaa.add_edge('shim', mem, depth=2, name=b_name)

        # Define memory tile to compute tile edge names
        a_name_mem = f"MEM_L2_L1_A{col*2+1}A{col*2+2}_col{col+1}"
        b_name_mem = f"MEM_L2_L1_B{col*2+1}B{col*2+2}_col{col+1}"
        d_name_mem = f"MEM_L1_L2_D{col*2+1}D{col*2+2}_col{col+1}"

        # Per column 2 add workers and 2 relu workers
        for i in range(2):
            #Add row 4 and 5 compute tiles (add)
            add_row = 4 if i == 0 else 5
            add = f"A{col*2 + i +1}_B{col*2 + i +1}_worker"
            dataFlowGraph_aaa.add_node(add, placement=Placement(col, add_row), while_true=False, stack_size=1024,
                                   allocation_scheme='heap', trace=False, trace_events=None,
                                   core_fn='eltwise_add_bf16_scalar', type='compute')

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
        d_name = f"SHIM_L2_L3_D{col*2 +1}D{col*2 +2}_col{col+1}"
        dataFlowGraph_aaa.add_edge(mem, 'shim', depth=2, name=d_name)

    # Generate the code
    generated_file = generateIronCode(dataFlowGraph_aaa, 'add_relu_generated_design.py', external_functions)
    print(f"Generated IRON code at: {generated_file}")

    #-------------Graph variant 2------------------------
    # Create dataFlowGraph for add matmul
    dataFlowGraph_amma = nx.MultiDiGraph()

    # Add shim node at (0,0)
    dataFlowGraph_amma.add_node('shim', placement=Placement(0, 0), type='shim')

    # Four columns with 1 memory tile and 4 compute nodes
    for col in range(4):
        #Memory node at (0,1), (1,1), (2,1), (3,1)
        mem = f"mem_col{col+1}"
        dataFlowGraph_amma.add_node(mem, placement=Placement(col, 1), type='mem')


        # Add input edges from shim to memory (for inputs A and B)
        a_name = f"SHIM_L3_L2_A{col*2+1}A{col*2+2}_col{col+1}"
        b_name = f"SHIM_L3_L2_B{col*2+1}B{col*2+2}_col{col+1}"
        dataFlowGraph_amma.add_edge('shim', mem, depth=2, name=a_name)
        dataFlowGraph_amma.add_edge('shim', mem, depth=2, name=b_name)

        # Define memory tile to compute tile edge names
        a_name_mem = f"MEM_L2_L1_A{col*2+1}A{col*2+2}_col{col+1}"
        b_name_mem = f"MEM_L2_L1_B{col*2+1}B{col*2+2}_col{col+1}"
        d_name_mem = f"MEM_L1_L2_D{col*2+1}D{col*2+2}_col{col+1}"

        # Per column 2 add workers and 2 matmul workers
        for i in range(2):
            #Add row 4 and 5 compute tiles (add)
            add_row = 4 if i == 0 else 5
            add = f"A{col*2 + i +1}_B{col*2 + i +1}_worker"
            dataFlowGraph_amma.add_node(add, placement=Placement(col, add_row), while_true=False, stack_size=1024,
                                   allocation_scheme='heap', trace=False, trace_events=None,
                                   core_fn='eltwise_add_bf16_scalar', type='compute')

            #Edges from mem to add node
            dataFlowGraph_amma.add_edge(mem, add, depth=2, name=a_name_mem)
            dataFlowGraph_amma.add_edge(mem, add, depth=2, name=b_name_mem)

            #Add rows 2 and 3 compute tiles (matmul)
            matmul_row = 2 if i == 0 else 3
            matmul = f"C{col*2 + i +1}_worker"
            dataFlowGraph_amma.add_node(matmul, placement=Placement(col, matmul_row), while_true=False, stack_size=1024,
                                   allocation_scheme='heap', trace=False, trace_events=None,
                                   core_fn='matmul_bf16', type='compute')

            # Edge from add to matmul
            dataFlowGraph_amma.add_edge(add, matmul, depth=2, name="L1_L1_elwiseadd_matmul")

            #Edge from matmul to mem
            dataFlowGraph_amma.add_edge(matmul, mem, depth=2, name=d_name_mem)

        #Output from mem to shim
        d_name = f"SHIM_L2_L3_D{col*2 +1}D{col*2 +2}_col{col+1}"
        dataFlowGraph_amma.add_edge(mem, 'shim', depth=2, name=d_name)

    # Generate the code
    generated_file = generateIronCode(dataFlowGraph_amma, 'add_matmul_generated_design.py', external_functions)
    print(f"Generated IRON code at: {generated_file}")


#---------main function---------------
if __name__ == "__main__":
    main()