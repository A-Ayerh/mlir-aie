import networkx as nx
from collections import defaultdict

# Generator function that will take the dataFlowGraph and the new python file as arguments, 
# will iterate through the graph nodes and edges, and write the corresponding IRON code to 
# the file

def generateIronCode(dataFlowGraph: nx.MultiDiGraph, filepath: str, external_functions:list):
    
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

    # Collect external inputs and outputs of program through shim tiles
    inputs = set()
    outputs = set()
    for prod, cons, key in dataFlowGraph.edges(keys=True):
        data = dataFlowGraph.edges[prod, cons, key]
        name = data.get('name')
        source_part = name.split('_')[0] if '_' in name else name
        if prod == 'shim':
            inputs.add(source_part)
        elif cons == 'shim':
            outputs.add(source_part)
    sorted_inputs = sorted(inputs)
    sorted_outputs = sorted(outputs)
    argument_names = [f"input{source}" for source in sorted_inputs] + [f"output{source}" for source in sorted_outputs]
    sequential_variables = sorted_inputs + sorted_outputs
    first_input = argument_names[0] if argument_names else None

    # Group input/output FIFOs by source
    input_fifos = defaultdict(list)
    output_fifos = defaultdict(list)
    for prod, cons, key in dataFlowGraph.edges(keys=True):
        data = dataFlowGraph.edges[prod, cons, key]
        name = data.get('name')
        source_part = name.split('_')[0] if '_' in name else name
        fifo_variable = f"of_from_{prod}_to_{cons}_{key}"
        fifo_map[(prod, cons, key)] = fifo_variable
        if prod == 'shim':
            input_fifos[source_part].append(fifo_variable)
        elif cons == 'shim':
            output_fifos[source_part].append(fifo_variable)

    # Iterate through graph nodes to generate tile and worker definitions
    for node in dataFlowGraph.nodes():
        # Extract node data for placement and config
        attributes = dataFlowGraph.nodes[node]
        placement = attributes.get('placement')
        column = placement.column
        row = placement.row
        while_true = attributes.get('while_true')
        stack_size = attributes.get('stack_size')
        allocation_scheme = attributes.get('allocation_scheme')
        trace = attributes.get('trace')
        trace_events = attributes.get('trace_events')
        core_fn = attributes.get('core_fn')
        tile_type = attributes.get('type')


        # write tile generated code to tile_buffer
        tile_variable = f"tile_{column}_{row}"
        tile_buffer.append(f"{tile_variable} = tile({column}, {row})")
        tile_map[node] = tile_variable

        # Process compute nodes
        # Generate external function and core function code
        if( tile_type == 'compute'):
            external_function_def = next((fn for fn in external_functions if fn['name'] == core_fn), None)
            if not external_function_def:
                raise ValueError(f"Unknown core_fn: {core_fn}")
            source_file = external_function_def['source_file']
            include_dirs = external_function_def['include_dirs']
            in_edges = list(dataFlowGraph.in_edges(node, keys=True))
            out_edges = list(dataFlowGraph.out_edges(node, keys=True))
            num_arguments = len(in_edges) + len(out_edges)
            argument_types = external_function_def.get('arg_types', f"[data_ty] * {num_arguments}")

            #generate ExternalFunction definitions
            external_variable = f"external_{node}"
            external_buffer.append(
                f"{external_variable} = ExternalFunction(\n"
                f"    name=\"{core_fn}\",\n"
                f"    source_file=\"{source_file}\",\n"
                f"    arg_types={argument_types},\n"
                f"    include_dirs={include_dirs}\n"
                f")"
            )
            externalfn_map[node] = external_variable

            # Genrate core function definitions, handles acquire and release
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

            function_parameters.append(external_variable)
            worker_args.append(external_variable)

            corefn_buffer.append(f"def {function_name}({', '.join(function_parameters)}):")
            for line in in_acquires + out_acquires:
                corefn_buffer.append(f"{line}")
            corefn_buffer.append(f"    {external_variable}({', '.join(call_args)})")
            for line in in_releases + out_releases:
                corefn_buffer.append(f"{line}")

            # write worker generated code to worker_buffer
            # define workers to execute core functions on specific tiles
            worker_variable = f"worker_{node}"
            worker_buffer.append(
                f"{worker_variable} = Worker({function_name}, [{', '.join(worker_args)}], "
                f"placement={tile_variable}, while_true={while_true}, stack_size={stack_size}, "
                f"allocation_scheme='{allocation_scheme}', trace={trace}, trace_events={trace_events})")
            worker_map[node] = worker_variable

    # Create objectFifo connections between tiles for data streaming
    for prod, cons, key in dataFlowGraph.edges(keys=True):
        data = dataFlowGraph.edges[prod, cons, key]
        depth = data.get('depth')
        name = data.get('name', f"of_{prod}_to{cons}")

        fifo_variable = fifo_map[(prod, cons, key)]

        # attach tiles to objectFifo
        if prod in tile_map and cons in tile_map:
            objectFifo_buffer.append(f"{fifo_variable} = ObjectFifo(data_ty, {tile_map[prod]}, {tile_map[cons]}, depth={depth}, name='{name}')")

        fifo_map[(prod, cons, key)] = fifo_variable

    # extract appropriate info need for runtime environment code and write to runtime_buffer 
    # starting workers, manage input, output data movement
    runtime_buffer.append("rt = Runtime()")
    seq_types = ', '.join(['data_ty'] * len(sequential_variables))
    runtime_buffer.append(f"with rt.sequence({seq_types}) as ({','.join(sequential_variables)}):")
    for worker in worker_map.values():
        runtime_buffer.append(f"    rt.start({worker})")
    for source, fifos in input_fifos.items():
        for fifo_var in fifos:
            runtime_buffer.append(f"    rt.fill({fifo_var}.prod(), {source})")
    for source, fifos in output_fifos.items():
        for fifo_var in fifos:
            runtime_buffer.append(f"    rt.drain({fifo_var}.cons(), {source}, wait=True)")

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
            
            
@jit(is_placed=False)
def generated_design(""" + ", ".join(argument_names) + """):

    element_type = bfloat16
    data_size = """ + (f"{first_input}.numel()" if first_input else "0") + """
    data_ty = np.ndarray[(data_size,), np.dtype[element_type]]
""")
        

        # write tile_buffer to file
        python_file.write("    # Define tiles for compute and shim nodes\n")
        for line in tile_buffer:
            python_file.write("    " + line + '\n')

        # write externalfn_buffer to file
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

        # write objectFifo_buffer to file
        python_file.write("\n    # Define object FIFOs for data streaming between tiles\n")
        for line in objectFifo_buffer:
            python_file.write("    " + line + '\n')

        # write runtime date movement to file
        python_file.write("\n    # Define runtime sequence for starting workers and moving data\n")
        for line in runtime_buffer:
            python_file.write("    " + line + '\n')

        # create and return Program and finish file
        python_file.write(
            """    my_program = Program(iron.get_current_device(), rt)
    my_program = my_program.resolve_program(SequentialPlacer())
    return my_program

def main():
    datatype = bfloat16
    data_size = 256
""")
        
        # write input and output sources to file
        for source in sorted_inputs:
            python_file.write(f"    input{source} = iron.arange(data_size, dtype=datatype, device=\"npu\")\n")
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
    
def main():
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
        }
    ]

    # Create graph equivalent of base_aaa.py add-activate-add design to compare generated code
    dataFlowGraph = nx.MultiDiGraph()
    dataFlowGraph.add_node('shim', placement=Placement(0, 0), type='shim')
    dataFlowGraph.add_node('CT1', placement=Placement(0, 2), while_true=False, stack_size=1024,
                           allocation_scheme='heap', trace=False, trace_events=None,
                           core_fn='eltwise_add_bf16_scalar', type='compute')
    dataFlowGraph.add_node('CT2', placement=Placement(0, 3), while_true=False, stack_size=1024,
                           allocation_scheme='heap', trace=False, trace_events=None,
                           core_fn='bf16_relu', type='compute')
    dataFlowGraph.add_node('CT3', placement=Placement(0, 4), while_true=False, stack_size=1024,
                           allocation_scheme='heap', trace=False, trace_events=None,
                           core_fn='eltwise_add_bf16_scalar', type='compute')
    dataFlowGraph.add_edge('shim', 'CT1', depth=1, name='A_L3_L1_CT1', obj_type='data_ty')
    dataFlowGraph.add_edge('shim', 'CT1', depth=1, name='B_L3_L1', obj_type='data_ty')
    dataFlowGraph.add_edge('CT1', 'CT2', depth=1, name='tempC_CT1_CT2', obj_type='data_ty')
    dataFlowGraph.add_edge('shim', 'CT3', depth=1, name='A_L3_L1_CT3', obj_type='data_ty')
    dataFlowGraph.add_edge('CT2', 'CT3', depth=1, name='tempC_CT2_CT3', obj_type='data_ty')
    dataFlowGraph.add_edge('CT3', 'shim', depth=1, name='C_L1_L3', obj_type='data_ty')

    # Generate the code
    generated_file = generateIronCode(dataFlowGraph, 'generated_design.py', external_functions)
    print(f"Generated IRON code at: {generated_file}")


# Call main
if __name__ == "__main__":
    main()



    

