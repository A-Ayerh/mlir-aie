import networkx as nx

# Generator function that will take the dataFlowGraph and the new python file as arguments, 
# will iterate through the graph nodes and edges, and write the corresponding IRON code to 
# the file

def generateIronCode(dataFlowGraph: nx.DiGraph, filepath: str):
    
    # create buffers for code insertions
    worker_buffer = []
    tile_buffer = []
    corefn_buffer = []
    objectFifo_buffer = []
    runtime_buffer = []

    # Maps for variables (to be able to reference later)
    tile_map = {}
    worker_map = {}
    corefn_map = {}

    for node in dataFlowGraph.nodes():
        # generate worker code to worker_buffer
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

        if( tile_type == 'compute'):
            # generate core function code to corefn_buffer
            core_variable = f"core_{node}"
            corefn_buffer.append(f"{core_variable} = ExternalFunction('{core_fn}')")
            corefn_map[node] = core_variable
            
            # write worker generated code to worker_buffer
            worker_variable = f"worker_{node}"
            worker_buffer.append(f"{worker_variable} = Worker({core_variable}, [], placement={tile_variable}, while_true={while_true}, stack_size={stack_size}, allocation_scheme='{allocation_scheme}', trace={trace}, trace_events={trace_events})")
            worker_map[node] = worker_variable

    for prod, cons in dataFlowGraph.edges():
        # generate objectFifo code to objectFifo_buffer
        data = graph.edges[prod, cons]
        depth = data.get('depth')
        name = data.get('name')
        obj_type = data.get('obj_type')

        fifo_variable = f"of_name_from_{prod}_to_{cons}"
        objectFifo_buffer.append(f"{fifo_variable} = ObjectFifo({obj_type}, depth={depth}, name={fifo_variable})")

        # need to add core_fn arguments to object fifo

    # extract appropriate info need for runtime environment code and write to runtime_buffer 
        # starting workers, etc

    # open file
    with open(filepath, 'w') as python_file:
        # write imports and header code to file
        python_file.write(
            """import aie.iron as iron
            from aie.iron import ExternalFunction, jit
            from aie.iron import ObjectFifo, Worker, Runtime, Program
            from aie.iron.placers import SequentialPlacer
            
            
            @jit(is_placed=False)""")
        
        # write tile_buffer to file
        for line in tile_buffer:
            python_file.write(line + '\n')

        # write corefn_buffer to file
        for line in corefn_buffer:
            python_file.write(line + '\n')

        # write worker_buffer to file
        for line in worker_buffer:
            python_file.write(line + '\n')

        # write objectFifo_buffer to file
        for line in objectFifo_buffer:
            python_file.write(line + '\n')

        # write runtime date movement to file
        for line in runtime_buffer:
            python_file.write(line + '\n')

        # create and return Program and finish file

        # close file and return
        return filepath



    

