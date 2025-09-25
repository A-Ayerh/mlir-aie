

# Generator function that will take the dataFlowGraph and the new python file as arguments, 
# will iterate through the graph nodes and edges, and write the corresponding IRON code to 
# the file

def generateIronCode(graph: dataFlowGraph, filepath: pythonFile):
    
    # create buffers for code insertions
    worker_buffer = []
    tile_buffer = []
    corefn_buffer = []
    objectFifo_buffer = []


    
    for node in dataFlowGraph.nodes():
        # generate worker code to worker_buffer
        placement = node.get('placement')
        column = placement.column
        row = placement.row
        while_true = node.get('while_true')
        stack_size = node.get('stack_size')
        allocation_scheme = node.get('allocation_scheme')
        trace = node.get('trace')
        trace_events = node.get('trace_events')

        # write tile generated code to tile_buffer
        tile_variable = f"tile_{column}_{row}"
        tile_buffer.append(f"{tile_variable} = tile({column}, {row})")

        # write worker generated code to worker_buffer
        
        # generate core function code to corefn_buffer
    
    for edge in dataFlowGraph.edges():
        # generate objectFifo code to objectFifo_buffer

    # open file

    # write imports and header code to file

    # write objectFifo_buffer to file

    # write corefn_buffer to file

    # write runtime date movement to file

    # create and return Program and finish file

    # close file



    

