# exercise_1.py -*- Python -*- Updated by Brock Sorenson
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc. or its affiliates


# Code Changes: 
# Bring in an input array (input0) and use a memory tile to DISTRIBUTE
# Split the array for different compute tiles 
# Process the data on 3 compute tiles (3 workers) 
# Combine using JOIN into an output array (output0)
# Workers copy segment of input array into output array
import sys
import numpy as np

from aie.iron import Program, Runtime, Worker, ObjectFifo, LocalBuffer
from aie.iron.placers import Placer, SequentialPlacer
from aie.iron.controlflow import range_
from aie.iron import ExternalFunction
from aie.iron.device import AnyComputeTile, AnyMemTile, AnyShimTile, Tile
from aie.iron.dataflow import ObjectFifoHandle, ObjectFifoLink, ObjectFifoEndpoint

import aie.iron as iron


@iron.jit(is_placed=False)
def exercise_1(input0, output0): #add input 0
    data_size = input0.numel() #data size received from input argument
    element_type = input0.dtype #data type received from input argument
    data_ty = np.ndarray[(data_size,), np.dtype[element_type]]

    #Compute Tiles (workers)
    num_workers = 3 #set number of compute tiles to use (number of workers)
    vector_length = 16 # int32 vector length (512 bit vector unit)
    if data_size % (num_workers * vector_length) != 0: #check to see if data is divisible by num_workers * vector_length
        raise ValueError("input data_size must be divisible by num_workers * vector_length")
    worker_data_size = data_size // num_workers #compute array elements per worker
    worker_vector_size = worker_data_size // vector_length # number of vector operations for each worker
    worker_data_type = np.ndarray[(worker_data_size,), np.dtype[element_type]] #data type of each split segment
    
    #ObjectFIFO Offsets for splitting and joining
    of_offsets = [i * worker_data_size for i in range(num_workers)] #array of offsets of starting indices for the ObjectFIFOS ie [0,16,32]
    

    # ObjectFifo to input from runtime
    of_in = ObjectFifo(data_ty, name="in") #receive full input array from runtime

    # ObjectFifo to forward through memory tile for distribution
    #of_mem = ObjectFifo(data_ty, name="mem")  # New ObjectFifo for memory tile
    #of_in.cons().forward_to(of_mem.prod())
    #of_mem = of_in.cons().forward() #input data forwarded through memory tile

    # Split the input fifo for distribution to multiple compute tiles
    # Splits the memory tile ObjectFifo into 3 ObjectFifos which move the split arrays
    # Each has name in_0, in_1, ...
    of_ins = of_in.cons().split(of_offsets, obj_types=[worker_data_type] * num_workers, names=[f"in_{i}" for i in range(num_workers)])
    
    # ObjectFifo to output to runtime
    of_out = ObjectFifo(data_ty, name="out")

    # ObjectFifo to join outputs of multiple workers
    # Creates a join structure combining outputs from the 3 worker tiles
    # Each worker outputs a smaller array, offsets ensure array is in proper order
    # Each has name out_o, out_1, ...
    of_outs = of_out.prod().join(of_offsets, obj_types=[worker_data_type] * num_workers, names=[f"out_{i}" for i in range(num_workers)])

    # Define ExternalFunction that performs vectorized multiplication
    vector_multiplication = ExternalFunction(
            "internal_multiply",
            source_file="./multiply.cc",
            arg_types=[
                np.ndarray[(vector_length,), np.dtype[np.int32]],
                np.ndarray[(vector_length,), np.dtype[np.int32]],
                np.int32,
                np.int32
            ]
    )

    # Task for the core to perform
    def core_fn(of_in, of_out, multiplier, function_to_apply): # takes 2 arguments (input objectFifo of worker segment and output objectFifo)
        elem_in = of_in.acquire(1) 
        elem_out = of_out.acquire(1)
        function_to_apply(elem_in, elem_out, vector_length, multiplier)
        of_in.release(1) 
        of_out.release(1)

    # Replace single worker with a list of num_workers for each compute tile
    # Each worker is assigned core_fn and its corresponding split FIFO and join FIFO
    # Allows the array to be parrallel proccessed acrossed num_worker compute tiles each handling n/num_worker elements of the array
    # Each worker will copy the input array into the output array, multiplying each index by their worker id
    workers = []
    for i in range(num_workers):
        worker = Worker(core_fn, [of_ins[i].cons(), of_outs[i].prod(), i+1, vector_multiplication])
        workers.append(worker)

    # Custom placer class for manual tile assignment
    class CustomPlacer(Placer): # Inherits from Placer class, must implement "make_placement(device, rt, workers, object_fifos)"
        def make_placement(self, device, rt, workers, object_fifos):
            # Resolve tiles for MemTile (0,1) and ComputeTiles (0,2), (0,3), (0,4)
            shim_tile = Tile(0,0) # use shim_tile on (0,0)
            mem_tile = Tile(0, 1) # use mem_tile on (0,1)
            device.resolve_tile(shim_tile) # allows .place operations
            device.resolve_tile(mem_tile) 
            compute_tiles = [
                Tile(0, 2),
                Tile(0, 3),
                Tile(0, 4)
            ]
            for tile in compute_tiles:
                device.resolve_tile(tile)

            # Assign workers to compute tiles: (0,2), (0,3), (0,4)
            for i, worker in enumerate(workers): # place workers to compute tiles
                worker.place(compute_tiles[i])
                for buffer in worker.buffers:
                    buffer.place(compute_tiles[i])

            #Track placed endpoints to avoid duplicates
            placed_endpoints = set()
            # Assign ObjectFifo endpoints to MemTile (0,1), skipping Worker endpoints
            for ofh in object_fifos:
                if ofh.name == "in": 
                    # Place of_in producer on ShimTile (0,0), consumer on MemTile (0,1)
                    prod_endpoint = ofh._object_fifo._get_endpoint(is_prod=True)
                    for ep in prod_endpoint:
                        if id(ep) not in placed_endpoints:
                            ep.place(shim_tile)
                            placed_endpoints.add(id(ep))
                    cons_endpoint = ofh._object_fifo._get_endpoint(is_prod=False)
                    for ep in cons_endpoint:
                        if id(ep) not in placed_endpoints and not isinstance(ep, Worker):
                            ep.place(mem_tile)
                            placed_endpoints.add(id(ep))
                elif ofh.name == "out":
                    # Place of_out consumer on ShimTile (0,0), producer on MemTile (0,1)
                    cons_endpoint = ofh._object_fifo._get_endpoint(is_prod=False)
                    for ep in cons_endpoint:
                        if id(ep) not in placed_endpoints:
                            ep.place(shim_tile)
                            placed_endpoints.add(id(ep))
                    prod_endpoint = ofh._object_fifo._get_endpoint(is_prod=True)
                    for ep in prod_endpoint:
                        if id(ep) not in placed_endpoints and not isinstance(ep, Worker):
                            ep.place(mem_tile)
                            placed_endpoints.add(id(ep))
                else:
                    # Place other ObjectFifo endpoints (e.g., in_0, in_1, in_2, out_0, out_1, out_2) on MemTile (0,1)
                    for ep in ofh.all_of_endpoints():
                        if id(ep) not in placed_endpoints and not isinstance(ep, Worker):
                            ep.place(mem_tile)
                            placed_endpoints.add(id(ep))

            return device

    # To/from AIE-array runtime data movement
    # Handles two tensors c_in and c_out
    # Added rt.fill to transfer input tensor (input0) to input ObjectFifo (of_in)
    # Wait ensures all workers finish before draining output to output array
    rt = Runtime() # Implicitily handles shim tile not included in my_custom_placer
    with rt.sequence(data_ty, data_ty) as (c_in, c_out):
        for worker in workers:
            rt.start(worker)
        rt.fill(of_in.prod(), c_in, wait=True) # transfer input (input0) tensor into input ObjectFifo
        rt.drain(of_out.cons(), c_out, wait=True) # transfer joined output from of_out to output tensor

    # Create the program from the device type and runtime
    my_program = Program(iron.get_current_device(), rt)

    # Place components (assign them resources on the device) and generate an MLIR module
    placer = CustomPlacer()
    return my_program.resolve_program(placer)


def main():
    # Define tensor shapes and data types
    num_workers = 3
    data_size = 48
    element_type = np.int32
    worker_data_size = data_size // num_workers

    # Construct an input tensor and an output zeroed tensor
    # The two tensors are in memory accessible to the NPU
    input0 = iron.arange(data_size, dtype=element_type, device="npu")
    output0 = iron.zeros_like(input0)

    # JIT-compile the kernel then launches the kernel with the given arguments. Future calls
    # to the kernel will use the same compiled kernel and loaded code objects
    exercise_1(input0, output0) # update to pass input0 array

    # Expected verification output
    expected = np.zeros(data_size, dtype=element_type)
    for i in range(num_workers):
        starting_index = i * worker_data_size
        ending_index = (i+1) * worker_data_size
        expected[starting_index:ending_index] = input0[starting_index:ending_index] * (i+1)

    # Check the correctness of the result
    e = np.equal(expected, output0.numpy())
    errors = np.size(e) - np.count_nonzero(e)

    # Print the results
    print(f"{'input0':>8} {'output0':>8} {'expected':>8}")
    print("-" * 34)
    count = input0.numel()
    for idx, (a, c, exp) in enumerate(zip(input0[:count], output0[:count], expected[:count])):
        print(f"{idx:2}: {a:6} {c:8} {exp:8}")

    # If the result is correct, exit with a success code.
    # Otherwise, exit with a failure code
    if not errors:
        print("\nPASS!\n")
        sys.exit(0)
    else:
        print("\nError count: ", errors)
        print("\nfailed.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
