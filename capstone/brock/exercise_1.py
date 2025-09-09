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
from aie.iron.placers import SequentialPlacer
from aie.iron.controlflow import range_

import aie.iron as iron


@iron.jit(is_placed=False)
def exercise_1(input0, output0): #add input 0
    data_size = input0.numel() #data size received from input argument
    element_type = input0.dtype #data type received from input argument
    data_ty = np.ndarray[(data_size,), np.dtype[element_type]]

    #Compute Tiles (workers)
    num_workers = 3 #set number of compute tiles to use (number of workers)
    if data_size % num_workers != 0: #check to see if data is divisible by num_workers
        raise ValueError("input data_size must be divisible by 3")
    worker_data_size = data_size // num_workers #compute array elements per worker
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

    # Task for the core to perform
    def core_fn(of_in, of_out, multiplier): # takes 2 arguments (input objectFifo of worker segment and output objectFifo)
        elem_in = of_in.acquire(1) 
        elem_out = of_out.acquire(1)
        for i in range_(worker_data_size): # loops over smaller data size receiving from split (of_ins[i]) and writing to join (of_outs[i])
            elem_out[i] = multiplier * elem_in[i] # copies to out array
        of_in.release(1) 
        of_out.release(1)

    # Replace single worker with a list of num_workers for each compute tile
    # Each worker is assigned core_fn and its corresponding split FIFO and join FIFO
    # Allows the array to be parrallel proccessed acrossed num_worker compute tiles each handling n/num_worker elements of the array
    # Each worker will copy the input array into the output array, multiplying each index by their worker id
    workers = []
    for i in range(num_workers):
        worker = Worker(core_fn, [of_ins[i].cons(), of_outs[i].prod(), i+1])
        workers.append(worker)


    # To/from AIE-array runtime data movement
    # Handles two tensors c_in and c_out
    # Added rt.fill to transfer input tensor (input0) to input ObjectFifo (of_in)
    # Wait ensures all workers finish before draining output to output array
    rt = Runtime()
    with rt.sequence(data_ty, data_ty) as (c_in, c_out):
        for worker in workers:
            rt.start(worker)
        rt.fill(of_in.prod(), c_in, wait=True) # transfer input (input0) tensor into input ObjectFifo
        rt.drain(of_out.cons(), c_out, wait=True) # transfer joined output from of_out to output tensor

    # Create the program from the device type and runtime
    my_program = Program(iron.get_current_device(), rt)

    # Place components (assign them resources on the device) and generate an MLIR module
    return my_program.resolve_program(SequentialPlacer())


def main():
    # Define tensor shapes and data types
    data_size = 48
    element_type = np.int32

    # Construct an input tensor and an output zeroed tensor
    # The two tensors are in memory accessible to the NPU
    input0 = iron.arange(data_size, dtype=element_type, device="npu")
    output0 = iron.zeros_like(input0)

    # JIT-compile the kernel then launches the kernel with the given arguments. Future calls
    # to the kernel will use the same compiled kernel and loaded code objects
    exercise_1(input0, output0) # update to pass input0 array

    # Check the correctness of the result
    e = np.equal(input0.numpy(), output0.numpy())
    errors = np.size(e) - np.count_nonzero(e)

    # Print the results
    print(f"{'input0':>4} = {'output0':>4}")
    print("-" * 34)
    count = input0.numel()
    for idx, (a, c) in enumerate(zip(input0[:count], output0[:count])):
        print(f"{idx:2}: {a:4} = {c:4}")

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
