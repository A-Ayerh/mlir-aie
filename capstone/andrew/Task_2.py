# answer_3.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc. or its affiliates

import sys
import numpy as np

from aie.iron import Program, Runtime, Worker, ObjectFifo
from aie.iron.placers import SequentialPlacer
from aie.iron.controlflow import range_
from aie.iron.device.tile import AnyComputeTile
from aie.helpers.taplib.tap import TensorAccessPattern
import aie.iron as iron

@iron.jit(is_placed=False)
def exercise_5a(input0, output, l3_l2, l3_l1, l2_l3, l2_l1, l1_l2, l1_l3):
    data_size = input0.numel()
    element_type = input0.dtype

    data_ty = np.ndarray[(data_size,), np.dtype[element_type]]
    
    #TODO:generate object fifos to handle amount of data:
    #Shim -> Mem tiles
    if(l3_l2):
        l3_l2_fifos = []
        l3_l2_temp = []
        numMemFifos = 1
        if data_size//(((2**19) * 4) - 1) > 1:#Size of mem tile memory
            numMemFifos = (data_size//(((2**19) * 4) - 1))
        if numMemFifos > 4:
            print("datasize exceeds memory shared across L2")
            exit
        for i in range(numMemFifos):
            fifo = ObjectFifo(data_ty, name="l3_l2_fifos{i}")
            temp = fifo.cons().forward(name="l3_l2_temp{i}")
            l3_l2_fifos.append(fifo)
            l3_l2_temp.append(temp)
        
    #Mem tiles -> shim
    if(l2_l3):
        l2_l3_fifos = []
        l2_l3_temp = []
        numMemFifos = 1
        if data_size//(((2**19) * 4) - 1) > 1:#Size of mem tile memory
            numMemFifos = (data_size//(((2**19) * 4) - 1))
        if numMemFifos > 4:
            print("datasize exceeds memory shared across L2")
            exit
        for i in range(numMemFifos):
            fifo = ObjectFifo(data_ty, name="l2_l3_fifos{i}")
            temp = fifo.cons().forward(name="l2_l3_temp{i}")
            l2_l3_fifos.append(fifo)
            l2_l3_temp.append(temp)

    # Shim -> Compute tile
    if(l3_l1):
        l3_l1_fifos = []
        numCTFifos = 1
        if data_size//(((2**12) * 4) - 64) > 1:#Size of compute tile memory
            numCTFifos = (data_size//(((2**12) * 4) - 64))
        if numCTFifos > 16:
            print("datasize exceeds memory shared across L1")
            exit
        for i in range(numCTFifos):
            fifo = ObjectFifo(data_ty, name="l3_l1_fifos{i}")
            l3_l1_fifos.append(fifo)

    # Compute tile -> Shim
    if(l1_l3):
        l1_l3_fifos = []
        numCTFifos = 1
        if data_size//(((2**12) * 4) - 64) > 1:#Size of compute tile memory
            numCTFifos = (data_size//(((2**12) * 4) - 64))
        if numCTFifos > 16:
            print("datasize exceeds memory shared across L1")
            exit
        for i in range(numCTFifos):
            fifo = ObjectFifo(data_ty, name="l1_l3_fifos{i}")
            l1_l3_fifos.append(fifo)

    #Mem tile -> Compute tile: 4 mem tiles -> 16 compute tiles
    #need to have fifos for each memtile (64 fifos)
    if(l2_l1):
        l2_l1_fifos = []
        numCTFifos = 1
        if data_size//(((2**12) * 4) - 64) > 1:#Size of compute tile memory
            numCTFifos = (data_size//(((2**12) * 4) - 64))
        if numCTFifos > 16:
            print("datasize exceeds memory shared across L1")
            exit
        for i in range(numCTFifos):
            fifo = ObjectFifo(data_ty, name="l1_l3_fifos{i}")
            l2_l1_fifos.append(fifo)

    #Compute Tile -> Mem tile:
    if(l1_l2):
        l1_l2_fifos = []
        numCTFifos = 1
        if data_size//(((2**19) * 4) - 1) > 1:#Size of compute tile memory
            numCTFifos = (data_size//(((2**12) * 4) - 64))
        if numCTFifos > 4:
            print("datasize exceeds memory shared across L2")
            exit
        for i in range(numCTFifos):
            fifo = ObjectFifo(data_ty, name="l1_l3_fifos{i}")
            l1_l2_fifos.append(fifo)
    

    
     # Dataflow with ObjectFifos
    # of_in = ObjectFifo(data_ty, name="in")
    # of_in_mem = of_in.cons().forward(name="in_mem")

    of_out_mem = ObjectFifo(data_ty, name="out")
    of_out = of_out_mem.cons().forward(name="out_mem")
    

    # Task for the core to perform
    def core_fn(of_in, of_out):
        elem_in = of_in.acquire(1)
        elem_out = of_out.acquire(1)
        for i in range_(data_size):
            elem_out[i] = elem_in[i]
        of_out.release(1)
        of_in.release(1)

    # Create a worker to perform the task
    #my_worker = Worker(core_fn, [of_in_mem.cons(), of_out_mem.prod()])

    # To/from AIE-array runtime data movement
    #TODO: Need to split up data around weach memtile, the objectfifo generation works now, but the data is all being sent to the first mem fifo
    rt = Runtime()
    with rt.sequence(data_ty, data_ty) as (a_in, c_out):
        # rt.start(my_worker)
        for i in range(numMemFifos):
            if(l3_l1):
                rt.fill(l3_l1_fifos[i].prod(), a_in)
            if(l3_l2):
                rt.fill(l3_l2_temp[i].prod(), a_in)
            if(l1_l3):
                rt.drain(l1_l3_fifos[i].cons(), c_out, wait=True)
            if(l2_l3):
                rt.drain(l2_l3_temp[i].cons(), c_out, wait=True)
            
            

    # Create the program from the device type and runtime
    my_program = Program(iron.get_current_device(), rt)

    # Place components (assign them resources on the device) and generate an MLIR module
    
    my_program=  my_program.resolve_program(SequentialPlacer())
    #print(my_program)

    return my_program


def main():
    # Define tensor shapes and data types
    data_size = (2**19)-64 # size max for compute tiles streaming in/out int32 and using ping pong buffer for in/out
    element_type = np.int32

    input0 = iron.arange(data_size, dtype=element_type, device="npu")
    output = iron.zeros(data_size, dtype=element_type, device="npu")

    l3_l2 = False
    l3_l1 = True
    l2_l3 = False
    l2_l1 = False
    l1_l2 = False
    l1_l3 = True
    
    exercise_5a(input0, output, l3_l2, l3_l1, l2_l3, l2_l1, l1_l2, l1_l3)

    print(output)


if __name__ == "__main__":
    main()
