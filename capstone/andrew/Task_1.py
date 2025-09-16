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
def exercise_5a(input0, output):
    data_size = input0.numel()
    element_type = input0.dtype

    data_ty = np.ndarray[(data_size,), np.dtype[element_type]]

    # Dataflow with ObjectFifos
    # of_in = ObjectFifo(data_ty, depth=3, name="in")
    # of_out = ObjectFifo(data_ty, name="out")

    #for memtiles:
    of_in = ObjectFifo(data_ty, name="in")
    of_in_mem = of_in.cons().forward(name="in_mem")

    # Task for the core to perform
    def core_fn(of_in, of_out):
        elem_in = of_in.acquire(1)
        elem_out = of_out.acquire(1)
        for i in range_(data_size):
            elem_out[i] += elem_in[i]
        of_in.release(1)
        of_out.release(1)

    # Create a worker to perform the task
    # my_worker = Worker(core_fn, [of_in.cons(), of_out.prod()])

    # To/from AIE-array runtime data movement
    rt = Runtime()
    with rt.sequence(data_ty, data_ty) as (a_in, c_out):
        #rt.start(my_worker)
        rt.fill(of_in.prod(), a_in)
        rt.drain(of_in_mem.cons(), c_out, wait=True)

    # Create the program from the device type and runtime
    my_program = Program(iron.get_current_device(), rt)

    # Place components (assign them resources on the device) and generate an MLIR module
    my_program = my_program.resolve_program(SequentialPlacer())

    # print(my_program)

    return my_program


def main():
    # Define tensor shapes and data types
    data_size = (2**12)-64 # size max for compute tiles streaming in/out int32 and using ping pong buffer for in/out

    # mem data size cap:
    data_size = (2**17) -1 # size max for mem tiles : 524284 bytes (4 bytes away from 2**19)
    element_type = np.int32

    input0 = iron.arange(data_size, dtype=element_type, device="npu")
    output = iron.zeros(data_size, dtype=element_type, device="npu")
    
    exercise_5a(input0, output)

    print(output)


if __name__ == "__main__":
    main()