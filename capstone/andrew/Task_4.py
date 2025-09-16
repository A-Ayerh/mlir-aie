# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 AMD Inc.

# RUN: %run_on_npu1% %pytest %s
# RUN: %run_on_npu2% %pytest %s

import numpy as np
import tempfile

import aie.iron as iron
from aie.iron import ExternalFunction, jit
from aie.iron import ObjectFifo, Worker, Runtime, Program
from aie.iron.placers import SequentialPlacer
from aie.iron.controlflow import range_


@jit(is_placed=False)
def transform_with_internal_func_from_file(input, output):
    """Transform kernel that creates ExternalFunction internally from a file."""
    if input.shape != output.shape:
        raise ValueError(
            f"Input shapes are not the equal ({input.shape} != {output.shape})."
        )
    num_elements = np.size(input)

    # Create ExternalFunction inside the transform from a file
    internal_func = ExternalFunction(
        "internal_add_from_file",
        source_file="./add.cc",
        arg_types=[
            np.ndarray[(16,), np.dtype[np.int32]],
            np.ndarray[(16,), np.dtype[np.int32]],
            np.int32,
        ],
    )

    # Extract tile size from ExternalFunction
    tile_size = internal_func.tile_size(0)

    if num_elements % tile_size != 0:
        raise ValueError(
            f"Number of elements ({num_elements}) must be a multiple of {tile_size}."
        )
    num_tiles = num_elements // tile_size

    if input.dtype != output.dtype:
        raise ValueError(
            f"Input data types are not the same ({input.dtype} != {output.dtype})."
        )

    dtype = input.dtype

    # Define tensor types
    tensor_ty = np.ndarray[(num_elements,), np.dtype[dtype]]
    tile_ty = np.ndarray[(tile_size,), np.dtype[dtype]]

    # AIE-array data movement with object fifos
    of_in = ObjectFifo(tile_ty, name="in")
    of_out = ObjectFifo(tile_ty, name="out")

    # Define a task that will run on a compute tile
    def core_body(of_in, of_out, func_to_apply):
        # Extract tile size from ExternalFunction
        tile_size = func_to_apply.tile_size(0)

        # Number of sub-vector "tile" iterations
        for i in range_(num_tiles):
            elem_in = of_in.acquire(1)
            elem_out = of_out.acquire(1)
            func_to_apply(elem_in, elem_out, tile_size)
            of_in.release(1)
            of_out.release(1)

    # Create a worker to run the task on a compute tile
    worker = Worker(core_body, fn_args=[of_in.cons(), of_out.prod(), internal_func])

    # Runtime operations to move data to/from the AIE-array
    rt = Runtime()
    with rt.sequence(tensor_ty, tensor_ty) as (A, B):
        rt.start(worker)
        rt.fill(of_in.prod(), A)
        rt.drain(of_out.cons(), B, wait=True)

    # Place program components and generate an MLIR module
    return Program(iron.get_current_device(), rt).resolve_program(SequentialPlacer())

def main():
    input0 = iron.arange(48, dtype=np.int32, device="npu")

    output = iron.zeros_like(input0)

    transform_with_internal_func_from_file(input0, output)
    print(output)

if __name__ == "__main__":
    main()