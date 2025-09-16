# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 AMD Inc.

# Hardoding the matrix multiplication and cascade streams to foolow: 4x4x4 micro MNK dimensions.
# Matrix multiplication kernel inspired from: https://xilinx.github.io/aie_api/group__group__mmul.html#structaie_1_1mmul
# cascading results inspired from mm_cascade example and kernel: 
# - Kernel: mlir-aie/aie_kernels/aie2/cascade_mm.cc
# - Python: mlir-aie/programming_examples/basic/matrix_multiplication/cascade/cascade.py


import numpy as np

import aie.iron as iron
from aie.iron import ExternalFunction, jit
from aie.iron import ObjectFifo, Worker, Runtime, Program
from aie.iron.placers import SequentialPlacer
from aie.iron.controlflow import range_
from aie.iron.device import Tile


@jit(is_placed=False)
def transform_with_internal_func_from_file(input1, input2, output):
    num_elements = np.size(input1)
    data_size = input1.numel()
    element_type = input1.dtype
    data_ty = np.ndarray[(data_size,), np.dtype[element_type]]

    m = 32
    n = 32
    k = 32

    # a_ty = np.ndarray[(8,), np.dtype[np.int32]]
    # b_ty = np.ndarray[(8,), np.dtype[np.int32]]
    # c_ty = np.ndarray[(8,), np.dtype[np.int32]]
    # Create ExternalFunction inside the transform from a file
    internal_func = ExternalFunction(
        "internal_add_from_file",
        source_file="./add.cc",
        arg_types=[
            data_ty,
            data_ty,
            data_ty,
            np.int32,
        ],
    )
    matmul_scalar_cascade_get_only = ExternalFunction(
        "matmul_scalar_cascade_get_only",
        source_file="./add.cc",
        arg_types=[
            data_ty,
            data_ty,
            data_ty,
        ],
    )
    matmul_scalar_cascade_put_only = ExternalFunction(
        "matmul_scalar_cascade_put_only",
        source_file="./add.cc",
        arg_types=[
            data_ty,
            data_ty,
            data_ty,
            np.int32,
        ],
    )
    matmul_scalar_cascade_put_get = ExternalFunction(
        "matmul_scalar_cascade_put_get",
        source_file="./add.cc",
        arg_types=[
            data_ty,
            data_ty,
            data_ty,
            np.int32,
        ],
    )

    # Extract tile size from ExternalFunction
    tile_size = internal_func.tile_size(0)

    # AIE-array data movement with object fifos
    # of_in = ObjectFifo(tile_ty, name="in")
    # of_out = ObjectFifo(tile_ty, name="out")
    # Input A
    inA = ObjectFifo(data_ty, name="inA")
    a_dims = None
    memA = inA.cons().forward(placement=Tile(0,2), name="memA", dims_to_stream=a_dims)

    # Input B
    inB = ObjectFifo(data_ty, name="inB")
    b_dims = None
    memB = inB.cons().forward(placement=Tile(1,2), name="memB", dims_to_stream=b_dims)

    # Output C
    memC = ObjectFifo(data_ty, name="memC")
    c_dims = None
    outC = memC.cons().forward(placement=Tile(0,2), name="outC", dims_to_stream=c_dims)

    # Define a task that will run on a compute tile
    def core_put_only(of_in_a, of_in_b, matmul_scalar_cascade_put_only):
            elem_in_a = of_in_a.acquire(1)
            elem_in_b = of_in_b.acquire(1)
            matmul_scalar_cascade_put_only(elem_in_a, elem_in_b)
            of_in_a.release(1)
            of_in_b.release(1)

    def core_put_get(of_in_a, of_in_b, matmul_scalar_cascade_put_get):
        for row in range_(4):
            elem_in_a = of_in_a.acquire(1)
            elem_in_b = of_in_b.acquire(1)
            matmul_scalar_cascade_put_get(elem_in_a, elem_in_b)
            of_in_a.release(1)
            of_in_b.release(1)

    def core_get_only(of_in_a, of_in_b, of_out_c, matmul_scalar_cascade_get_only):
        for row in range_(4):
            elem_out = of_out_c.acquire(1)
            elem_in_a = of_in_a.acquire(1)
            elem_in_b = of_in_b.acquire(1)
            matmul_scalar_cascade_get_only(elem_in_a, elem_in_b, elem_out)
            of_in_a.release(1)
            of_in_b.release(1)
        of_out_c.release(1)

    workers = [
        Worker(
            core_put_only,
            [memA.cons(), memB.cons(), matmul_scalar_cascade_put_only],
            placement=Tile(0, 2),
        ),
        Worker(
            core_put_get,
            [memA.cons(), memB.cons(), matmul_scalar_cascade_put_get],
            placement=Tile(0, 3),
        ),
        Worker(
            core_get_only,
            [memA.cons(), memB.cons(), memC.prod(), matmul_scalar_cascade_get_only],
            placement=Tile(0, 4),
        ),
    ]
    
    # Runtime operations to move data to/from the AIE-array
    rt = Runtime()
    with rt.sequence(data_ty, data_ty, data_ty) as (A, B, C):
        rt.start(*workers)
        rt.fill(inA.prod(), A)
        rt.fill(inB.prod(), B)

        rt.drain(outC.cons(), C, wait=True)

    # Place program components and generate an MLIR module
    return Program(iron.get_current_device(), rt).resolve_program(SequentialPlacer())

def main():
    input1 = iron.arange(48, dtype=np.int32, device="npu")
    input2 = iron.arange(48, dtype=np.int32, device="npu")

    output = iron.zeros_like(input1)

    transform_with_internal_func_from_file(input1, input2, output)
    print(output)

if __name__ == "__main__":
    main()