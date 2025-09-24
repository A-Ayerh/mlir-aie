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

    # Systolic Array Dimensions:
    n_aie_rows_A = 4
    n_aie_cols_B = 4
    n_aie_cols_C = 4

    A_tiles = [(0, 2), (0, 3), (0, 4), (0, 5)] 
    B_tiles = [(0, 5), (1, 5), (2, 5), (3, 5)]
    C_tiles = [(0, 2), (1, 2), (2, 2), (3, 2)]

    m, k, n = 64, 64, 64
    dtype_in = np.int32
    dtype_out = np.int32
    A_l2_ty = np.ndarray[(m * k), np.dtype[dtype_in]]
    B_l2_ty = np.ndarray[(k * n), np.dtype[dtype_in]]
    C_l2_ty = np.ndarray[(m * n), np.dtype[dtype_out]]
    A_l1_ty = np.ndarray[(m, k), np.dtype[dtype_in]]
    B_l1_ty = np.ndarray[(k, n), np.dtype[dtype_in]]
    C_l1_ty = np.ndarray[(m, n), np.dtype[dtype_out]]

    num_elements = np.size(input1)
    data_size = input1.numel()
    element_type = input1.dtype
    data_ty = np.ndarray[(data_size,), np.dtype[element_type]]

    # External functions
    matmul_scalar_cascade_get_only = ExternalFunction(
        "matmul_scalar_cascade_get_only_i16_i16",
        source_file="./add.cc",
        arg_types=[data_ty, data_ty, data_ty],
    )
    matmul_scalar_cascade_put_only = ExternalFunction(
        "matmul_scalar_cascade_put_only_i16_i16",
        source_file="./add.cc",
        arg_types=[data_ty, data_ty],
    )
    matmul_scalar_cascade_put_get = ExternalFunction(
        "matmul_scalar_cascade_put_get_i16_i16",
        source_file="./add.cc",
        arg_types=[data_ty, data_ty],
    )

    A_l3l2_0 = ObjectFifo(A_l2_ty, name="A_L3L2_0")
    A_l3l2_1 = ObjectFifo(A_l2_ty, name="A_L3L2_1")
    A_l3l2_2 = ObjectFifo(A_l2_ty, name="A_L3L2_2")
    A_l3l2_3 = ObjectFifo(A_l2_ty, name="A_L3L2_3")

    A_l2l1_0 = A_l3l2_0.cons().forward(
        obj_type=A_l1_ty, name="A_L2L1_0", dims_to_stream=[(m, k)], placement=Tile(0, 2)
    )
    A_l2l1_1 = A_l3l2_1.cons().forward(
        obj_type=A_l1_ty, name="A_L2L1_1", dims_to_stream=[(m, k)], placement=Tile(0, 3)
    )
    A_l2l1_2 = A_l3l2_2.cons().forward(
        obj_type=A_l1_ty, name="A_L2L1_2", dims_to_stream=[(m, k)], placement=Tile(0, 4)
    )
    A_l2l1_3 = A_l3l2_3.cons().forward(
        obj_type=A_l1_ty, name="A_L2L1_3", dims_to_stream=[(m, k)], placement=Tile(0, 5)
    )

    def core_put_only_A(of_in_a, matmul_scalar_cascade_put_only):
        elem_in_a = of_in_a.acquire(1)
        matmul_scalar_cascade_put_only(elem_in_a, None)
        of_in_a.release(1)

    worker_A_0 = Worker(core_put_only_A,[A_l2l1_0.cons(),matmul_scalar_cascade_put_only],placement=Tile(0, 2))
    worker_A_1 = Worker(core_put_only_A, [A_l2l1_1.cons(), matmul_scalar_cascade_put_only], placement=Tile(0, 3))
    worker_A_2 = Worker(core_put_only_A, [A_l2l1_2.cons(), matmul_scalar_cascade_put_only], placement=Tile(0, 4))
    worker_A_3 = Worker(core_put_only_A, [A_l2l1_3.cons(), matmul_scalar_cascade_put_only], placement=Tile(0, 5))
    B_l3l2_0 = ObjectFifo(B_l2_ty, name="B_L3L2_0")
    B_l3l2_1 = ObjectFifo(B_l2_ty, name="B_L3L2_1")
    B_l3l2_2 = ObjectFifo(B_l2_ty, name="B_L3L2_2")
    B_l3l2_3 = ObjectFifo(B_l2_ty, name="B_L3L2_3")

    B_l2l1_0 = B_l3l2_0.cons().forward(
        obj_type=B_l1_ty, name="B_L2L1_0", dims_to_stream=[(k, n)], placement=Tile(0, 5)
    )
    B_l2l1_1 = B_l3l2_1.cons().forward(
        obj_type=B_l1_ty, name="B_L2L1_1", dims_to_stream=[(k, n)], placement=Tile(1, 5)
    )
    B_l2l1_2 = B_l3l2_2.cons().forward(
        obj_type=B_l1_ty, name="B_L2L1_2", dims_to_stream=[(k, n)], placement=Tile(2, 5)
    )
    B_l2l1_3 = B_l3l2_3.cons().forward(
        obj_type=B_l1_ty, name="B_L2L1_3", dims_to_stream=[(k, n)], placement=Tile(3, 5)
    )

    def core_put_only_B(of_in_b, matmul_scalar_cascade_put_only):
        elem_in_b = of_in_b.acquire(1)
        matmul_scalar_cascade_put_only(None, elem_in_b)
        of_in_b.release(1)

    # worker_B_0 = Worker(core_put_only_B, [B_l2l1_0.cons(), matmul_scalar_cascade_put_only], placement=Tile(0, 5))
    # worker_B_1 = Worker(core_put_only_B, [B_l2l1_1.cons(), matmul_scalar_cascade_put_only], placement=Tile(1, 5))
    # worker_B_2 = Worker(core_put_only_B, [B_l2l1_2.cons(), matmul_scalar_cascade_put_only], placement=Tile(2, 5))
    # worker_B_3 = Worker(core_put_only_B, [B_l2l1_3.cons(), matmul_scalar_cascade_put_only], placement=Tile(3, 5))

    C_l2l3_0 = ObjectFifo(C_l2_ty, name="C_L2L3_0")
    C_l2l3_1 = ObjectFifo(C_l2_ty, name="C_L2L3_1")
    C_l2l3_2 = ObjectFifo(C_l2_ty, name="C_L2L3_2")
    C_l2l3_3 = ObjectFifo(C_l2_ty, name="C_L2L3_3")

    C_l1l2_0 = C_l2l3_0.cons().forward(
        obj_type=C_l1_ty, name="C_L1L2_0", placement=Tile(0, 2)
    )
    C_l1l2_1 = C_l2l3_1.cons().forward(
        obj_type=C_l1_ty, name="C_L1L2_1", placement=Tile(1, 2)
    )
    C_l1l2_2 = C_l2l3_2.cons().forward(
        obj_type=C_l1_ty, name="C_L1L2_2", placement=Tile(2, 2)
    )
    C_l1l2_3 = C_l2l3_3.cons().forward(
        obj_type=C_l1_ty, name="C_L1L2_3", placement=Tile(3, 2)
    )

    def core_get_only_C(of_out_c, matmul_scalar_cascade_get_only):
        elem_out = of_out_c.acquire(1)
        matmul_scalar_cascade_get_only(None, None, elem_out)
        of_out_c.release(1)

    worker_C_0 = Worker(core_get_only_C, [C_l1l2_0.cons(), matmul_scalar_cascade_get_only], placement=Tile(0, 2))
    worker_C_1 = Worker(core_get_only_C, [C_l1l2_1.cons(), matmul_scalar_cascade_get_only], placement=Tile(1, 2))
    worker_C_2 = Worker(core_get_only_C, [C_l1l2_2.cons(), matmul_scalar_cascade_get_only], placement=Tile(2, 2))
    worker_C_3 = Worker(core_get_only_C, [C_l1l2_3.cons(), matmul_scalar_cascade_get_only], placement=Tile(3, 2))

    rt = Runtime()
    with rt.sequence(data_ty, data_ty, data_ty) as (A, B, C):
        rt.start(worker_A_0, worker_A_1, worker_A_2, worker_A_3)
        #rt.start(worker_B_0, worker_B_1, worker_B_2, worker_B_3)
        rt.start(worker_C_0, worker_C_1, worker_C_2, worker_C_3)
        rt.fill()
    p = Program(iron.get_current_device(), rt).resolve_program(SequentialPlacer())
    print(p)
    return p

def main():
    input1 = iron.arange(60, dtype=np.int32, device="npu")
    input2 = iron.arange(60, dtype=np.int32, device="npu")
    output = iron.zeros_like(input1)
    transform_with_internal_func_from_file(input1, input2, output)
    print(output)

if __name__ == "__main__":
    main()