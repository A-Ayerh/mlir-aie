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
from aie.iron.dataflow.objectfifo import ObjectFifoLink, ObjectFifoHandle, object_fifo_link
from aie.iron.device import NPU1Col1, NPU1Col2, NPU1, NPU2, Tile
from aie.iron.jit import compile_external_kernel


@iron.jit(is_placed=False)
def exercise_5a(input0, output, l3_l2, l3_l1, l2_l3, l2_l1, l1_l2, l1_l3):
    data_size = input0.numel()
    element_type = input0.dtype

    Cols = 4

    CTRows = 4
    MTRows = 1
    ITRows = 1
    
    n_aie_rows = 4
    n_aie_cols = 4
    n_aie_cores = n_aie_rows * n_aie_cols

    fifo_depth = 2

    #n_tiles_per_core = (M // m) * (N // n) // n_aie_cores

    # When using more AIE columns than n_aie_rows (4) (applicable to NPU2),
    # restrict the number of shim/mem tiles to n_aie_rows,
    # since we have only n_aie_rows row tiles for matrix A
    if n_aie_cols > n_aie_rows:
        n_shim_mem_A = n_aie_rows
    # When using n_aie_rows (4) or less AIE columns (both NPU and NPU2),
    # the number of shim/mem tiles are equal to n_aie_cols.
    # We use the distribute pattern in object FIFO (see linking for A below),
    # since we have n_aie_rows (4) row tiles for matrix A
    else:
        n_shim_mem_A = n_aie_cols

    # Integer division when n_aie_cols < 4, otherwise set to 1
    n_A_tiles_per_shim = n_aie_rows // n_aie_cols if n_aie_cols < 4 else 1

    data_ty = np.ndarray[(data_size,), np.dtype[element_type]]
    A_ty = np.ndarray[(M * K,), np.dtype[dtype_in]]
    B_ty = np.ndarray[(K * N,), np.dtype[dtype_in]]
    C_ty = np.ndarray[(M * N,), np.dtype[dtype_out]]
    A_l2_ty = np.ndarray[(m * k * n_A_tiles_per_shim,), np.dtype[dtype_in]]
    B_l2_ty = np.ndarray[(k * n,), np.dtype[dtype_in]]
    C_l2_ty = np.ndarray[(m * n * n_aie_rows,), np.dtype[dtype_out]]
    A_l1_ty = np.ndarray[(m, k), np.dtype[dtype_in]]
    B_l1_ty = np.ndarray[(k, n), np.dtype[dtype_in]]
    C_l1_ty = np.ndarray[(m, n), np.dtype[dtype_out]]

    
    #TODO:generate object fifos to handle amount of data:
    # Tile declarations as tile[row][col]
    tiles = [[(col, row) for col in range(0, n_aie_cols)] for row in range(0, 6)]
    core_tiles = tiles[2:]

    # AIE-array data movement with object fifos
    A_l3l2_fifos = [None] * n_shim_mem_A
    A_l2l1_fifos = [None] * n_aie_rows

    B_l3l2_fifos = [None] * n_aie_cols
    B_l2l1_fifos = [None] * n_aie_cols

    C_l1l2_fifos = [[None] * n_aie_cols for _ in range(n_aie_rows)]
    C_l2l3_fifos = [None] * n_aie_cols

    # Input A
    for i in range(n_shim_mem_A):
        A_l3l2_fifos[i] = ObjectFifo(A_l2_ty, name=f"A_L3L2_{i}", depth=fifo_depth)
        # If n_shim_mem_A == n_rows, n_A_tiles_per_shim is 1 and
        # this simply links a_l3l2_fifos[i] to a_l2l1_fifos[i] directly,
        # If n_shim_mem_A < n_rows, each column receives multiple rows of
        # tiles; distribute it along rows of AIE cores.
        start_row = i * n_A_tiles_per_shim
        stop_row = start_row + n_A_tiles_per_shim
        of_offsets = [m * k * j for j in range(stop_row - start_row)]
        dims_to_stream = [
            [
                (m // r, r * k),
                (k // s, s),
                (r, k),
                (s, 1),
            ]
        ] * (stop_row - start_row)
        a_tmp_fifos = (
            A_l3l2_fifos[i]
            .cons()
            .split(
                of_offsets,
                obj_types=[A_l1_ty] * (stop_row - start_row),
                names=[f"A_L2L1_{row}" for row in range(start_row, stop_row)],
                dims_to_stream=dims_to_stream,
                placement=Tile(
                    2 * i if n_aie_cols == 8 else i, 1
                ),  # alternate columns in full 4x8 NPU2 case
            )
        )

        for j in range(stop_row - start_row):
            A_l2l1_fifos[j + start_row] = a_tmp_fifos[j]

    # Input B
    for col in range(n_aie_cols):
        B_l3l2_fifos[col] = ObjectFifo(B_l2_ty, name=f"B_L3L2_{col}", depth=fifo_depth)
        if b_col_maj:
            dims_to_stream = [(n // t, t * k), (k // s, s), (t, k), (s, 1)]
        else:
            dims_to_stream = [(k // s, s * n), (n // t, t), (s, n), (t, 1)]
        B_l2l1_fifos[col] = (
            B_l3l2_fifos[col]
            .cons()
            .forward(
                obj_type=B_l1_ty,
                name=f"B_L2L1_{col}",
                dims_to_stream=dims_to_stream,
                placement=Tile(col, 1),
            )
        )

        # Output C
        C_l2l3_fifos[col] = ObjectFifo(
            C_l2_ty,
            name=f"C_L2L3_{col}",
            depth=fifo_depth,
            dims_to_stream=[(m // r, r * n), (r, t), (n // t, r * t), (t, 1)],
        )
        of_offsets = [m * n * i for i in range(n_aie_rows)]

        # join along one column
        c_tmp_fifos = (
            C_l2l3_fifos[col]
            .prod()
            .join(
                of_offsets,
                obj_types=[C_l1_ty] * n_aie_rows,
                names=[f"C_L1L2_{col}_{row}" for row in range(n_aie_rows)],
                depths=[fifo_depth] * n_aie_rows,
                placement=Tile(col, 1),
            )
        )
        for j in range(n_aie_rows):
            C_l1l2_fifos[j][col] = c_tmp_fifos[j]
    

    
     # Dataflow with ObjectFifos
    # of_in = ObjectFifo(data_ty, name="in")
    # of_in_mem = of_in.cons().forward(name="in_mem")

    of_out_mem = ObjectFifo(data_ty, name="out")
    ObjectFifoLink()
    object_fifo_link()
    ObjectFifoHandle()
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
        if(l3_l1):
            for i in range(numCTFifos):
                rt.fill(l3_l1_fifos[i].prod(), a_in)
        if(l3_l2):
            for i in range(numMemFifos):
                rt.fill(l3_l2_temp[i].prod(), a_in)
        if(l1_l3):
            for i in range(numCTFifos):
                rt.drain(l1_l3_fifos[i].cons(), c_out, wait=True)
        if(l2_l3):
            for i in range(numMemFifos):
                rt.drain(l2_l3_temp[i].cons(), c_out, wait=True)
            
            

    # Create the program from the device type and runtime
    my_program = Program(iron.get_current_device(), rt)

    # Place components (assign them resources on the device) and generate an MLIR module
    
    my_program=  my_program.resolve_program(SequentialPlacer())
    #print(my_program)

    return my_program


def main():
    # Define tensor shapes and data types
    data_size = (2**12)-64 # size max for compute tiles streaming in/out int32 and using ping pong buffer for in/out
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
