#!/usr/bin/env python3
import os
import aie.iron as iron
from aie.iron import ExternalFunction, jit
from aie.iron import ObjectFifo, Worker, Runtime, Program
from aie.iron.placers import SequentialPlacer
from aie.iron.device import Tile
import numpy as np
from ml_dtypes import bfloat16
from aie.helpers.taplib import TensorAccessPattern

@iron.jit(is_placed=False)
def generated_design(inputA, inputB, outputD):

    element_type = bfloat16
    data_size = inputA.numel() if inputA else 1024
    num_mem_nodes = 4
    col_data_size = data_size // num_mem_nodes  # 256
    chunk_size = 128  # Reduced to minimize BD usage

    max_chunk_size = 256
    data_ty = np.ndarray[(data_size,), np.dtype[element_type]]
    chunk_ty = np.ndarray[(chunk_size,), np.dtype[element_type]]
    col_ty = np.ndarray[(col_data_size,), np.dtype[element_type]]
    
    # Input/output specific types
    data_a_ty = np.ndarray[(data_size,), np.dtype[element_type]]
    data_b_ty = np.ndarray[(data_size,), np.dtype[element_type]]
    data_d_ty = np.ndarray[(data_size,), np.dtype[element_type]]
    
    # Define tiles for compute and shim nodes
    tile_0_0 = Tile(0, 0)
    tile_1_0 = Tile(1, 0)
    tile_2_0 = Tile(2, 0)
    tile_3_0 = Tile(3, 0)
    tile_0_1 = Tile(0, 1)
    tile_0_4 = Tile(0, 4)
    tile_0_2 = Tile(0, 2)
    tile_0_5 = Tile(0, 5)
    tile_0_3 = Tile(0, 3)
    tile_1_1 = Tile(1, 1)
    tile_1_4 = Tile(1, 4)
    tile_1_2 = Tile(1, 2)
    tile_1_5 = Tile(1, 5)
    tile_1_3 = Tile(1, 3)
    tile_2_1 = Tile(2, 1)
    tile_2_4 = Tile(2, 4)
    tile_2_2 = Tile(2, 2)
    tile_2_5 = Tile(2, 5)
    tile_2_3 = Tile(2, 3)
    tile_3_1 = Tile(3, 1)
    tile_3_4 = Tile(3, 4)
    tile_3_2 = Tile(3, 2)
    tile_3_5 = Tile(3, 5)
    tile_3_3 = Tile(3, 3)

    # Define base object FIFOs for shim <-> memory connections (use col_ty, not full data_ty)
    of_from_shim_col0_to_mem_col0_0 = ObjectFifo(col_ty, depth=2, name='SHIM_L3_L2_A1A2_col0')  # Reduced depth
    of_from_shim_col0_to_mem_col0_1 = ObjectFifo(col_ty, depth=2, name='SHIM_L3_L2_B1B2_col0')
    of_from_shim_col1_to_mem_col1_0 = ObjectFifo(col_ty, depth=2, name='SHIM_L3_L2_A3A4_col1')
    of_from_shim_col1_to_mem_col1_1 = ObjectFifo(col_ty, depth=2, name='SHIM_L3_L2_B3B4_col1')
    of_from_shim_col2_to_mem_col2_0 = ObjectFifo(col_ty, depth=2, name='SHIM_L3_L2_A5A6_col2')
    of_from_shim_col2_to_mem_col2_1 = ObjectFifo(col_ty, depth=2, name='SHIM_L3_L2_B5B6_col2')
    of_from_shim_col3_to_mem_col3_0 = ObjectFifo(col_ty, depth=2, name='SHIM_L3_L2_A7A8_col3')
    of_from_shim_col3_to_mem_col3_1 = ObjectFifo(col_ty, depth=2, name='SHIM_L3_L2_B7B8_col3')
    of_from_mem_col0_to_shim_col0_0 = ObjectFifo(col_ty, depth=2, name='SHIM_L2_L3_D1D2_col0')
    of_from_mem_col1_to_shim_col1_0 = ObjectFifo(col_ty, depth=2, name='SHIM_L2_L3_D3D4_col1')
    of_from_mem_col2_to_shim_col2_0 = ObjectFifo(col_ty, depth=2, name='SHIM_L2_L3_D5D6_col2')
    of_from_mem_col3_to_shim_col3_0 = ObjectFifo(col_ty, depth=2, name='SHIM_L2_L3_D7D8_col3')

    # Split/Join operations on memory tiles - proper offsets for 128 chunks
    split_mem_col0_A = of_from_shim_col0_to_mem_col0_0.cons().split(offsets=[0, 128], obj_types=[chunk_ty] * 2, depths=[2] * 2, names=[f'split_mem_col0_A_{i}' for i in range(2)], placement=tile_0_1)
    split_mem_col0_B = of_from_shim_col0_to_mem_col0_1.cons().split(offsets=[0, 128], obj_types=[chunk_ty] * 2, depths=[2] * 2, names=[f'split_mem_col0_B_{i}' for i in range(2)], placement=tile_0_1)
    join_mem_col0_D = of_from_mem_col0_to_shim_col0_0.prod().join(offsets=[0, 128], obj_types=[chunk_ty] * 2, depths=[2] * 2, names=[f'join_mem_col0_D_{i}' for i in range(2)], placement=tile_0_1)
    split_mem_col1_A = of_from_shim_col1_to_mem_col1_0.cons().split(offsets=[0, 128], obj_types=[chunk_ty] * 2, depths=[2] * 2, names=[f'split_mem_col1_A_{i}' for i in range(2)], placement=tile_1_1)
    split_mem_col1_B = of_from_shim_col1_to_mem_col1_1.cons().split(offsets=[0, 128], obj_types=[chunk_ty] * 2, depths=[2] * 2, names=[f'split_mem_col1_B_{i}' for i in range(2)], placement=tile_1_1)
    join_mem_col1_D = of_from_mem_col1_to_shim_col1_0.prod().join(offsets=[0, 128], obj_types=[chunk_ty] * 2, depths=[2] * 2, names=[f'join_mem_col1_D_{i}' for i in range(2)], placement=tile_1_1)
    split_mem_col2_A = of_from_shim_col2_to_mem_col2_0.cons().split(offsets=[0, 128], obj_types=[chunk_ty] * 2, depths=[2] * 2, names=[f'split_mem_col2_A_{i}' for i in range(2)], placement=tile_2_1)
    split_mem_col2_B = of_from_shim_col2_to_mem_col2_1.cons().split(offsets=[0, 128], obj_types=[chunk_ty] * 2, depths=[2] * 2, names=[f'split_mem_col2_B_{i}' for i in range(2)], placement=tile_2_1)
    join_mem_col2_D = of_from_mem_col2_to_shim_col2_0.prod().join(offsets=[0, 128], obj_types=[chunk_ty] * 2, depths=[2] * 2, names=[f'join_mem_col2_D_{i}' for i in range(2)], placement=tile_2_1)
    split_mem_col3_A = of_from_shim_col3_to_mem_col3_0.cons().split(offsets=[0, 128], obj_types=[chunk_ty] * 2, depths=[2] * 2, names=[f'split_mem_col3_A_{i}' for i in range(2)], placement=tile_3_1)
    split_mem_col3_B = of_from_shim_col3_to_mem_col3_1.cons().split(offsets=[0, 128], obj_types=[chunk_ty] * 2, depths=[2] * 2, names=[f'split_mem_col3_B_{i}' for i in range(2)], placement=tile_3_1)
    join_mem_col3_D = of_from_mem_col3_to_shim_col3_0.prod().join(offsets=[0, 128], obj_types=[chunk_ty] * 2, depths=[2] * 2, names=[f'join_mem_col3_D_{i}' for i in range(2)], placement=tile_3_1)

    # Define external C/C++ kernel functions
    external_eltwiseaddbf16vector = ExternalFunction(name="eltwise_add_bf16_vector", source_file="./add.cc", arg_types=[chunk_ty, chunk_ty, chunk_ty], include_dirs=['./'])
    external_bf16relu = ExternalFunction(name="bf16_relu", source_file="./relu.cc", arg_types=[chunk_ty, chunk_ty], include_dirs=['./'])

    # Define core functions - single execution (no infinite loops)
    def core_fn_A1B1worker(of_in1, of_in2, of_out1, external_eltwiseaddbf16vector):
        elem_in1 = of_in1.acquire(1)
        if elem_in1 == 0: return
        elem_in2 = of_in2.acquire(1)
        if elem_in2 == 0: return
        elem_out1 = of_out1.acquire(1)
        if elem_out1 == 0: return
        external_eltwiseaddbf16vector(elem_in1, elem_in2, elem_out1)
        of_in1.release(1)
        of_in2.release(1)
        of_out1.release(1)

    def core_fn_C1worker(of_in1, of_out1, external_bf16relu):
        elem_in1 = of_in1.acquire(1)
        if elem_in1 == 0: return
        elem_out1 = of_out1.acquire(1)
        if elem_out1 == 0: return
        external_bf16relu(elem_in1, elem_out1)
        of_in1.release(1)
        of_out1.release(1)

    def core_fn_A2B2worker(of_in1, of_in2, of_out1, external_eltwiseaddbf16vector):
        elem_in1 = of_in1.acquire(1)
        if elem_in1 == 0: return
        elem_in2 = of_in2.acquire(1)
        if elem_in2 == 0: return
        elem_out1 = of_out1.acquire(1)
        if elem_out1 == 0: return
        external_eltwiseaddbf16vector(elem_in1, elem_in2, elem_out1)
        of_in1.release(1)
        of_in2.release(1)
        of_out1.release(1)

    def core_fn_C2worker(of_in1, of_out1, external_bf16relu):
        elem_in1 = of_in1.acquire(1)
        if elem_in1 == 0: return
        elem_out1 = of_out1.acquire(1)
        if elem_out1 == 0: return
        external_bf16relu(elem_in1, elem_out1)
        of_in1.release(1)
        of_out1.release(1)

    def core_fn_A3B3worker(of_in1, of_in2, of_out1, external_eltwiseaddbf16vector):
        elem_in1 = of_in1.acquire(1); 
        if elem_in1 == 0: return
        elem_in2 = of_in2.acquire(1); 
        if elem_in2 == 0: return
        elem_out1 = of_out1.acquire(1); 
        if elem_out1 == 0: return
        external_eltwiseaddbf16vector(elem_in1, elem_in2, elem_out1)
        of_in1.release(1); of_in2.release(1); of_out1.release(1)

    def core_fn_C3worker(of_in1, of_out1, external_bf16relu):
        elem_in1 = of_in1.acquire(1); 
        if elem_in1 == 0: return
        elem_out1 = of_out1.acquire(1); 
        if elem_out1 == 0: return
        external_bf16relu(elem_in1, elem_out1)
        of_in1.release(1); of_out1.release(1)

    def core_fn_A4B4worker(of_in1, of_in2, of_out1, external_eltwiseaddbf16vector):
        elem_in1 = of_in1.acquire(1); 
        if elem_in1 == 0: return
        elem_in2 = of_in2.acquire(1); 
        if elem_in2 == 0: return
        elem_out1 = of_out1.acquire(1); 
        if elem_out1 == 0: return
        external_eltwiseaddbf16vector(elem_in1, elem_in2, elem_out1)
        of_in1.release(1); of_in2.release(1); of_out1.release(1)

    def core_fn_C4worker(of_in1, of_out1, external_bf16relu):
        elem_in1 = of_in1.acquire(1); 
        if elem_in1 == 0: return
        elem_out1 = of_out1.acquire(1); 
        if elem_out1 == 0: return
        external_bf16relu(elem_in1, elem_out1)
        of_in1.release(1); of_out1.release(1)

    def core_fn_A5B5worker(of_in1, of_in2, of_out1, external_eltwiseaddbf16vector):
        elem_in1 = of_in1.acquire(1); 
        if elem_in1 == 0: return
        elem_in2 = of_in2.acquire(1); 
        if elem_in2 == 0: return
        elem_out1 = of_out1.acquire(1); 
        if elem_out1 == 0: return
        external_eltwiseaddbf16vector(elem_in1, elem_in2, elem_out1)
        of_in1.release(1); of_in2.release(1); of_out1.release(1)

    def core_fn_C5worker(of_in1, of_out1, external_bf16relu):
        elem_in1 = of_in1.acquire(1); 
        if elem_in1 == 0: return
        elem_out1 = of_out1.acquire(1); 
        if elem_out1 == 0: return
        external_bf16relu(elem_in1, elem_out1)
        of_in1.release(1); of_out1.release(1)

    def core_fn_A6B6worker(of_in1, of_in2, of_out1, external_eltwiseaddbf16vector):
        elem_in1 = of_in1.acquire(1); 
        if elem_in1 == 0: return
        elem_in2 = of_in2.acquire(1); 
        if elem_in2 == 0: return
        elem_out1 = of_out1.acquire(1); 
        if elem_out1 == 0: return
        external_eltwiseaddbf16vector(elem_in1, elem_in2, elem_out1)
        of_in1.release(1); of_in2.release(1); of_out1.release(1)

    def core_fn_C6worker(of_in1, of_out1, external_bf16relu):
        elem_in1 = of_in1.acquire(1); 
        if elem_in1 == 0: return
        elem_out1 = of_out1.acquire(1); 
        if elem_out1 == 0: return
        external_bf16relu(elem_in1, elem_out1)
        of_in1.release(1); of_out1.release(1)

    def core_fn_A7B7worker(of_in1, of_in2, of_out1, external_eltwiseaddbf16vector):
        elem_in1 = of_in1.acquire(1); 
        if elem_in1 == 0: return
        elem_in2 = of_in2.acquire(1); 
        if elem_in2 == 0: return
        elem_out1 = of_out1.acquire(1); 
        if elem_out1 == 0: return
        external_eltwiseaddbf16vector(elem_in1, elem_in2, elem_out1)
        of_in1.release(1); of_in2.release(1); of_out1.release(1)

    def core_fn_C7worker(of_in1, of_out1, external_bf16relu):
        elem_in1 = of_in1.acquire(1); 
        if elem_in1 == 0: return
        elem_out1 = of_out1.acquire(1); 
        if elem_out1 == 0: return
        external_bf16relu(elem_in1, elem_out1)
        of_in1.release(1); of_out1.release(1)

    def core_fn_A8B8worker(of_in1, of_in2, of_out1, external_eltwiseaddbf16vector):
        elem_in1 = of_in1.acquire(1); 
        if elem_in1 == 0: return
        elem_in2 = of_in2.acquire(1); 
        if elem_in2 == 0: return
        elem_out1 = of_out1.acquire(1); 
        if elem_out1 == 0: return
        external_eltwiseaddbf16vector(elem_in1, elem_in2, elem_out1)
        of_in1.release(1); of_in2.release(1); of_out1.release(1)

    def core_fn_C8worker(of_in1, of_out1, external_bf16relu):
        elem_in1 = of_in1.acquire(1); 
        if elem_in1 == 0: return
        elem_out1 = of_out1.acquire(1); 
        if elem_out1 == 0: return
        external_bf16relu(elem_in1, elem_out1)
        of_in1.release(1); of_out1.release(1)

    # KEY FIX: Create intermediate FIFOs for add->relu communication
    of_add_to_relu_col0_0 = ObjectFifo(chunk_ty, depth=2, name='add_relu_col0_0')
    of_add_to_relu_col0_1 = ObjectFifo(chunk_ty, depth=2, name='add_relu_col0_1')
    of_add_to_relu_col1_0 = ObjectFifo(chunk_ty, depth=2, name='add_relu_col1_0')
    of_add_to_relu_col1_1 = ObjectFifo(chunk_ty, depth=2, name='add_relu_col1_1')
    of_add_to_relu_col2_0 = ObjectFifo(chunk_ty, depth=2, name='add_relu_col2_0')
    of_add_to_relu_col2_1 = ObjectFifo(chunk_ty, depth=2, name='add_relu_col2_1')
    of_add_to_relu_col3_0 = ObjectFifo(chunk_ty, depth=2, name='add_relu_col3_0')
    of_add_to_relu_col3_1 = ObjectFifo(chunk_ty, depth=2, name='add_relu_col3_1')

    # Define workers - PROPERLY CONNECTED
    worker_A1B1worker = Worker(core_fn_A1B1worker, [split_mem_col0_A[0].cons(), split_mem_col0_B[0].cons(), of_add_to_relu_col0_0.prod(), external_eltwiseaddbf16vector], placement=tile_0_4, while_true=True, stack_size=1024, allocation_scheme='heap')
    worker_C1worker = Worker(core_fn_C1worker, [of_add_to_relu_col0_0.cons(), join_mem_col0_D[0].prod(), external_bf16relu], placement=tile_0_2, while_true=True, stack_size=1024, allocation_scheme='heap')
    
    worker_A2B2worker = Worker(core_fn_A2B2worker, [split_mem_col0_A[1].cons(), split_mem_col0_B[1].cons(), of_add_to_relu_col0_1.prod(), external_eltwiseaddbf16vector], placement=tile_0_5, while_true=True, stack_size=1024, allocation_scheme='heap')
    worker_C2worker = Worker(core_fn_C2worker, [of_add_to_relu_col0_1.cons(), join_mem_col0_D[1].prod(), external_bf16relu], placement=tile_0_3, while_true=True, stack_size=1024, allocation_scheme='heap')
    
    worker_A3B3worker = Worker(core_fn_A3B3worker, [split_mem_col1_A[0].cons(), split_mem_col1_B[0].cons(), of_add_to_relu_col1_0.prod(), external_eltwiseaddbf16vector], placement=tile_1_4, while_true=True, stack_size=1024, allocation_scheme='heap')
    worker_C3worker = Worker(core_fn_C3worker, [of_add_to_relu_col1_0.cons(), join_mem_col1_D[0].prod(), external_bf16relu], placement=tile_1_2, while_true=True, stack_size=1024, allocation_scheme='heap')
    
    worker_A4B4worker = Worker(core_fn_A4B4worker, [split_mem_col1_A[1].cons(), split_mem_col1_B[1].cons(), of_add_to_relu_col1_1.prod(), external_eltwiseaddbf16vector], placement=tile_1_5, while_true=True, stack_size=1024, allocation_scheme='heap')
    worker_C4worker = Worker(core_fn_C4worker, [of_add_to_relu_col1_1.cons(), join_mem_col1_D[1].prod(), external_bf16relu], placement=tile_1_3, while_true=True, stack_size=1024, allocation_scheme='heap')
    
    worker_A5B5worker = Worker(core_fn_A5B5worker, [split_mem_col2_A[0].cons(), split_mem_col2_B[0].cons(), of_add_to_relu_col2_0.prod(), external_eltwiseaddbf16vector], placement=tile_2_4, while_true=True, stack_size=1024, allocation_scheme='heap')
    worker_C5worker = Worker(core_fn_C5worker, [of_add_to_relu_col2_0.cons(), join_mem_col2_D[0].prod(), external_bf16relu], placement=tile_2_2, while_true=True, stack_size=1024, allocation_scheme='heap')
    
    worker_A6B6worker = Worker(core_fn_A6B6worker, [split_mem_col2_A[1].cons(), split_mem_col2_B[1].cons(), of_add_to_relu_col2_1.prod(), external_eltwiseaddbf16vector], placement=tile_2_5, while_true=True, stack_size=1024, allocation_scheme='heap')
    worker_C6worker = Worker(core_fn_C6worker, [of_add_to_relu_col2_1.cons(), join_mem_col2_D[1].prod(), external_bf16relu], placement=tile_2_3, while_true=True, stack_size=1024, allocation_scheme='heap')
    
    worker_A7B7worker = Worker(core_fn_A7B7worker, [split_mem_col3_A[0].cons(), split_mem_col3_B[0].cons(), of_add_to_relu_col3_0.prod(), external_eltwiseaddbf16vector], placement=tile_3_4, while_true=True, stack_size=1024, allocation_scheme='heap')
    worker_C7worker = Worker(core_fn_C7worker, [of_add_to_relu_col3_0.cons(), join_mem_col3_D[0].prod(), external_bf16relu], placement=tile_3_2, while_true=True, stack_size=1024, allocation_scheme='heap')
    
    worker_A8B8worker = Worker(core_fn_A8B8worker, [split_mem_col3_A[1].cons(), split_mem_col3_B[1].cons(), of_add_to_relu_col3_1.prod(), external_eltwiseaddbf16vector], placement=tile_3_5, while_true=True, stack_size=1024, allocation_scheme='heap')
    worker_C8worker = Worker(core_fn_C8worker, [of_add_to_relu_col3_1.cons(), join_mem_col3_D[1].prod(), external_bf16relu], placement=tile_3_3, while_true=True, stack_size=1024, allocation_scheme='heap')

    # Runtime sequence with proper data sizes
    rt = Runtime()
    with rt.sequence(data_a_ty, data_b_ty, data_d_ty) as (A,B,D):
        workers = [worker_A1B1worker, worker_C1worker, worker_A2B2worker, worker_C2worker,
                  worker_A3B3worker, worker_C3worker, worker_A4B4worker, worker_C4worker,
                  worker_A5B5worker, worker_C5worker, worker_A6B6worker, worker_C6worker,
                  worker_A7B7worker, worker_C7worker, worker_A8B8worker, worker_C8worker]
        rt.start(*workers)
       
        # Fill column data (256 elements per column)
        rt.fill(of_from_shim_col0_to_mem_col0_0.prod(), A, tap=TensorAccessPattern(tensor_dims=[1024,], offset=0, sizes=[256,], strides=[1,]))
        rt.fill(of_from_shim_col1_to_mem_col1_0.prod(), A, tap=TensorAccessPattern(tensor_dims=[1024,], offset=256, sizes=[256,], strides=[1,]))
        rt.fill(of_from_shim_col2_to_mem_col2_0.prod(), A, tap=TensorAccessPattern(tensor_dims=[1024,], offset=512, sizes=[256,], strides=[1,]))
        rt.fill(of_from_shim_col3_to_mem_col3_0.prod(), A, tap=TensorAccessPattern(tensor_dims=[1024,], offset=768, sizes=[256,], strides=[1,]))
       
        rt.fill(of_from_shim_col0_to_mem_col0_1.prod(), B, tap=TensorAccessPattern(tensor_dims=[1024,], offset=0, sizes=[256,], strides=[1,]))
        rt.fill(of_from_shim_col1_to_mem_col1_1.prod(), B, tap=TensorAccessPattern(tensor_dims=[1024,], offset=256, sizes=[256,], strides=[1,]))
        rt.fill(of_from_shim_col2_to_mem_col2_1.prod(), B, tap=TensorAccessPattern(tensor_dims=[1024,], offset=512, sizes=[256,], strides=[1,]))
        rt.fill(of_from_shim_col3_to_mem_col3_1.prod(), B, tap=TensorAccessPattern(tensor_dims=[1024,], offset=768, sizes=[256,], strides=[1,]))
       
        # Drain with proper offsets and wait for completion
        rt.drain(of_from_mem_col0_to_shim_col0_0.cons(), D, wait=True, tap=TensorAccessPattern(tensor_dims=[1024,], offset=0, sizes=[256,], strides=[1,]))
        rt.drain(of_from_mem_col1_to_shim_col1_0.cons(), D, wait=True, tap=TensorAccessPattern(tensor_dims=[1024,], offset=256, sizes=[256,], strides=[1,]))
        rt.drain(of_from_mem_col2_to_shim_col2_0.cons(), D, wait=True, tap=TensorAccessPattern(tensor_dims=[1024,], offset=512, sizes=[256,], strides=[1,]))
        rt.drain(of_from_mem_col3_to_shim_col3_0.cons(), D, wait=True, tap=TensorAccessPattern(tensor_dims=[1024,], offset=768, sizes=[256,], strides=[1,]))
       
    my_program = Program(iron.get_current_device(), rt)

    return my_program.resolve_program(SequentialPlacer())

def main():
    datatype = bfloat16
    data_size = 1024
    inputA = iron.rand(data_size, dtype=datatype, device="npu")
    inputB = iron.rand(data_size, dtype=datatype, device="npu")
    outputD = iron.zeros(data_size, dtype=datatype, device="npu")
    program = generated_design(inputA, inputB, outputD)
    program()
    print(iron.to_numpy(outputD))

if __name__ == "__main__":
    main()
