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
    data_size = inputA.numel()
    data_ty = np.ndarray[(data_size,), np.dtype[element_type]]
    data_a_ty = np.ndarray[(data_size,), np.dtype[element_type]]
    data_b_ty = np.ndarray[(data_size,), np.dtype[element_type]]
    data_d_ty = np.ndarray[(data_size,), np.dtype[element_type]]
    # Define tiles for compute and shim nodes
    tile_0_0 = Tile(0, 0)
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

    # Define object FIFOs for data streaming between tiles
    of_from_shim_to_mem_col1_0 = ObjectFifo(data_a_ty, depth=2, name='SHIM_L3_L2_A1A2_col1')
    of_from_shim_to_mem_col1_1 = ObjectFifo(data_b_ty, depth=2, name='SHIM_L3_L2_B1B2_col1')
    of_from_shim_to_mem_col2_0 = ObjectFifo(data_a_ty, depth=2, name='SHIM_L3_L2_A3A4_col2')
    of_from_shim_to_mem_col2_1 = ObjectFifo(data_b_ty, depth=2, name='SHIM_L3_L2_B3B4_col2')
    of_from_shim_to_mem_col3_0 = ObjectFifo(data_a_ty, depth=2, name='SHIM_L3_L2_A5A6_col3')
    of_from_shim_to_mem_col3_1 = ObjectFifo(data_b_ty, depth=2, name='SHIM_L3_L2_B5B6_col3')
    of_from_shim_to_mem_col4_0 = ObjectFifo(data_a_ty, depth=2, name='SHIM_L3_L2_A7A8_col4')
    of_from_shim_to_mem_col4_1 = ObjectFifo(data_b_ty, depth=2, name='SHIM_L3_L2_B7B8_col4')
    of_from_mem_col1_to_shim_0 = ObjectFifo(data_d_ty, depth=2, name='SHIM_L2_L3_D1D2_col1')
    of_from_A1_B1_worker_to_C1_worker_0 = ObjectFifo(data_ty, tile_0_4, tile_0_2, depth=2, name='L1_L1_elwiseadd_matmul')
    of_from_A2_B2_worker_to_C2_worker_0 = ObjectFifo(data_ty, tile_0_5, tile_0_3, depth=2, name='L1_L1_elwiseadd_matmul')
    of_from_mem_col2_to_shim_0 = ObjectFifo(data_d_ty, depth=2, name='SHIM_L2_L3_D3D4_col2')
    of_from_A3_B3_worker_to_C3_worker_0 = ObjectFifo(data_ty, tile_1_4, tile_1_2, depth=2, name='L1_L1_elwiseadd_matmul')
    of_from_A4_B4_worker_to_C4_worker_0 = ObjectFifo(data_ty, tile_1_5, tile_1_3, depth=2, name='L1_L1_elwiseadd_matmul')
    of_from_mem_col3_to_shim_0 = ObjectFifo(data_d_ty, depth=2, name='SHIM_L2_L3_D5D6_col3')
    of_from_A5_B5_worker_to_C5_worker_0 = ObjectFifo(data_ty, tile_2_4, tile_2_2, depth=2, name='L1_L1_elwiseadd_matmul')
    of_from_A6_B6_worker_to_C6_worker_0 = ObjectFifo(data_ty, tile_2_5, tile_2_3, depth=2, name='L1_L1_elwiseadd_matmul')
    of_from_mem_col4_to_shim_0 = ObjectFifo(data_d_ty, depth=2, name='SHIM_L2_L3_D7D8_col4')
    of_from_A7_B7_worker_to_C7_worker_0 = ObjectFifo(data_ty, tile_3_4, tile_3_2, depth=2, name='L1_L1_elwiseadd_matmul')
    of_from_A8_B8_worker_to_C8_worker_0 = ObjectFifo(data_ty, tile_3_5, tile_3_3, depth=2, name='L1_L1_elwiseadd_matmul')
    split_of_from_mem_col1_to_worker_A_key = of_from_shim_to_mem_col1_0.cons().split(offsets=[1024.0, 2048.0], obj_type=data_a_ty, depth=2, name='split_of_from_mem_col1_to_worker_A_key', placement=tile_0_1)
    split_of_from_mem_col1_to_worker_B_key = of_from_shim_to_mem_col1_1.cons().split(offsets=[1024.0, 2048.0], obj_type=data_b_ty, depth=2, name='split_of_from_mem_col1_to_worker_B_key', placement=tile_0_1)
    join_of_from_worker_mem_col1_to_D_key = of_from_mem_col1_to_shim_0.prod().join(obj_type=data_d_ty, depth=2, name='join_of_from_worker_mem_col1_to_D_key', placement=tile_0_1)
    split_of_from_mem_col2_to_worker_A_key = of_from_shim_to_mem_col2_0.cons().split(offsets=[3072.0, 4096.0], obj_type=data_a_ty, depth=2, name='split_of_from_mem_col2_to_worker_A_key', placement=tile_1_1)
    split_of_from_mem_col2_to_worker_B_key = of_from_shim_to_mem_col2_1.cons().split(offsets=[3072.0, 4096.0], obj_type=data_b_ty, depth=2, name='split_of_from_mem_col2_to_worker_B_key', placement=tile_1_1)
    join_of_from_worker_mem_col2_to_D_key = of_from_mem_col2_to_shim_0.prod().join(obj_type=data_d_ty, depth=2, name='join_of_from_worker_mem_col2_to_D_key', placement=tile_1_1)
    split_of_from_mem_col3_to_worker_A_key = of_from_shim_to_mem_col3_0.cons().split(offsets=[5120.0, 6144.0], obj_type=data_a_ty, depth=2, name='split_of_from_mem_col3_to_worker_A_key', placement=tile_2_1)
    split_of_from_mem_col3_to_worker_B_key = of_from_shim_to_mem_col3_1.cons().split(offsets=[5120.0, 6144.0], obj_type=data_b_ty, depth=2, name='split_of_from_mem_col3_to_worker_B_key', placement=tile_2_1)
    join_of_from_worker_mem_col3_to_D_key = of_from_mem_col3_to_shim_0.prod().join(obj_type=data_d_ty, depth=2, name='join_of_from_worker_mem_col3_to_D_key', placement=tile_2_1)
    split_of_from_mem_col4_to_worker_A_key = of_from_shim_to_mem_col4_0.cons().split(offsets=[7168.0, 8192.0], obj_type=data_a_ty, depth=2, name='split_of_from_mem_col4_to_worker_A_key', placement=tile_3_1)
    split_of_from_mem_col4_to_worker_B_key = of_from_shim_to_mem_col4_1.cons().split(offsets=[7168.0, 8192.0], obj_type=data_b_ty, depth=2, name='split_of_from_mem_col4_to_worker_B_key', placement=tile_3_1)
    join_of_from_worker_mem_col4_to_D_key = of_from_mem_col4_to_shim_0.prod().join(obj_type=data_d_ty, depth=2, name='join_of_from_worker_mem_col4_to_D_key', placement=tile_3_1)

    # Define external C/C++ kernel functions
    external_A1_B1_worker = ExternalFunction(
    name="eltwise_add_bf16_scalar",
    source_file="../../../aie_kernels/aie2/add.cc",
    arg_types=[data_ty] * 3,
    include_dirs=['/scratch/andrewa/mlir-aie/aie_kernels/']
)
    external_C1_worker = ExternalFunction(
    name="matmul_bf16",
    source_file="../../../aie_kernels/aie2/matmul.cc",
    arg_types=[data_ty] * 3,
    include_dirs=['/scratch/andrewa/mlir-aie/aie_kernels/']
)
    external_A2_B2_worker = ExternalFunction(
    name="eltwise_add_bf16_scalar",
    source_file="../../../aie_kernels/aie2/add.cc",
    arg_types=[data_ty] * 3,
    include_dirs=['/scratch/andrewa/mlir-aie/aie_kernels/']
)
    external_C2_worker = ExternalFunction(
    name="matmul_bf16",
    source_file="../../../aie_kernels/aie2/matmul.cc",
    arg_types=[data_ty] * 3,
    include_dirs=['/scratch/andrewa/mlir-aie/aie_kernels/']
)
    external_A3_B3_worker = ExternalFunction(
    name="eltwise_add_bf16_scalar",
    source_file="../../../aie_kernels/aie2/add.cc",
    arg_types=[data_ty] * 3,
    include_dirs=['/scratch/andrewa/mlir-aie/aie_kernels/']
)
    external_C3_worker = ExternalFunction(
    name="matmul_bf16",
    source_file="../../../aie_kernels/aie2/matmul.cc",
    arg_types=[data_ty] * 3,
    include_dirs=['/scratch/andrewa/mlir-aie/aie_kernels/']
)
    external_A4_B4_worker = ExternalFunction(
    name="eltwise_add_bf16_scalar",
    source_file="../../../aie_kernels/aie2/add.cc",
    arg_types=[data_ty] * 3,
    include_dirs=['/scratch/andrewa/mlir-aie/aie_kernels/']
)
    external_C4_worker = ExternalFunction(
    name="matmul_bf16",
    source_file="../../../aie_kernels/aie2/matmul.cc",
    arg_types=[data_ty] * 3,
    include_dirs=['/scratch/andrewa/mlir-aie/aie_kernels/']
)
    external_A5_B5_worker = ExternalFunction(
    name="eltwise_add_bf16_scalar",
    source_file="../../../aie_kernels/aie2/add.cc",
    arg_types=[data_ty] * 3,
    include_dirs=['/scratch/andrewa/mlir-aie/aie_kernels/']
)
    external_C5_worker = ExternalFunction(
    name="matmul_bf16",
    source_file="../../../aie_kernels/aie2/matmul.cc",
    arg_types=[data_ty] * 3,
    include_dirs=['/scratch/andrewa/mlir-aie/aie_kernels/']
)
    external_A6_B6_worker = ExternalFunction(
    name="eltwise_add_bf16_scalar",
    source_file="../../../aie_kernels/aie2/add.cc",
    arg_types=[data_ty] * 3,
    include_dirs=['/scratch/andrewa/mlir-aie/aie_kernels/']
)
    external_C6_worker = ExternalFunction(
    name="matmul_bf16",
    source_file="../../../aie_kernels/aie2/matmul.cc",
    arg_types=[data_ty] * 3,
    include_dirs=['/scratch/andrewa/mlir-aie/aie_kernels/']
)
    external_A7_B7_worker = ExternalFunction(
    name="eltwise_add_bf16_scalar",
    source_file="../../../aie_kernels/aie2/add.cc",
    arg_types=[data_ty] * 3,
    include_dirs=['/scratch/andrewa/mlir-aie/aie_kernels/']
)
    external_C7_worker = ExternalFunction(
    name="matmul_bf16",
    source_file="../../../aie_kernels/aie2/matmul.cc",
    arg_types=[data_ty] * 3,
    include_dirs=['/scratch/andrewa/mlir-aie/aie_kernels/']
)
    external_A8_B8_worker = ExternalFunction(
    name="eltwise_add_bf16_scalar",
    source_file="../../../aie_kernels/aie2/add.cc",
    arg_types=[data_ty] * 3,
    include_dirs=['/scratch/andrewa/mlir-aie/aie_kernels/']
)
    external_C8_worker = ExternalFunction(
    name="matmul_bf16",
    source_file="../../../aie_kernels/aie2/matmul.cc",
    arg_types=[data_ty] * 3,
    include_dirs=['/scratch/andrewa/mlir-aie/aie_kernels/']
)

    # Define core functions for each compute node
    def core_fn_A1_B1_worker(of_in1, of_in2, of_out1, external_A1_B1_worker):
        elem_in1 = of_in1.acquire(1)
        elem_in2 = of_in2.acquire(1)
        elem_out1 = of_out1.acquire(1)
        external_A1_B1_worker(elem_in1, elem_in2, elem_out1)
        of_in1.release(1)
        of_in2.release(1)
        of_out1.release(1)
    def core_fn_C1_worker(of_in1, of_out1, external_C1_worker):
        elem_in1 = of_in1.acquire(1)
        elem_out1 = of_out1.acquire(1)
        external_C1_worker(elem_in1, elem_out1)
        of_in1.release(1)
        of_out1.release(1)
    def core_fn_A2_B2_worker(of_in1, of_in2, of_out1, external_A2_B2_worker):
        elem_in1 = of_in1.acquire(1)
        elem_in2 = of_in2.acquire(1)
        elem_out1 = of_out1.acquire(1)
        external_A2_B2_worker(elem_in1, elem_in2, elem_out1)
        of_in1.release(1)
        of_in2.release(1)
        of_out1.release(1)
    def core_fn_C2_worker(of_in1, of_out1, external_C2_worker):
        elem_in1 = of_in1.acquire(1)
        elem_out1 = of_out1.acquire(1)
        external_C2_worker(elem_in1, elem_out1)
        of_in1.release(1)
        of_out1.release(1)
    def core_fn_A3_B3_worker(of_in1, of_in2, of_out1, external_A3_B3_worker):
        elem_in1 = of_in1.acquire(1)
        elem_in2 = of_in2.acquire(1)
        elem_out1 = of_out1.acquire(1)
        external_A3_B3_worker(elem_in1, elem_in2, elem_out1)
        of_in1.release(1)
        of_in2.release(1)
        of_out1.release(1)
    def core_fn_C3_worker(of_in1, of_out1, external_C3_worker):
        elem_in1 = of_in1.acquire(1)
        elem_out1 = of_out1.acquire(1)
        external_C3_worker(elem_in1, elem_out1)
        of_in1.release(1)
        of_out1.release(1)
    def core_fn_A4_B4_worker(of_in1, of_in2, of_out1, external_A4_B4_worker):
        elem_in1 = of_in1.acquire(1)
        elem_in2 = of_in2.acquire(1)
        elem_out1 = of_out1.acquire(1)
        external_A4_B4_worker(elem_in1, elem_in2, elem_out1)
        of_in1.release(1)
        of_in2.release(1)
        of_out1.release(1)
    def core_fn_C4_worker(of_in1, of_out1, external_C4_worker):
        elem_in1 = of_in1.acquire(1)
        elem_out1 = of_out1.acquire(1)
        external_C4_worker(elem_in1, elem_out1)
        of_in1.release(1)
        of_out1.release(1)
    def core_fn_A5_B5_worker(of_in1, of_in2, of_out1, external_A5_B5_worker):
        elem_in1 = of_in1.acquire(1)
        elem_in2 = of_in2.acquire(1)
        elem_out1 = of_out1.acquire(1)
        external_A5_B5_worker(elem_in1, elem_in2, elem_out1)
        of_in1.release(1)
        of_in2.release(1)
        of_out1.release(1)
    def core_fn_C5_worker(of_in1, of_out1, external_C5_worker):
        elem_in1 = of_in1.acquire(1)
        elem_out1 = of_out1.acquire(1)
        external_C5_worker(elem_in1, elem_out1)
        of_in1.release(1)
        of_out1.release(1)
    def core_fn_A6_B6_worker(of_in1, of_in2, of_out1, external_A6_B6_worker):
        elem_in1 = of_in1.acquire(1)
        elem_in2 = of_in2.acquire(1)
        elem_out1 = of_out1.acquire(1)
        external_A6_B6_worker(elem_in1, elem_in2, elem_out1)
        of_in1.release(1)
        of_in2.release(1)
        of_out1.release(1)
    def core_fn_C6_worker(of_in1, of_out1, external_C6_worker):
        elem_in1 = of_in1.acquire(1)
        elem_out1 = of_out1.acquire(1)
        external_C6_worker(elem_in1, elem_out1)
        of_in1.release(1)
        of_out1.release(1)
    def core_fn_A7_B7_worker(of_in1, of_in2, of_out1, external_A7_B7_worker):
        elem_in1 = of_in1.acquire(1)
        elem_in2 = of_in2.acquire(1)
        elem_out1 = of_out1.acquire(1)
        external_A7_B7_worker(elem_in1, elem_in2, elem_out1)
        of_in1.release(1)
        of_in2.release(1)
        of_out1.release(1)
    def core_fn_C7_worker(of_in1, of_out1, external_C7_worker):
        elem_in1 = of_in1.acquire(1)
        elem_out1 = of_out1.acquire(1)
        external_C7_worker(elem_in1, elem_out1)
        of_in1.release(1)
        of_out1.release(1)
    def core_fn_A8_B8_worker(of_in1, of_in2, of_out1, external_A8_B8_worker):
        elem_in1 = of_in1.acquire(1)
        elem_in2 = of_in2.acquire(1)
        elem_out1 = of_out1.acquire(1)
        external_A8_B8_worker(elem_in1, elem_in2, elem_out1)
        of_in1.release(1)
        of_in2.release(1)
        of_out1.release(1)
    def core_fn_C8_worker(of_in1, of_out1, external_C8_worker):
        elem_in1 = of_in1.acquire(1)
        elem_out1 = of_out1.acquire(1)
        external_C8_worker(elem_in1, elem_out1)
        of_in1.release(1)
        of_out1.release(1)

    # Define workers to execute core functions on tiles
    worker_A1_B1_worker = Worker(core_fn_A1_B1_worker, [split_of_from_mem_col1_to_worker_A_key.cons(), split_of_from_mem_col1_to_worker_B_key.cons(), of_from_A1_B1_worker_to_C1_worker_0.prod(), external_A1_B1_worker], placement=tile_0_4, while_true=False, stack_size=1024, allocation_scheme='heap', trace=False, trace_events=None)
    worker_C1_worker = Worker(core_fn_C1_worker, [of_from_A1_B1_worker_to_C1_worker_0.cons(), join_of_from_worker_mem_col1_to_D_key.prod(), external_C1_worker], placement=tile_0_2, while_true=False, stack_size=1024, allocation_scheme='heap', trace=False, trace_events=None)
    worker_A2_B2_worker = Worker(core_fn_A2_B2_worker, [split_of_from_mem_col1_to_worker_A_key.cons(), split_of_from_mem_col1_to_worker_B_key.cons(), of_from_A2_B2_worker_to_C2_worker_0.prod(), external_A2_B2_worker], placement=tile_0_5, while_true=False, stack_size=1024, allocation_scheme='heap', trace=False, trace_events=None)
    worker_C2_worker = Worker(core_fn_C2_worker, [of_from_A2_B2_worker_to_C2_worker_0.cons(), join_of_from_worker_mem_col1_to_D_key.prod(), external_C2_worker], placement=tile_0_3, while_true=False, stack_size=1024, allocation_scheme='heap', trace=False, trace_events=None)
    worker_A3_B3_worker = Worker(core_fn_A3_B3_worker, [split_of_from_mem_col2_to_worker_A_key.cons(), split_of_from_mem_col2_to_worker_B_key.cons(), of_from_A3_B3_worker_to_C3_worker_0.prod(), external_A3_B3_worker], placement=tile_1_4, while_true=False, stack_size=1024, allocation_scheme='heap', trace=False, trace_events=None)
    worker_C3_worker = Worker(core_fn_C3_worker, [of_from_A3_B3_worker_to_C3_worker_0.cons(), join_of_from_worker_mem_col2_to_D_key.prod(), external_C3_worker], placement=tile_1_2, while_true=False, stack_size=1024, allocation_scheme='heap', trace=False, trace_events=None)
    worker_A4_B4_worker = Worker(core_fn_A4_B4_worker, [split_of_from_mem_col2_to_worker_A_key.cons(), split_of_from_mem_col2_to_worker_B_key.cons(), of_from_A4_B4_worker_to_C4_worker_0.prod(), external_A4_B4_worker], placement=tile_1_5, while_true=False, stack_size=1024, allocation_scheme='heap', trace=False, trace_events=None)
    worker_C4_worker = Worker(core_fn_C4_worker, [of_from_A4_B4_worker_to_C4_worker_0.cons(), join_of_from_worker_mem_col2_to_D_key.prod(), external_C4_worker], placement=tile_1_3, while_true=False, stack_size=1024, allocation_scheme='heap', trace=False, trace_events=None)
    worker_A5_B5_worker = Worker(core_fn_A5_B5_worker, [split_of_from_mem_col3_to_worker_A_key.cons(), split_of_from_mem_col3_to_worker_B_key.cons(), of_from_A5_B5_worker_to_C5_worker_0.prod(), external_A5_B5_worker], placement=tile_2_4, while_true=False, stack_size=1024, allocation_scheme='heap', trace=False, trace_events=None)
    worker_C5_worker = Worker(core_fn_C5_worker, [of_from_A5_B5_worker_to_C5_worker_0.cons(), join_of_from_worker_mem_col3_to_D_key.prod(), external_C5_worker], placement=tile_2_2, while_true=False, stack_size=1024, allocation_scheme='heap', trace=False, trace_events=None)
    worker_A6_B6_worker = Worker(core_fn_A6_B6_worker, [split_of_from_mem_col3_to_worker_A_key.cons(), split_of_from_mem_col3_to_worker_B_key.cons(), of_from_A6_B6_worker_to_C6_worker_0.prod(), external_A6_B6_worker], placement=tile_2_5, while_true=False, stack_size=1024, allocation_scheme='heap', trace=False, trace_events=None)
    worker_C6_worker = Worker(core_fn_C6_worker, [of_from_A6_B6_worker_to_C6_worker_0.cons(), join_of_from_worker_mem_col3_to_D_key.prod(), external_C6_worker], placement=tile_2_3, while_true=False, stack_size=1024, allocation_scheme='heap', trace=False, trace_events=None)
    worker_A7_B7_worker = Worker(core_fn_A7_B7_worker, [split_of_from_mem_col4_to_worker_A_key.cons(), split_of_from_mem_col4_to_worker_B_key.cons(), of_from_A7_B7_worker_to_C7_worker_0.prod(), external_A7_B7_worker], placement=tile_3_4, while_true=False, stack_size=1024, allocation_scheme='heap', trace=False, trace_events=None)
    worker_C7_worker = Worker(core_fn_C7_worker, [of_from_A7_B7_worker_to_C7_worker_0.cons(), join_of_from_worker_mem_col4_to_D_key.prod(), external_C7_worker], placement=tile_3_2, while_true=False, stack_size=1024, allocation_scheme='heap', trace=False, trace_events=None)
    worker_A8_B8_worker = Worker(core_fn_A8_B8_worker, [split_of_from_mem_col4_to_worker_A_key.cons(), split_of_from_mem_col4_to_worker_B_key.cons(), of_from_A8_B8_worker_to_C8_worker_0.prod(), external_A8_B8_worker], placement=tile_3_5, while_true=False, stack_size=1024, allocation_scheme='heap', trace=False, trace_events=None)
    worker_C8_worker = Worker(core_fn_C8_worker, [of_from_A8_B8_worker_to_C8_worker_0.cons(), join_of_from_worker_mem_col4_to_D_key.prod(), external_C8_worker], placement=tile_3_3, while_true=False, stack_size=1024, allocation_scheme='heap', trace=False, trace_events=None)

    # Define runtime sequence for starting workers and moving data
    rt = Runtime()
    with rt.sequence(data_a_ty, data_b_ty, data_d_ty) as (A,B,D):
       Workers = [worker_A1_B1_worker, worker_C1_worker, worker_A2_B2_worker, worker_C2_worker, worker_A3_B3_worker, worker_C3_worker, worker_A4_B4_worker, worker_C4_worker, worker_A5_B5_worker, worker_C5_worker, worker_A6_B6_worker, worker_C6_worker, worker_A7_B7_worker, worker_C7_worker, worker_A8_B8_worker, worker_C8_worker]
       rt.start(*Workers)
       rt.fill(in_fifo=of_from_shim_to_mem_col1_0.prod(), in_data=A, tap=TensorAccessPattern(tensor_dims=[1,1024], offset=2048.0, sizes=[1024, (data_size/4)/1024], strides=[1,1024]))
       rt.fill(in_fifo=of_from_shim_to_mem_col2_0.prod(), in_data=A, tap=TensorAccessPattern(tensor_dims=[1,1024], offset=4096.0, sizes=[1024, (data_size/4)/1024], strides=[1,1024]))
       rt.fill(in_fifo=of_from_shim_to_mem_col3_0.prod(), in_data=A, tap=TensorAccessPattern(tensor_dims=[1,1024], offset=6144.0, sizes=[1024, (data_size/4)/1024], strides=[1,1024]))
       rt.fill(in_fifo=of_from_shim_to_mem_col4_0.prod(), in_data=A, tap=TensorAccessPattern(tensor_dims=[1,1024], offset=8192.0, sizes=[1024, (data_size/4)/1024], strides=[1,1024]))
       rt.fill(in_fifo=of_from_shim_to_mem_col1_1.prod(), in_data=B, tap=TensorAccessPattern(tensor_dims=[1,1024], offset=2048.0, sizes=[1024, (data_size/4)/1024], strides=[1,1024]))
       rt.fill(in_fifo=of_from_shim_to_mem_col2_1.prod(), in_data=B, tap=TensorAccessPattern(tensor_dims=[1,1024], offset=4096.0, sizes=[1024, (data_size/4)/1024], strides=[1,1024]))
       rt.fill(in_fifo=of_from_shim_to_mem_col3_1.prod(), in_data=B, tap=TensorAccessPattern(tensor_dims=[1,1024], offset=6144.0, sizes=[1024, (data_size/4)/1024], strides=[1,1024]))
       rt.fill(in_fifo=of_from_shim_to_mem_col4_1.prod(), in_data=B, tap=TensorAccessPattern(tensor_dims=[1,1024], offset=8192.0, sizes=[1024, (data_size/4)/1024], strides=[1,1024]))
       rt.drain(out_fifo=of_from_mem_col1_to_shim_0.cons(), out_data=D, tap=TensorAccessPattern(tensor_dims=[1,1024], offset=2048.0, sizes=[1024, (data_size/4)/1024], strides=[1, 1024]))
       rt.drain(out_fifo=of_from_mem_col2_to_shim_0.cons(), out_data=D, tap=TensorAccessPattern(tensor_dims=[1,1024], offset=4096.0, sizes=[1024, (data_size/4)/1024], strides=[1, 1024]))
       rt.drain(out_fifo=of_from_mem_col3_to_shim_0.cons(), out_data=D, tap=TensorAccessPattern(tensor_dims=[1,1024], offset=6144.0, sizes=[1024, (data_size/4)/1024], strides=[1, 1024]))
       rt.drain(out_fifo=of_from_mem_col4_to_shim_0.cons(), out_data=D, tap=TensorAccessPattern(tensor_dims=[1,1024], offset=8192.0, sizes=[1024, (data_size/4)/1024], strides=[1, 1024]))
    my_program = Program(iron.get_current_device(), rt)
    my_program = my_program.resolve_program(SequentialPlacer())
    return my_program

def main():
    datatype = bfloat16
    data_size = 8192
    inputA = iron.rand(data_size, dtype=datatype, device="npu")
    inputB = iron.arange(data_size, dtype=datatype, device="npu", step=-1)
    outputD = iron.zeros(data_size, dtype=datatype, device="npu")
    generated_design(inputA, inputB, outputD)
    print(outputD)
if __name__ == "__main__":
    main()
