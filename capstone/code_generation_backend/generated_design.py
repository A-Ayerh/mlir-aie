import aie.iron as iron
from aie.iron import ExternalFunction, jit
from aie.iron import ObjectFifo, Worker, Runtime, Program
from aie.iron.placers import SequentialPlacer
from aie.iron.device import Tile
import numpy as np
from ml_dtypes import bfloat16
            
            
@jit(is_placed=False)
def generated_design(inputA, inputB, outputC):

    element_type = bfloat16
    data_size = inputA.numel()
    data_ty = np.ndarray[(data_size,), np.dtype[element_type]]
    # Define tiles for compute and shim nodes
    tile_0_0 = tile(0, 0)
    tile_0_2 = tile(0, 2)
    tile_0_3 = tile(0, 3)
    tile_0_4 = tile(0, 4)

    # Define external C/C++ kernel functions
    external_CT1 = ExternalFunction(
    name="eltwise_add_bf16_scalar",
    source_file="../../../aie_kernels/aie2/add.cc",
    arg_types=[data_ty] * 3,
    include_dirs=['/scratch/andrewa/mlir-aie/aie_kernels/']
)
    external_CT2 = ExternalFunction(
    name="bf16_relu",
    source_file="../../../aie_kernels/aie2/relu.cc",
    arg_types=[data_ty] * 2,
    include_dirs=['/scratch/andrewa/mlir-aie/aie_kernels/']
)
    external_CT3 = ExternalFunction(
    name="eltwise_add_bf16_scalar",
    source_file="../../../aie_kernels/aie2/add.cc",
    arg_types=[data_ty] * 3,
    include_dirs=['/scratch/andrewa/mlir-aie/aie_kernels/']
)

    # Define core functions for each compute node
    def core_fn_CT1(of_in1, of_in2, of_out1, external_CT1):
        elem_in1 = of_in1.acquire(1)
        elem_in2 = of_in2.acquire(1)
        elem_out1 = of_out1.acquire(1)
        external_CT1(elem_in1, elem_in2, elem_out1)
        of_in1.release(1)
        of_in2.release(1)
        of_out1.release(1)
    def core_fn_CT2(of_in1, of_out1, external_CT2):
        elem_in1 = of_in1.acquire(1)
        elem_out1 = of_out1.acquire(1)
        external_CT2(elem_in1, elem_out1)
        of_in1.release(1)
        of_out1.release(1)
    def core_fn_CT3(of_in1, of_in2, of_out1, external_CT3):
        elem_in1 = of_in1.acquire(1)
        elem_in2 = of_in2.acquire(1)
        elem_out1 = of_out1.acquire(1)
        external_CT3(elem_in1, elem_in2, elem_out1)
        of_in1.release(1)
        of_in2.release(1)
        of_out1.release(1)

    # Define workers to execute core functions on tiles
    worker_CT1 = Worker(core_fn_CT1, [of_from_shim_to_CT1_0.cons(), of_from_shim_to_CT1_1.cons(), of_from_CT1_to_CT2_0.prod(), external_CT1], placement=tile_0_2, while_true=False, stack_size=1024, allocation_scheme='heap', trace=False, trace_events=None)
    worker_CT2 = Worker(core_fn_CT2, [of_from_CT1_to_CT2_0.cons(), of_from_CT2_to_CT3_0.prod(), external_CT2], placement=tile_0_3, while_true=False, stack_size=1024, allocation_scheme='heap', trace=False, trace_events=None)
    worker_CT3 = Worker(core_fn_CT3, [of_from_shim_to_CT3_0.cons(), of_from_CT2_to_CT3_0.cons(), of_from_CT3_to_shim_0.prod(), external_CT3], placement=tile_0_4, while_true=False, stack_size=1024, allocation_scheme='heap', trace=False, trace_events=None)

    # Define object FIFOs for data streaming between tiles
    of_from_shim_to_CT1_0 = ObjectFifo(data_ty, tile_0_0, tile_0_2, depth=1, name='A_L3_L1_CT1')
    of_from_shim_to_CT1_1 = ObjectFifo(data_ty, tile_0_0, tile_0_2, depth=1, name='B_L3_L1')
    of_from_shim_to_CT3_0 = ObjectFifo(data_ty, tile_0_0, tile_0_4, depth=1, name='A_L3_L1_CT3')
    of_from_CT1_to_CT2_0 = ObjectFifo(data_ty, tile_0_2, tile_0_3, depth=1, name='tempC_CT1_CT2')
    of_from_CT2_to_CT3_0 = ObjectFifo(data_ty, tile_0_3, tile_0_4, depth=1, name='tempC_CT2_CT3')
    of_from_CT3_to_shim_0 = ObjectFifo(data_ty, tile_0_4, tile_0_0, depth=1, name='C_L1_L3')

    # Define runtime sequence for starting workers and moving data
    rt = Runtime()
    with rt.sequence(data_ty, data_ty, data_ty) as (A,B,C):
        rt.start(worker_CT1)
        rt.start(worker_CT2)
        rt.start(worker_CT3)
        rt.fill(of_from_shim_to_CT1_0.prod(), A)
        rt.fill(of_from_shim_to_CT3_0.prod(), A)
        rt.fill(of_from_shim_to_CT1_1.prod(), B)
        rt.drain(of_from_CT3_to_shim_0.cons(), C, wait=True)
    my_program = Program(iron.get_current_device(), rt)
    my_program = my_program.resolve_program(SequentialPlacer())
    return my_program

def main():
    datatype = bfloat16
    data_size = 256
    inputA = iron.arange(data_size, dtype=datatype, device="npu")
    inputB = iron.arange(data_size, dtype=datatype, device="npu")
    outputC = iron.zeros(data_size, dtype=datatype, device="npu")
    generated_design(inputA, inputB, outputC)
    print(outputC)
if __name__ == "__main__":
    main()
