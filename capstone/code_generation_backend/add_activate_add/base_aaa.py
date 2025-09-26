from aie.iron import Program, Runtime, Worker, ObjectFifo
from aie.iron.placers import SequentialPlacer
from aie.iron.device.tile import AnyComputeTile
from aie.iron import ExternalFunction, jit
from aie.iron.device import Tile
import numpy as np
import aie.iron as iron
from ml_dtypes import bfloat16


# Defining Problem:
#...    We want a benchmark code to test the code generators output against to compare for readability, 
#...    optimalization(no duplication), dataflow correctness, syntax, and input/output correctness.
#...    Below we have setup an example:
#...    1. We want the data to first go through an elementwise add operation.
#...    2. After the add, we want the data to go through an activation function.
#...    3. Finally the data should then go through another elementwise add operation.

#...    This dataflow we have described will use 4 tiles:
#...    1 Shim tile for streaming data from (input) and to (output) main memory.
#...    2 Compute tiles for the element-wise add operation.
#...    1 compute tile for the relu activation operation.

#...    The data should have object fifos all streaming the data from one tile to the next so 
#...    each input and output are streamed at each ~cycle.


@iron.jit(is_placed=False)
def base_aaa(inputA, inputB, outputC):
    data_size = inputA.numel()
    element_type = bfloat16
    data_ty = np.ndarray[(data_size,), np.dtype[element_type]]

    # Object fifos goes here... --------------------------------------------\/
    inA_CT1 = ObjectFifo(data_ty, name="A_L3_L1_CT1") # input A *** shim-> compute tile 1
    inB_CT1 = ObjectFifo(data_ty, name="B_L3_L1") # input B *** shim-> compute tile 1
    CT1_CT2 = ObjectFifo(data_ty, name="tempC_CT1_CT2") # output C temp *** compute tile 1-> compute tile 2
    inA_CT3 = ObjectFifo(data_ty, name="A_L3_L1_CT3") # input A *** shim-> compute tile 3
    CT2_CT3 = ObjectFifo(data_ty, name="tempC_CT2_CT3") # compute tile 2-> compute tile 3
    outC = ObjectFifo(data_ty, name="C_L1_L3") # compute tile 3 -> shim

    #Define kernels here... ------------------------------------------------\/
    element_wise_add = ExternalFunction(
        "eltwise_add",
        source_file="../../aie_kernels/aie2/add.cc",
        arg_types=[data_ty, data_ty, data_ty],
    )

    relu_activation = ExternalFunction(
        "relu",
        source_file="../../aie_kernels/aie2/relu.cc",
        arg_types=[data_ty, data_ty],
    )

    # Core functions go here... --------------------------------------------\/
    def core_fn_eltwise_add(of_inA, of_inB, of_outC, element_wise_add):
        elemA = of_inA.acquire(1)
        elemB = of_inB.acquire(1)
        elemC = of_outC.acquire(1)
        element_wise_add(elemA, elemB, elemC)
        of_inA.release(1)
        of_inB.release(1)
        of_outC.release(1)

    def core_fn_relu(of_in, of_out, relu_activation):
        elem_in = of_in.acquire(1)
        elem_out = of_out.acquire(1)
        relu_activation(elem_in, elem_out)
        of_in.release(1)
        of_out.release(1)

    # Workers go here... ---------------------------------------------------\/
    worker1 = Worker(core_fn_eltwise_add, [inA_CT1.cons(), inB_CT1.cons(), CT1_CT2.prod(), element_wise_add], placement=Tile(0, 2))
    worker2 = Worker(core_fn_relu, [CT1_CT2.cons(), CT2_CT3.prod(), relu_activation], placement=Tile(0, 3))
    worker3 = Worker(core_fn_eltwise_add, [inA_CT3.cons(), CT2_CT3.cons(), outC.prod(), element_wise_add], placement=Tile(0, 4))

    # Runtime data movement
    rt = Runtime()
    with rt.sequence(data_ty, data_ty, data_ty) as (A, B, C):
        rt.start(worker1)
        rt.start(worker2)
        rt.start(worker3)
        rt.fill(inA_CT1.prod(), A)
        rt.fill(inB_CT1.prod(), B)
        rt.fill(inA_CT3.prod(), A)
        rt.drain(outC.cons(), C, wait=True)

    my_program = Program(iron.get_current_device(), rt)
    my_program = my_program.resolve_program(SequentialPlacer())
    return my_program

def main():
    # Define Data here ... -------------------------------------------------\/
    datatype = bfloat16
    data_size = 256
    inputA = iron.arange(data_size, dtype=datatype, device="npu")
    inputB = iron.arange(data_size, dtype=datatype, device="npu")
    outputC = iron.zeros(data_size, dtype=datatype, device="npu")
    base_aaa(inputA, inputB, outputC)
    print(outputC)

if __name__ == "__main__":
    main()