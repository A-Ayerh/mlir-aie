#Define imports here... ------------------------------------------------\/
from aie.iron import Program, Runtime, Worker, ObjectFifo
from aie.iron.placers import SequentialPlacer
from aie.iron.device.tile import AnyComputeTile
from aie.iron import ExternalFunction, jit
from aie.iron.device import Tile
import numpy as np
import aie.iron as iron
from ml_dtypes import bfloat16

from aie.helpers.taplib import TensorAccessPattern


@iron.jit(is_placed=False)
def base_aaa(inputA, inputB, outputD):

    data_a_ty = np.ndarray[(inputA.numel(),), np.dtype[bfloat16]]
    data_b_ty = np.ndarray[(inputB.numel(),), np.dtype[bfloat16]]
    data_d_ty = np.ndarray[(outputD.numel(),), np.dtype[bfloat16]]

    # Object fifos goes here... --------------------------------------------\/
    # L3_L2 Object Fifos:
    SHIM_L3_L2_A1A2_col1 = ObjectFifo(obj_type=data_a_ty, depth=2, name="SHIM_L3_L2_A1A2_col1")
    SHIM_L3_L2_B1B2_col1 = ObjectFifo(obj_type=data_b_ty, depth=2, name="SHIM_L3_L2_B1B2_col1")

    SHIM_L3_L2_A3A4_col2 = ObjectFifo(obj_type=data_a_ty, depth=2, name="SHIM_L3_L2_A3A4_col2")
    SHIM_L3_L2_B3B4_col2 = ObjectFifo(obj_type=data_b_ty, depth=2, name="SHIM_L3_L2_B3B4_col2")

    SHIM_L3_L2_A5A6_col3 = ObjectFifo(obj_type=data_a_ty, depth=2, name="SHIM_L3_L2_A5A6_col3")
    SHIM_L3_L2_B5B6_col3 = ObjectFifo(obj_type=data_b_ty, depth=2, name="SHIM_L3_L2_B5B6_col3")

    SHIM_L3_L2_A7A8_col4 = ObjectFifo(obj_type=data_a_ty, depth=2, name="SHIM_L3_L2_A7A8_col4")
    SHIM_L3_L2_B7B8_col4 = ObjectFifo(obj_type=data_b_ty, depth=2, name="SHIM_L3_L2_B7B8_col4")
    
    # L2_L1 Object Fifos:
    MEM_L2_L1_A1A2_col1 = SHIM_L3_L2_A1A2_col1.cons().split(
        obj_types=[data_a_ty,data_a_ty],
        offsets=[(int(inputA.numel()) / 8) * 0, (inputA.numel() / 8) * 1],
        names=["MEM_L2_L1_A1_col1", "MEM_L2_L1_A2_col1"],
        placement=Tile(0, 1)
    )

    MEM_L2_L1_B1B2_col1 = SHIM_L3_L2_B1B2_col1.cons().split(
        obj_types=[data_b_ty,data_b_ty],
        offsets=[(int(inputB.numel()) / 8) * 0, (int(inputB.numel()) / 8) * 1],
        names=["MEM_L2_L1_B1_col1", "MEM_L2_L1_B2_col1"],
        placement=Tile(0, 1)
    )

    MEM_L2_L1_A3A4_col2 = SHIM_L3_L2_A3A4_col2.cons().split(
        obj_types=[data_a_ty,data_a_ty],
        offsets=[(int(inputA.numel()) / 8) * 2, (int(inputA.numel()) / 8) * 3],
        names=["MEM_L2_L1_A3_col2", "MEM_L2_L1_A4_col2"],
        placement=Tile(1, 1)
    )

    MEM_L2_L1_B3B4_col2 = SHIM_L3_L2_B3B4_col2.cons().split(
        obj_types=[data_b_ty,data_b_ty],
        offsets=[(int(inputB.numel()) / 8) * 2, (int(inputB.numel()) / 8) * 3],
        names=["MEM_L2_L1_B3_col2", "MEM_L2_L1_B4_col2"],
        placement=Tile(1, 1)
    )

    MEM_L2_L1_A5A6_col3 = SHIM_L3_L2_A5A6_col3.cons().split(
        obj_types=[data_a_ty,data_a_ty],
        offsets=[(int(inputA.numel()) / 8) * 4, (int(inputA.numel()) / 8) * 5],
        names=["MEM_L2_L1_A5_col3", "MEM_L2_L1_A6_col3"],
        placement=Tile(2, 1)
    )

    MEM_L2_L1_B5B6_col3 = SHIM_L3_L2_B5B6_col3.cons().split(
        obj_types=[data_b_ty,data_b_ty],
        offsets=[(int(inputB.numel()) / 8) * 4, (int(inputB.numel()) / 8) * 5],
        names=["MEM_L2_L1_B5_col3", "MEM_L2_L1_B6_col3"],
        placement=Tile(2, 1)
    )

    MEM_L2_L1_A7A8_col4 = SHIM_L3_L2_A7A8_col4.cons().split(
        obj_types=[data_a_ty,data_a_ty],
        offsets=[(int(inputA.numel()) / 8) * 6, (int(inputA.numel()) / 8) * 7],
        names=["MEM_L2_L1_A7_col4", "MEM_L2_L1_A8_col4"],
        placement=Tile(3, 1)
    )

    MEM_L2_L1_B7B8_col4 = SHIM_L3_L2_B7B8_col4.cons().split(
        obj_types=[data_b_ty,data_b_ty],
        offsets=[(int(inputB.numel()) / 8) * 6, (int(inputB.numel()) / 8) * 7],
        names=["MEM_L2_L1_B7_col4", "MEM_L2_L1_B8_col4"],
        placement=Tile(3, 1)
    )

    # L1_L1 Object Fifos:
    L1_L1_elwiseadd_relu_1 = ObjectFifo(obj_type=data_d_ty, depth=2, name="L1_L1_elwiseadd_relu_1")
    L1_L1_elwiseadd_relu_2 = ObjectFifo(obj_type=data_d_ty, depth=2, name="L1_L1_elwiseadd_relu_2")
    L1_L1_elwiseadd_relu_3 = ObjectFifo(obj_type=data_d_ty, depth=2, name="L1_L1_elwiseadd_relu_3")
    L1_L1_elwiseadd_relu_4 = ObjectFifo(obj_type=data_d_ty, depth=2, name="L1_L1_elwiseadd_relu_4")
    L1_L1_elwiseadd_relu_5 = ObjectFifo(obj_type=data_d_ty, depth=2, name="L1_L1_elwiseadd_relu_5")
    L1_L1_elwiseadd_relu_6 = ObjectFifo(obj_type=data_d_ty, depth=2, name="L1_L1_elwiseadd_relu_6")
    L1_L1_elwiseadd_relu_7 = ObjectFifo(obj_type=data_d_ty, depth=2, name="L1_L1_elwiseadd_relu_7")
    L1_L1_elwiseadd_relu_8 = ObjectFifo(obj_type=data_d_ty, depth=2, name="L1_L1_elwiseadd_relu_8")

    # L2_L3 Object Fifos:
    SHIM_L2_L3_D1D2_col1 = ObjectFifo(obj_type=data_d_ty, depth=2, name="SHIM_L2_L3_D1D2_col1")
    SHIM_L2_L3_D3D4_col2 = ObjectFifo(obj_type=data_d_ty, depth=2, name="SHIM_L2_L3_D3D4_col2")
    SHIM_L2_L3_D5D6_col3 = ObjectFifo(obj_type=data_d_ty, depth=2, name="SHIM_L2_L3_D5D6_col3")
    SHIM_L2_L3_D7D8_col4 = ObjectFifo(obj_type=data_d_ty, depth=2, name="SHIM_L2_L3_D7D8_col4")

    # L1_L2 Object Fifos:
    MEM_L1_L2_D1D2_col1 = SHIM_L2_L3_D1D2_col1.prod().join(
        obj_types=[data_d_ty,data_d_ty],
        names=["MEM_L1_L2_D1_col1", "MEM_L1_L2_D2_col1"], 
        placement=Tile(0,1),
        offsets=[(int(outputD.numel()) / 8) * 0, (outputD.numel() / 8) * 1],
    )
    MEM_L1_L2_D3D4_col2 = SHIM_L2_L3_D3D4_col2.prod().join(
        obj_types=[data_d_ty,data_d_ty], 
        names=["MEM_L1_L2_D3_col1", "MEM_L1_L2_D4_col1"], 
        placement=Tile(1,1),
        offsets=[(int(outputD.numel()) / 8) * 2, (outputD.numel() / 8) * 3],
    )
    MEM_L1_L2_D5D6_col3 = SHIM_L2_L3_D5D6_col3.prod().join(
        obj_types=[data_d_ty,data_d_ty],
        names=["MEM_L1_L2_D5_col1", "MEM_L1_L2_D6_col1"], 
        placement=Tile(2,1),
        offsets=[(int(outputD.numel()) / 8) * 4, (outputD.numel() / 8) * 5],
    )
    MEM_L1_L2_D7D8_col4 = SHIM_L2_L3_D7D8_col4.prod().join(
        obj_types=[data_d_ty,data_d_ty],
        names=["MEM_L1_L2_D7_col1", "MEM_L1_L2_D8_col1"], 
        placement=Tile(3,1),
        offsets=[(int(outputD.numel()) / 8) * 6, (outputD.numel() / 8) * 7],
    )

    #Define kernels here... ------------------------------------------------\/
    element_wise_add = ExternalFunction(
        name="eltwise_add_bf16_scalar",
        source_file="../../../aie_kernels/aie2/add.cc",
        arg_types=[data_a_ty, data_b_ty, data_d_ty],
        include_dirs=["/scratch/andrewa/mlir-aie/aie_kernels/"]

    )

    relu_activation = ExternalFunction(
        name="bf16_relu",
        source_file="../../../aie_kernels/aie2/relu.cc",
        arg_types=[data_d_ty, data_d_ty],
        include_dirs=["/scratch/andrewa/mlir-aie/aie_kernels/"]
    )

    # core_fn here:
    def eltwise_add(element_wise_add, inputA, inputB, outputC):
        elementA = inputA.acquire()
        elementB = inputB.acquire()
        elementC = outputC.acquire()
        element_wise_add(elementA, elementB, elementC)
        inputA.release()
        inputB.release()
        outputC.release()

    def relu(relu_activation, inputC, outputD):
        elementC = inputC.acquire()
        elementD = outputD.acquire()
        relu_activation(elementC, elementD)
        inputC.release()
        outputD.release()

    #Workers defined here:
    Workers = []
    A1_B1_worker = Worker(core_fn=eltwise_add,fn_args=[element_wise_add, MEM_L2_L1_A1A2_col1[0].cons(),MEM_L2_L1_B1B2_col1[0].cons(),L1_L1_elwiseadd_relu_1.prod()],placement=Tile(0,5))
    A2_B2_worker = Worker(core_fn=eltwise_add,fn_args=[element_wise_add, MEM_L2_L1_A1A2_col1[1].cons(),MEM_L2_L1_B1B2_col1[1].cons(),L1_L1_elwiseadd_relu_2.prod()],placement=Tile(0,3))
    A3_B3_worker = Worker(core_fn=eltwise_add,fn_args=[element_wise_add, MEM_L2_L1_A3A4_col2[0].cons(),MEM_L2_L1_B3B4_col2[0].cons(),L1_L1_elwiseadd_relu_3.prod()],placement=Tile(1,5))
    A4_B4_worker = Worker(core_fn=eltwise_add,fn_args=[element_wise_add, MEM_L2_L1_A3A4_col2[1].cons(),MEM_L2_L1_B3B4_col2[1].cons(),L1_L1_elwiseadd_relu_4.prod()],placement=Tile(1,3))
    A5_B5_worker = Worker(core_fn=eltwise_add,fn_args=[element_wise_add, MEM_L2_L1_A5A6_col3[0].cons(),MEM_L2_L1_B5B6_col3[0].cons(),L1_L1_elwiseadd_relu_5.prod()],placement=Tile(2,5))
    A6_B6_worker = Worker(core_fn=eltwise_add,fn_args=[element_wise_add, MEM_L2_L1_A5A6_col3[1].cons(),MEM_L2_L1_B5B6_col3[1].cons(),L1_L1_elwiseadd_relu_6.prod()],placement=Tile(2,3))
    A7_B7_worker = Worker(core_fn=eltwise_add,fn_args=[element_wise_add, MEM_L2_L1_A7A8_col4[0].cons(),MEM_L2_L1_B7B8_col4[0].cons(),L1_L1_elwiseadd_relu_7.prod()],placement=Tile(3,5))
    A8_B8_worker = Worker(core_fn=eltwise_add,fn_args=[element_wise_add, MEM_L2_L1_A7A8_col4[1].cons(),MEM_L2_L1_B7B8_col4[1].cons(),L1_L1_elwiseadd_relu_8.prod()],placement=Tile(3,3))
    
    AB_workers = [A1_B1_worker, A2_B2_worker, A3_B3_worker, A4_B4_worker, A5_B5_worker, A6_B6_worker, A7_B7_worker, A8_B8_worker]
    for worker in AB_workers:
        Workers.append(worker)

    C1_worker = Worker(core_fn=relu,fn_args=[relu_activation, L1_L1_elwiseadd_relu_1.cons(), MEM_L1_L2_D1D2_col1[0].prod()],placement=Tile(0,4))
    C2_worker = Worker(core_fn=relu,fn_args=[relu_activation, L1_L1_elwiseadd_relu_2.cons(), MEM_L1_L2_D1D2_col1[1].prod()],placement=Tile(0,2))
    C3_worker = Worker(core_fn=relu,fn_args=[relu_activation, L1_L1_elwiseadd_relu_3.cons(), MEM_L1_L2_D3D4_col2[0].prod()],placement=Tile(1,4))
    C4_worker = Worker(core_fn=relu,fn_args=[relu_activation, L1_L1_elwiseadd_relu_4.cons(), MEM_L1_L2_D3D4_col2[1].prod()],placement=Tile(1,2))
    C5_worker = Worker(core_fn=relu,fn_args=[relu_activation, L1_L1_elwiseadd_relu_5.cons(), MEM_L1_L2_D5D6_col3[0].prod()],placement=Tile(2,4))
    C6_worker = Worker(core_fn=relu,fn_args=[relu_activation, L1_L1_elwiseadd_relu_6.cons(), MEM_L1_L2_D5D6_col3[1].prod()],placement=Tile(2,2))
    C7_worker = Worker(core_fn=relu,fn_args=[relu_activation, L1_L1_elwiseadd_relu_7.cons(), MEM_L1_L2_D7D8_col4[0].prod()],placement=Tile(3,4))
    C8_worker = Worker(core_fn=relu,fn_args=[relu_activation, L1_L1_elwiseadd_relu_8.cons(), MEM_L1_L2_D7D8_col4[1].prod()],placement=Tile(3,2))
    
    C_workers = [C1_worker, C2_worker, C3_worker, C4_worker, C5_worker, C6_worker, C7_worker, C8_worker]
    for worker in C_workers:
        Workers.append(worker)

    #Define runtime here:
    rt = Runtime()
    with rt.sequence(data_a_ty, data_b_ty, data_d_ty) as (A, B, D):
        rt.start(*Workers)

        rt.fill(in_fifo=SHIM_L3_L2_A1A2_col1.prod(), source=A, tap=TensorAccessPattern(tensor_dims=[float(inputA.numel()),],offset=((int(inputA.numel())/4)*0), sizes=[1024, (int(inputA.numel())/4)/1024], strides=[1, 1024],))
        rt.fill(in_fifo=SHIM_L3_L2_A3A4_col2.prod(), source=A, tap=TensorAccessPattern(tensor_dims=[flo(inputA.numel()),],offset=((int(inputA.numel())/4)*1), sizes=[1024, (int(inputA.numel())/4)/1024], strides=[1, 1024],))
        rt.fill(in_fifo=SHIM_L3_L2_A5A6_col3.prod(), source=A, tap=TensorAccessPattern(tensor_dims=[int(inputA.numel()),],offset=((int(inputA.numel())/4)*2), sizes=[1024, (int(inputA.numel())/4)/1024], strides=[1, 1024],))
        rt.fill(in_fifo=SHIM_L3_L2_A7A8_col4.prod(), source=A, tap=TensorAccessPattern(tensor_dims=[int(inputA.numel()),],offset=((int(inputA.numel())/4)*3), sizes=[1024, (int(inputA.numel())/4)/1024], strides=[1, 1024],))

        rt.fill(in_fifo=SHIM_L3_L2_B1B2_col1.prod(), source=B, tap=TensorAccessPattern(tensor_dims=[int(inputB.numel()),],offset=((int(inputB.numel())/4)*0), sizes=[1024, (int(inputB.numel())/4)/1024], strides=[1, 1024],))
        rt.fill(in_fifo=SHIM_L3_L2_B3B4_col2.prod(), source=B, tap=TensorAccessPattern(tensor_dims=[int(inputB.numel()),],offset=((int(inputB.numel())/4)*1), sizes=[1024, (int(inputB.numel())/4)/1024], strides=[1, 1024],))
        rt.fill(in_fifo=SHIM_L3_L2_B5B6_col3.prod(), source=B, tap=TensorAccessPattern(tensor_dims=[int(inputB.numel()),],offset=((int(inputB.numel())/4)*2), sizes=[1024, (int(inputB.numel())/4)/1024], strides=[1, 1024],))
        rt.fill(in_fifo=SHIM_L3_L2_B7B8_col4.prod(), source=B, tap=TensorAccessPattern(tensor_dims=[int(inputB.numel()),],offset=((int(inputB.numel())/4)*3), sizes=[1024, (int(inputB.numel())/4)/1024], strides=[1, 1024],))

        rt.drain(out_fifo=SHIM_L2_L3_D1D2_col1.cons(), dest=D, tap=TensorAccessPattern(tensor_dims=[int(outputD.numel()),],offset=((int(outputD.numel())/4)*0), sizes=[1024, (int(outputD.numel())/4)/1024], strides=[1, 1024],))
        rt.drain(out_fifo=SHIM_L2_L3_D3D4_col2.cons(), dest=D, tap=TensorAccessPattern(tensor_dims=[int(outputD.numel()),],offset=((int(outputD.numel())/4)*1), sizes=[1024, (int(outputD.numel())/4)/1024], strides=[1, 1024],))
        rt.drain(out_fifo=SHIM_L2_L3_D5D6_col3.cons(), dest=D, tap=TensorAccessPattern(tensor_dims=[int(outputD.numel()),],offset=((int(outputD.numel())/4)*2), sizes=[1024, (int(outputD.numel())/4)/1024], strides=[1, 1024],))
        rt.drain(out_fifo=SHIM_L2_L3_D7D8_col4.cons(), dest=D, tap=TensorAccessPattern(tensor_dims=[int(outputD.numel()),],offset=((int(outputD.numel())/4)*3), sizes=[1024, (int(outputD.numel())/4)/1024], strides=[1, 1024],))

    my_program = Program(iron.get_current_device(), rt)
    my_program = my_program.resolve_program(SequentialPlacer()) # No sequential placer for this program (all is explicitly placed)
    return my_program


def main():
    # Define Data here ... -------------------------------------------------\/
    datatype = bfloat16
    data_size = 8192
    inputA = iron.rand(data_size, dtype=datatype, device="npu")
    inputB = iron.arange(data_size, dtype=datatype, device="npu")
    outputD = iron.zeros(data_size, dtype=datatype, device="npu")
    base_aaa(inputA, inputB, outputD)
    print(outputD)
    
if __name__ == "__main__":
    main()

