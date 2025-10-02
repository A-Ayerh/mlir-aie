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
    MEM_L2_L1_A1A2_col1 = SHIM_L3_L2_A1A2_col1.cons().split(offsets=[(data_a_ty.size()/8)*1, (data_a_ty.size()/8)*2,], obj_type=data_a_ty, depth=2, name="MEM_L2_L1_A1A2_col1", placement=Tile(0,1))
    MEM_L2_L1_B1B2_col1 = SHIM_L3_L2_B1B2_col1.cons().split(offsets=[(data_b_ty.size()/8)*1, (data_b_ty.size()/8)*2,], obj_type=data_b_ty, depth=2, name="MEM_L2_L1_B1B2_col1", placement=Tile(0,1))

    MEM_L2_L1_A3A4_col2 = SHIM_L3_L2_A3A4_col2.cons().split(offsets=[(data_a_ty.size()/8)*3, (data_a_ty.size()/8)*4,], obj_type=data_a_ty, depth=2, name="MEM_L2_L1_A3A4_col2", placement=Tile(1,1))
    MEM_L2_L1_B3B4_col2 = SHIM_L3_L2_B3B4_col2.cons().split(offsets=[(data_b_ty.size()/8)*3, (data_b_ty.size()/8)*4,], obj_type=data_b_ty, depth=2, name="MEM_L2_L1_B3B4_col2", placement=Tile(1,1))

    MEM_L2_L1_A5A6_col3 = SHIM_L3_L2_A5A6_col3.cons().split(offsets=[(data_a_ty.size()/8)*5, (data_a_ty.size()/8)*6,], obj_type=data_a_ty, depth=2, name="MEM_L2_L1_A5A6_col3", placement=Tile(2,1))
    MEM_L2_L1_B5B6_col3 = SHIM_L3_L2_B5B6_col3.cons().split(offsets=[(data_b_ty.size()/8)*5, (data_b_ty.size()/8)*6,], obj_type=data_b_ty, depth=2, name="MEM_L2_L1_B5B6_col3", placement=Tile(2,1))

    MEM_L2_L1_A7A8_col4 = SHIM_L3_L2_A7A8_col4.cons().split(offsets=[(data_a_ty.size()/8)*7, (data_a_ty.size()/8)*8,], obj_type=data_a_ty, depth=2, name="MEM_L2_L1_A7A8_col4", placement=Tile(3,1))
    MEM_L2_L1_B7B8_col4 = SHIM_L3_L2_B7B8_col4.cons().split(offsets=[(data_b_ty.size()/8)*7, (data_b_ty.size()/8)*8,], obj_type=data_b_ty, depth=2, name="MEM_L2_L1_B7B8_col4", placement=Tile(3,1))


    # L1_L1 Object Fifos:
    L1_L1_elwiseadd_relu = ObjectFifo(obj_type=data_d_ty, depth=2, name="L1_L1_elwiseadd_relu")

    # L2_L3 Object Fifos:
    SHIM_L2_L3_D1D2_col1 = ObjectFifo(obj_type=data_d_ty, depth=2, name="SHIM_L2_L3_D1D2_col1")
    SHIM_L2_L3_D3D4_col2 = ObjectFifo(obj_type=data_d_ty, depth=2, name="SHIM_L2_L3_D3D4_col2")
    SHIM_L2_L3_D5D6_col3 = ObjectFifo(obj_type=data_d_ty, depth=2, name="SHIM_L2_L3_D1D2_col1")
    SHIM_L2_L3_D7D8_col4 = ObjectFifo(obj_type=data_d_ty, depth=2, name="SHIM_L2_L3_D3D4_col2")

    # L1_L2 Object Fifos:
    MEM_L1_L2_D1D2_col1 = SHIM_L2_L3_D1D2_col1.prod().join(obj_type=data_d_ty, depth=2, name="MEM_L1_L2_D1D2_col1", placement=Tile(0,1))
    MEM_L1_L2_D3D4_col2 = SHIM_L2_L3_D3D4_col2.prod().join(obj_type=data_d_ty, depth=2, name="MEM_L1_L2_D3D4_col2", placement=Tile(1,1))
    MEM_L1_L2_D5D6_col3 = SHIM_L2_L3_D5D6_col3.prod().join(obj_type=data_d_ty, depth=2, name="MEM_L1_L2_D5D6_col3", placement=Tile(2,1))
    MEM_L1_L2_D7D8_col4 = SHIM_L2_L3_D7D8_col4.prod().join(obj_type=data_d_ty, depth=2, name="MEM_L1_L2_D7D8_col4", placement=Tile(3,1))

    # core_fn here:
    def eltwise_add(inputA, inputB, outputC):
        elementA = inputA.acquire()
        elementB = inputB.acquire()
        elementC = outputC.acquire()
        for i in elementA:
            elementC[i] = elementA[i] + elementB[i]
        inputA.release()
        inputB.release()
        outputC.release()

    def relu(inputC, outputD):
        elementC = inputC.acquire()
        elementD = outputD.acquire()
        for i in elementC:
            elementD[i] = max(0, elementC)
        inputC.release()
        outputD.release()

    #Workers defined here:
    Workers = []
    A1_B1_worker = Worker(core_fn=eltwise_add,fn_args=[MEM_L2_L1_A1A2_col1,MEM_L2_L1_B1B2_col1,L1_L1_elwiseadd_relu],placement=Tile(0,5))
    A2_B2_worker = Worker(core_fn=eltwise_add,fn_args=[MEM_L2_L1_A1A2_col1,MEM_L2_L1_B1B2_col1,L1_L1_elwiseadd_relu],placement=Tile(0,4))
    A3_B3_worker = Worker(core_fn=eltwise_add,fn_args=[MEM_L2_L1_A3A4_col2,MEM_L2_L1_B3B4_col2,L1_L1_elwiseadd_relu],placement=Tile(1,5))
    A4_B4_worker = Worker(core_fn=eltwise_add,fn_args=[MEM_L2_L1_A3A4_col2,MEM_L2_L1_B3B4_col2,L1_L1_elwiseadd_relu],placement=Tile(1,4))
    A5_B5_worker = Worker(core_fn=eltwise_add,fn_args=[MEM_L2_L1_A5A6_col3,MEM_L2_L1_B5B6_col3,L1_L1_elwiseadd_relu],placement=Tile(2,5))
    A6_B6_worker = Worker(core_fn=eltwise_add,fn_args=[MEM_L2_L1_A5A6_col3,MEM_L2_L1_B5B6_col3,L1_L1_elwiseadd_relu],placement=Tile(2,4))
    A7_B7_worker = Worker(core_fn=eltwise_add,fn_args=[MEM_L2_L1_A7A8_col4,MEM_L2_L1_B7B8_col4,L1_L1_elwiseadd_relu],placement=Tile(3,5))
    A8_B8_worker = Worker(core_fn=eltwise_add,fn_args=[MEM_L2_L1_A7A8_col4,MEM_L2_L1_B7B8_col4,L1_L1_elwiseadd_relu],placement=Tile(3,4))

    Workers.extend(A1_B1_worker, A2_B2_worker, A3_B3_worker, A4_B4_worker, A5_B5_worker, A6_B6_worker, A7_B7_worker, A8_B8_worker)

    C1_worker = Worker(core_fn=relu,fn_args=[L1_L1_elwiseadd_relu, MEM_L1_L2_D1D2_col1],placement=Tile(0,3))
    C2_worker = Worker(core_fn=relu,fn_args=[L1_L1_elwiseadd_relu, MEM_L1_L2_D1D2_col1],placement=Tile(0,2))
    C3_worker = Worker(core_fn=relu,fn_args=[L1_L1_elwiseadd_relu, MEM_L1_L2_D3D4_col2],placement=Tile(1,3))
    C4_worker = Worker(core_fn=relu,fn_args=[L1_L1_elwiseadd_relu, MEM_L1_L2_D3D4_col2],placement=Tile(1,2))
    C5_worker = Worker(core_fn=relu,fn_args=[L1_L1_elwiseadd_relu, MEM_L1_L2_D5D6_col3],placement=Tile(2,3))
    C6_worker = Worker(core_fn=relu,fn_args=[L1_L1_elwiseadd_relu, MEM_L1_L2_D5D6_col3],placement=Tile(2,2))
    C7_worker = Worker(core_fn=relu,fn_args=[L1_L1_elwiseadd_relu, MEM_L1_L2_D7D8_col4],placement=Tile(3,3))
    C8_worker = Worker(core_fn=relu,fn_args=[L1_L1_elwiseadd_relu, MEM_L1_L2_D7D8_col4],placement=Tile(3,2))

    Workers.extend(C1_worker, C2_worker, C3_worker, C4_worker, C5_worker, C6_worker, C7_worker, C8_worker)

    #Define runtime here:
    rt = Runtime()
    with rt.sequence(data_a_ty, data_b_ty, data_d_ty) as (A, B, D):
        rt.start(*Workers)

        rt.fill(in_fifo=SHIM_L3_L2_A1A2_col1.prod(), tap=TensorAccessPattern(tensor_dims=[1,1024],offset=(data_a_ty.size/4)*1, sizes=[1024, (data_a_ty.size/4)/1024], strides=[1, 1024],))
        rt.fill(in_fifo=SHIM_L3_L2_A3A4_col2.prod(), tap=TensorAccessPattern(tensor_dims=[1,1024],offset=(data_a_ty.size/4)*2, sizes=[1024, (data_a_ty.size/4)/1024], strides=[1, 1024],))
        rt.fill(in_fifo=SHIM_L3_L2_A5A6_col3.prod(), tap=TensorAccessPattern(tensor_dims=[1,1024],offset=(data_a_ty.size/4)*3, sizes=[1024, (data_a_ty.size/4)/1024], strides=[1, 1024],))
        rt.fill(in_fifo=SHIM_L3_L2_A7A8_col4.prod(), tap=TensorAccessPattern(tensor_dims=[1,1024],offset=(data_a_ty.size/4)*4, sizes=[1024, (data_a_ty.size/4)/1024], strides=[1, 1024],))

        rt.drain(in_fifo=SHIM_L2_L3_D1D2_col1.cons(), tap=TensorAccessPattern(tensor_dims=[1,1024],offset=(data_a_ty.size/4)*1, sizes=[1024, (data_a_ty.size/4)/1024], strides=[1, 1024],))
        rt.drain(in_fifo=SHIM_L2_L3_D3D4_col2.cons(), tap=TensorAccessPattern(tensor_dims=[1,1024],offset=(data_a_ty.size/4)*2, sizes=[1024, (data_a_ty.size/4)/1024], strides=[1, 1024],))
        rt.drain(in_fifo=SHIM_L2_L3_D5D6_col3.cons(), tap=TensorAccessPattern(tensor_dims=[1,1024],offset=(data_a_ty.size/4)*3, sizes=[1024, (data_a_ty.size/4)/1024], strides=[1, 1024],))
        rt.drain(in_fifo=SHIM_L2_L3_D7D8_col4.cons(), tap=TensorAccessPattern(tensor_dims=[1,1024],offset=(data_a_ty.size/4)*4, sizes=[1024, (data_a_ty.size/4)/1024], strides=[1, 1024],))

    my_program = Program(iron.get_current_device(), rt)
    my_program = my_program.resolve_program() # No sequential placer for this program (all is explicitly placed)
    return my_program


def main():
    # Define Data here ... -------------------------------------------------\/
    datatype = bfloat16
    data_size = 8192
    inputA = iron.rand(data_size, dtype=datatype, device="npu")
    inputB = iron.arange(data_size, dtype=datatype, device="npu", step=-1)
    outputD = iron.zeros(data_size, dtype=datatype, device="npu")
    base_aaa(inputA, inputB, outputD)
    print(outputD)
    
if __name__ == "__main__":
    main()

