from aie.iron import Program, Runtime, Worker, ObjectFifo
from aie.iron.placers import SequentialPlacer
from aie.iron import ExternalFunction, jit
from aie.iron.dataflow import ObjectFifoLink
from aie.iron.device import Tile
import numpy as np
import aie.iron as iron
from ml_dtypes import bfloat16
from aie.helpers.taplib import TensorAccessPattern

@iron.jit(is_placed=False)
def design(inputA, inputB, outputD):
    data_a_ty = np.ndarray[(inputA.numel(),), np.dtype[bfloat16]]
    data_b_ty = np.ndarray[(inputB.numel(),), np.dtype[bfloat16]]
    data_d_ty = np.ndarray[(outputD.numel(),), np.dtype[bfloat16]]

    chunk_a = np.ndarray[(32,), np.dtype[bfloat16]]
    chunk_b = np.ndarray[(32,), np.dtype[bfloat16]]
    chunk_d = np.ndarray[(32,), np.dtype[bfloat16]]

    chunk_a_worker = np.ndarray[(16,), np.dtype[bfloat16]]
    chunk_b_worker = np.ndarray[(16,), np.dtype[bfloat16]]
    chunk_d_worker = np.ndarray[(16,), np.dtype[bfloat16]]
    # Object fifos goes here...
    fifo_shim_0_to_mem_0_A = ObjectFifo(obj_type=chunk_a, depth=2, name="fifo_shim_0_to_mem_0_A")
    fifo_shim_0_to_mem_0_B = ObjectFifo(obj_type=chunk_b, depth=2, name="fifo_shim_0_to_mem_0_B")
    fifo_shim_1_to_mem_1_A = ObjectFifo(obj_type=chunk_a, depth=2, name="fifo_shim_1_to_mem_1_A")
    fifo_shim_1_to_mem_1_B = ObjectFifo(obj_type=chunk_b, depth=2, name="fifo_shim_1_to_mem_1_B")
    fifo_shim_2_to_mem_2_A = ObjectFifo(obj_type=chunk_a, depth=2, name="fifo_shim_2_to_mem_2_A")
    fifo_shim_2_to_mem_2_B = ObjectFifo(obj_type=chunk_b, depth=2, name="fifo_shim_2_to_mem_2_B")
    fifo_shim_3_to_mem_3_A = ObjectFifo(obj_type=chunk_a, depth=2, name="fifo_shim_3_to_mem_3_A")
    fifo_shim_3_to_mem_3_B = ObjectFifo(obj_type=chunk_b, depth=2, name="fifo_shim_3_to_mem_3_B")
    fifo_compute_0_to_compute_1 = ObjectFifo(obj_type=chunk_d_worker, depth=2, name="fifo_compute_0_to_compute_1")
    fifo_compute_2_to_compute_3 = ObjectFifo(obj_type=chunk_d_worker, depth=2, name="fifo_compute_2_to_compute_3")
    fifo_compute_4_to_compute_5 = ObjectFifo(obj_type=chunk_d_worker, depth=2, name="fifo_compute_4_to_compute_5")
    fifo_compute_6_to_compute_7 = ObjectFifo(obj_type=chunk_d_worker, depth=2, name="fifo_compute_6_to_compute_7")
    fifo_compute_8_to_compute_9 = ObjectFifo(obj_type=chunk_d_worker, depth=2, name="fifo_compute_8_to_compute_9")
    fifo_compute_10_to_compute_11 = ObjectFifo(obj_type=chunk_d_worker, depth=2, name="fifo_compute_10_to_compute_11")
    fifo_compute_12_to_compute_13 = ObjectFifo(obj_type=chunk_d_worker, depth=2, name="fifo_compute_12_to_compute_13")
    fifo_compute_14_to_compute_15 = ObjectFifo(obj_type=chunk_d_worker, depth=2, name="fifo_compute_14_to_compute_15")
    fifo_mem_0_to_shim_0_D = ObjectFifo(obj_type=chunk_d, depth=2, name="fifo_mem_0_to_shim_0_D")
    fifo_mem_1_to_shim_1_D = ObjectFifo(obj_type=chunk_d, depth=2, name="fifo_mem_1_to_shim_1_D")
    fifo_mem_2_to_shim_2_D = ObjectFifo(obj_type=chunk_d, depth=2, name="fifo_mem_2_to_shim_2_D")
    fifo_mem_3_to_shim_3_D = ObjectFifo(obj_type=chunk_d, depth=2, name="fifo_mem_3_to_shim_3_D")
    # Split/Join operations:
    split_mem_0_A = fifo_shim_0_to_mem_0_A.cons().split(        obj_types=[chunk_a_worker,chunk_a_worker],        offsets=[0, 16],        names=['split_mem_0_A_0', 'split_mem_0_A_1'],        placement=Tile(0, 1)    )
    split_mem_0_B = fifo_shim_0_to_mem_0_B.cons().split(        obj_types=[chunk_b_worker,chunk_b_worker],        offsets=[0, 16],        names=['split_mem_0_B_0', 'split_mem_0_B_1'],        placement=Tile(0, 1)    )
    split_mem_1_A = fifo_shim_1_to_mem_1_A.cons().split(        obj_types=[chunk_a_worker,chunk_a_worker],        offsets=[0, 16],        names=['split_mem_1_A_0', 'split_mem_1_A_1'],        placement=Tile(1, 1)    )
    split_mem_1_B = fifo_shim_1_to_mem_1_B.cons().split(        obj_types=[chunk_b_worker,chunk_b_worker],        offsets=[0, 16],        names=['split_mem_1_B_0', 'split_mem_1_B_1'],        placement=Tile(1, 1)    )
    split_mem_2_A = fifo_shim_2_to_mem_2_A.cons().split(        obj_types=[chunk_a_worker,chunk_a_worker],        offsets=[0, 16],        names=['split_mem_2_A_0', 'split_mem_2_A_1'],        placement=Tile(2, 1)    )
    split_mem_2_B = fifo_shim_2_to_mem_2_B.cons().split(        obj_types=[chunk_b_worker,chunk_b_worker],        offsets=[0, 16],        names=['split_mem_2_B_0', 'split_mem_2_B_1'],        placement=Tile(2, 1)    )
    split_mem_3_A = fifo_shim_3_to_mem_3_A.cons().split(        obj_types=[chunk_a_worker,chunk_a_worker],        offsets=[0, 16],        names=['split_mem_3_A_0', 'split_mem_3_A_1'],        placement=Tile(3, 1)    )
    split_mem_3_B = fifo_shim_3_to_mem_3_B.cons().split(        obj_types=[chunk_b_worker,chunk_b_worker],        offsets=[0, 16],        names=['split_mem_3_B_0', 'split_mem_3_B_1'],        placement=Tile(3, 1)    )
    join_mem_0_D = fifo_mem_0_to_shim_0_D.prod().join(        obj_types=[chunk_d_worker,chunk_d_worker],        names=['join_mem_0_D_0', 'join_mem_0_D_1'],        placement=Tile(0, 1),        offsets=[0, 16],    )
    join_mem_1_D = fifo_mem_1_to_shim_1_D.prod().join(        obj_types=[chunk_d_worker,chunk_d_worker],        names=['join_mem_1_D_0', 'join_mem_1_D_1'],        placement=Tile(1, 1),        offsets=[0, 16],    )
    join_mem_2_D = fifo_mem_2_to_shim_2_D.prod().join(        obj_types=[chunk_d_worker,chunk_d_worker],        names=['join_mem_2_D_0', 'join_mem_2_D_1'],        placement=Tile(2, 1),        offsets=[0, 16],    )
    join_mem_3_D = fifo_mem_3_to_shim_3_D.prod().join(        obj_types=[chunk_d_worker,chunk_d_worker],        names=['join_mem_3_D_0', 'join_mem_3_D_1'],        placement=Tile(3, 1),        offsets=[0, 16],    )
    # External kernels:
    external_eltwiseaddbf16scalar = ExternalFunction(        name="eltwise_add_bf16_scalar",        source_file="/scratch/btsorens/mlir-aie/aie_kernels/aie2/add.cc",        arg_types=[chunk_a_worker, chunk_b_worker, chunk_d_worker],        include_dirs=['/scratch/btsorens/mlir-aie/aie_kernels']    )
    external_bf16relu = ExternalFunction(        name="bf16_relu",        source_file="/scratch/btsorens/mlir-aie/aie_kernels/aie2/relu.cc",        arg_types=[chunk_d_worker, chunk_d_worker],        include_dirs=['/scratch/btsorens/mlir-aie/aie_kernels']    )
    # Core functions:
    # Define kernels here...
    def eltwise_add(element_wise_add, inputA, inputB, outputC):
        elementA = inputA.acquire(1)
        elementB = inputB.acquire(1)
        elementC = outputC.acquire(1)
        element_wise_add(elementA, elementB, elementC)
        inputA.release(1)
        inputB.release(1)
        outputC.release(1)

    def relu(relu_activation, inputC, outputD):
        elementC = inputC.acquire(1)
        elementD = outputD.acquire(1)
        relu_activation(elementC, elementD)
        inputC.release(1)
        outputD.release(1)
    # Workers:
    Workers = []
    worker_compute_0 = Worker(core_fn=eltwise_add,fn_args=[external_eltwiseaddbf16scalar, split_mem_0_A[0].cons(), split_mem_0_B[0].cons(), fifo_compute_0_to_compute_1.prod()],placement=Tile(0,5))
    Workers.append(worker_compute_0)
    worker_compute_1 = Worker(core_fn=relu,fn_args=[external_bf16relu, fifo_compute_0_to_compute_1.cons(), join_mem_0_D[0].prod()],placement=Tile(0,4))
    Workers.append(worker_compute_1)
    worker_compute_2 = Worker(core_fn=eltwise_add,fn_args=[external_eltwiseaddbf16scalar, split_mem_0_A[1].cons(), split_mem_0_B[1].cons(), fifo_compute_2_to_compute_3.prod()],placement=Tile(0,3))
    Workers.append(worker_compute_2)
    worker_compute_3 = Worker(core_fn=relu,fn_args=[external_bf16relu, fifo_compute_2_to_compute_3.cons(), join_mem_0_D[1].prod()],placement=Tile(0,2))
    Workers.append(worker_compute_3)
    worker_compute_4 = Worker(core_fn=eltwise_add,fn_args=[external_eltwiseaddbf16scalar, split_mem_1_A[0].cons(), split_mem_1_B[0].cons(), fifo_compute_4_to_compute_5.prod()],placement=Tile(1,5))
    Workers.append(worker_compute_4)
    worker_compute_5 = Worker(core_fn=relu,fn_args=[external_bf16relu, fifo_compute_4_to_compute_5.cons(), join_mem_1_D[0].prod()],placement=Tile(1,4))
    Workers.append(worker_compute_5)
    worker_compute_6 = Worker(core_fn=eltwise_add,fn_args=[external_eltwiseaddbf16scalar, split_mem_1_A[1].cons(), split_mem_1_B[1].cons(), fifo_compute_6_to_compute_7.prod()],placement=Tile(1,3))
    Workers.append(worker_compute_6)
    worker_compute_7 = Worker(core_fn=relu,fn_args=[external_bf16relu, fifo_compute_6_to_compute_7.cons(), join_mem_1_D[1].prod()],placement=Tile(1,2))
    Workers.append(worker_compute_7)
    worker_compute_8 = Worker(core_fn=eltwise_add,fn_args=[external_eltwiseaddbf16scalar, split_mem_2_A[0].cons(), split_mem_2_B[0].cons(), fifo_compute_8_to_compute_9.prod()],placement=Tile(2,5))
    Workers.append(worker_compute_8)
    worker_compute_9 = Worker(core_fn=relu,fn_args=[external_bf16relu, fifo_compute_8_to_compute_9.cons(), join_mem_2_D[0].prod()],placement=Tile(2,4))
    Workers.append(worker_compute_9)
    worker_compute_10 = Worker(core_fn=eltwise_add,fn_args=[external_eltwiseaddbf16scalar, split_mem_2_A[1].cons(), split_mem_2_B[1].cons(), fifo_compute_10_to_compute_11.prod()],placement=Tile(2,3))
    Workers.append(worker_compute_10)
    worker_compute_11 = Worker(core_fn=relu,fn_args=[external_bf16relu, fifo_compute_10_to_compute_11.cons(), join_mem_2_D[1].prod()],placement=Tile(2,2))
    Workers.append(worker_compute_11)
    worker_compute_12 = Worker(core_fn=eltwise_add,fn_args=[external_eltwiseaddbf16scalar, split_mem_3_A[0].cons(), split_mem_3_B[0].cons(), fifo_compute_12_to_compute_13.prod()],placement=Tile(3,5))
    Workers.append(worker_compute_12)
    worker_compute_13 = Worker(core_fn=relu,fn_args=[external_bf16relu, fifo_compute_12_to_compute_13.cons(), join_mem_3_D[0].prod()],placement=Tile(3,4))
    Workers.append(worker_compute_13)
    worker_compute_14 = Worker(core_fn=eltwise_add,fn_args=[external_eltwiseaddbf16scalar, split_mem_3_A[1].cons(), split_mem_3_B[1].cons(), fifo_compute_14_to_compute_15.prod()],placement=Tile(3,3))
    Workers.append(worker_compute_14)
    worker_compute_15 = Worker(core_fn=relu,fn_args=[external_bf16relu, fifo_compute_14_to_compute_15.cons(), join_mem_3_D[1].prod()],placement=Tile(3,2))
    Workers.append(worker_compute_15)
    # Runtime configuration:
    # Define runtime here:
    rt = Runtime()
    with rt.sequence(chunk_a, chunk_b, chunk_d) as (A, B, D):
        rt.start(*Workers)
        rt.fill(placement=Tile(0,0), in_fifo=fifo_shim_0_to_mem_0_A.prod(), source=A, tap=TensorAccessPattern(tensor_dims=[(inputA.numel()),],offset=(((inputA.numel())//4)*0), sizes=[((inputA.numel())//4)//((inputA.numel())//8), ((inputA.numel())//8)], strides=[((inputA.numel())//8), 1],))
        rt.fill(placement=Tile(0,0), in_fifo=fifo_shim_0_to_mem_0_B.prod(), source=B, tap=TensorAccessPattern(tensor_dims=[(inputB.numel()),],offset=(((inputB.numel())//4)*0), sizes=[((inputB.numel())//4)//((inputB.numel())//8), ((inputB.numel())//8)], strides=[((inputB.numel())//8), 1],))
        rt.drain(placement=Tile(0,0), out_fifo=fifo_mem_0_to_shim_0_D.cons(), dest=D, wait=True, tap=TensorAccessPattern(tensor_dims=[(outputD.numel()),],offset=(((outputD.numel())//4)*0), sizes=[((outputD.numel())//4)//((outputD.numel())//8), ((outputD.numel())//8)], strides=[((outputD.numel())//8), 1],))
        rt.fill(placement=Tile(1,0), in_fifo=fifo_shim_1_to_mem_1_A.prod(), source=A, tap=TensorAccessPattern(tensor_dims=[(inputA.numel()),],offset=(((inputA.numel())//4)*1), sizes=[((inputA.numel())//4)//((inputA.numel())//8), ((inputA.numel())//8)], strides=[((inputA.numel())//8), 1],))
        rt.fill(placement=Tile(1,0), in_fifo=fifo_shim_1_to_mem_1_B.prod(), source=B, tap=TensorAccessPattern(tensor_dims=[(inputB.numel()),],offset=(((inputB.numel())//4)*1), sizes=[((inputB.numel())//4)//((inputB.numel())//8), ((inputB.numel())//8)], strides=[((inputB.numel())//8), 1],))
        rt.drain(placement=Tile(1,0), out_fifo=fifo_mem_1_to_shim_1_D.cons(), dest=D, wait=True, tap=TensorAccessPattern(tensor_dims=[(outputD.numel()),],offset=(((outputD.numel())//4)*1), sizes=[((outputD.numel())//4)//((outputD.numel())//8), ((outputD.numel())//8)], strides=[((outputD.numel())//8), 1],))
        rt.fill(placement=Tile(2,0), in_fifo=fifo_shim_2_to_mem_2_A.prod(), source=A, tap=TensorAccessPattern(tensor_dims=[(inputA.numel()),],offset=(((inputA.numel())//4)*2), sizes=[((inputA.numel())//4)//((inputA.numel())//8), ((inputA.numel())//8)], strides=[((inputA.numel())//8), 1],))
        rt.fill(placement=Tile(2,0), in_fifo=fifo_shim_2_to_mem_2_B.prod(), source=B, tap=TensorAccessPattern(tensor_dims=[(inputB.numel()),],offset=(((inputB.numel())//4)*2), sizes=[((inputB.numel())//4)//((inputB.numel())//8), ((inputB.numel())//8)], strides=[((inputB.numel())//8), 1],))
        rt.drain(placement=Tile(2,0), out_fifo=fifo_mem_2_to_shim_2_D.cons(), dest=D, wait=True, tap=TensorAccessPattern(tensor_dims=[(outputD.numel()),],offset=(((outputD.numel())//4)*2), sizes=[((outputD.numel())//4)//((outputD.numel())//8), ((outputD.numel())//8)], strides=[((outputD.numel())//8), 1],))
        rt.fill(placement=Tile(3,0), in_fifo=fifo_shim_3_to_mem_3_A.prod(), source=A, tap=TensorAccessPattern(tensor_dims=[(inputA.numel()),],offset=(((inputA.numel())//4)*3), sizes=[((inputA.numel())//4)//((inputA.numel())//8), ((inputA.numel())//8)], strides=[((inputA.numel())//8), 1],))
        rt.fill(placement=Tile(3,0), in_fifo=fifo_shim_3_to_mem_3_B.prod(), source=B, tap=TensorAccessPattern(tensor_dims=[(inputB.numel()),],offset=(((inputB.numel())//4)*3), sizes=[((inputB.numel())//4)//((inputB.numel())//8), ((inputB.numel())//8)], strides=[((inputB.numel())//8), 1],))
        rt.drain(placement=Tile(3,0), out_fifo=fifo_mem_3_to_shim_3_D.cons(), dest=D, wait=True, tap=TensorAccessPattern(tensor_dims=[(outputD.numel()),],offset=(((outputD.numel())//4)*3), sizes=[((outputD.numel())//4)//((outputD.numel())//8), ((outputD.numel())//8)], strides=[((outputD.numel())//8), 1],))
    my_program = Program(iron.get_current_device(), rt)
    my_program = my_program.resolve_program(SequentialPlacer())
    return my_program

def main():
    datatype = bfloat16
    data_size = 128
    inputA = iron.arange(data_size, dtype=datatype, device="npu")
    inputB = iron.arange(data_size, dtype=datatype, device="npu")
    outputD = iron.zeros(data_size, dtype=datatype, device="npu")
    design(inputA, inputB, outputD)
    print(outputD)
    # Validation check
    inputA_data = np.arange(data_size, dtype=np.float32)  # Fallback since np.asarray may fail for IRON tensors
    inputB_data = np.arange(data_size, dtype=np.float32)  # Fallback since np.asarray may fail for IRON tensors
    expected = np.maximum(0, inputA_data + inputB_data)
    actual = np.asarray(outputD, dtype=np.float32)
    print('Element-by-element comparison:')
    for i in range(data_size):
        print(f'Expected = ReLU({inputA_data[i]:.1f} + {inputB_data[i]:.1f}) = {expected[i]:.1f} : Received = {actual[i]:.1f}')
    tolerance = 1e-3  # Tolerance for bfloat16 comparison
    mismatches = np.where(~np.isclose(actual, expected, rtol=tolerance))[0]
    if len(mismatches) == 0:
        print('Validation passed: Output matches expected for all 128 elements')
    else:
        print(f'Validation failed: {len(mismatches)} mismatches found')
        for idx in mismatches[:5]:  # Print up to 5 mismatches
            print(f'Index {idx}: actual={actual[idx]:.1f}, expected={expected[idx]:.1f}')
        if len(mismatches) > 5:
            print(f'... and {len(mismatches) - 5} more mismatches')

if __name__ == "__main__":
    main()
