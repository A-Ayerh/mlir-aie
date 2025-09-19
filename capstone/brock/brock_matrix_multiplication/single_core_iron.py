#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc. or its affiliates
import argparse
import numpy as np
import sys
import os
import aie.iron as iron
from aie.iron import ExternalFunction, jit
from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker, str_to_dtype
from aie.iron.placers import SequentialPlacer
from aie.iron.controlflow import range_
from aie.helpers.taplib import TensorAccessPattern, TensorAccessSequence, TensorTiler2D
from aie.iron.jit import NPUKernel, IRON_CACHE_DIR
from aie.iron.jit import hash_module
from aie.dialects.aie import AIEDevice
from aie.utils.config import cxx_header_path

microkernel_mac_dim_map = {
    "npu": {
        "bf16": (4, 8, 4),
        "i8": (4, 8, 8),
        "i16": (8, 2, 8),
    },
    "npu2": {
        "bf16": {
            True: (8, 8, 8),
            False: (4, 8, 8),
        },
        "i8": (8, 8, 8),
        "i16": (4, 4, 8),
    },
}

# Helper function to map NumPy dtype names to microkernel_mac_dim_map keys
def dtype_name_to_key(dtype_name):
    dtype_map = {
        "int8": "i8",
        "int16": "i16",
        "float16": "bf16",
        "int32": "i32",
        "float32": "f32",
    }
    return dtype_map.get(dtype_name, dtype_name)

def ceildiv(a, b):
    return (a + b - 1) // b

@jit(is_placed=False)
def jit_matrix_multiplication(M, K, N, m, k, n, r, s, t, dtype_in, dtype_out, b_col_maj, emulate_bf16_mmul_with_bfp16, trace_size, input0, input1, output):
    assert M == input0.shape[0], f"Expected M ({M}) to match input0 rows ({input0.shape[0]})"
    assert K == input0.shape[1] == input1.shape[0], f"Expected K ({K}) to match input0 cols ({input0.shape[1]}) and input1 rows ({input1.shape[0]})"
    assert N == input1.shape[1], f"Expected N ({N}) to match input1 cols ({input1.shape[1]})"
    assert M % m == 0
    assert K % k == 0
    assert N % n == 0
    assert m % r == 0
    assert k % s == 0
    assert n % t == 0

    vectorized = True
    enable_tracing = True if trace_size > 0 else False

    assert np.issubdtype(dtype_in, np.integer) == np.issubdtype(dtype_out, np.integer), \
        f"Input dtype ({dtype_in}) and output dtype({dtype_out}) must either both be integral or both be float"
    assert (np.dtype(dtype_out).itemsize >= np.dtype(dtype_in).itemsize), \
        f"Output dtype ({dtype_out}) must be equal or larger to input dtype ({dtype_in})"

    M_div_m = M // m
    K_div_k = K // k
    N_div_n = N // n
    tiles = M_div_m * N_div_n

    A_ty = np.ndarray[(M, K), np.dtype[dtype_in]]
    B_ty = np.ndarray[(K, N), np.dtype[dtype_in]]
    C_ty = np.ndarray[(M, N), np.dtype[dtype_out]]
    a_ty = np.ndarray[(m, k), np.dtype[dtype_in]]
    b_ty = np.ndarray[(k, n), np.dtype[dtype_in]]
    c_ty = np.ndarray[(m, n), np.dtype[dtype_out]]

    function_type = "" if vectorized else "scalar_"
    matmul_kernel = ExternalFunction(
        "matmul_and_zero",
        source_file=os.path.join(os.path.dirname(__file__), "my_matrix_multiplication.cc"),
        arg_types=[a_ty, b_ty, c_ty],
        include_dirs=[cxx_header_path()],
    )

    inA = ObjectFifo(a_ty, name="inA")
    a_dims = None
    if vectorized:
        a_dims = [(m // r, r * k), (k // s, s), (r, k), (s, 1)]
    memA = inA.cons().forward(name="memA", dims_to_stream=a_dims)

    inB = ObjectFifo(b_ty, name="inB")
    b_dims = None
    if vectorized:
        if b_col_maj:
            b_dims = [(n // t, t * k), (k // s, s), (t, k), (s, 1)]
        else:
            b_dims = [(k // s, s * n), (n // t, t), (s, n), (t, 1)]
    memB = inB.cons().forward(name="memB", dims_to_stream=b_dims)

    memC = ObjectFifo(c_ty, name="memC")
    c_dims = None
    if vectorized:
        c_dims = [(m // r, r * n), (r, t), (n // t, r * t), (t, 1)]
    outC = memC.cons().forward(name="outC", dims_to_stream=c_dims)

    def core_fn(of_a, of_b, of_c, matmul):
        for _ in range_(tiles) if tiles > 1 else range(1):
            elem_out = of_c.acquire(1)
            for _ in range_(K_div_k) if K_div_k > 1 else range(1):
                elem_in_a = of_a.acquire(1)
                elem_in_b = of_b.acquire(1)
                matmul(elem_in_a, elem_in_b, elem_out)
                of_a.release(1)
                of_b.release(1)
            of_c.release(1)

    worker = Worker(
        core_fn, [memA.cons(), memB.cons(), memC.prod(), matmul_kernel], stack_size=0xD00
    )

    rows_per_block = 4

    A_tiles = TensorTiler2D.group_tiler(
        (M, K), (m, k), (1, K_div_k), pattern_repeat=N_div_n
    )
    if b_col_maj:
        b_tap = TensorTiler2D.group_tiler((N, K), (n, k), (N_div_n, K_div_k))[0]
    else:
        b_tap = TensorTiler2D.group_tiler(
            (K, N), (k, n), (K_div_k, N_div_n), tile_group_col_major=True
        )[0]

    C_tiles = TensorTiler2D.group_tiler((M, N), (m, n), (rows_per_block // 2, N_div_n))
    c_index = 0

    rt = Runtime()
    with rt.sequence(A_ty, B_ty, C_ty) as (A, B, C):
        rt.enable_trace(trace_size, workers=[worker])
        rt.start(worker)

        tgs = []
        for tile_row_block in range(ceildiv(M_div_m, rows_per_block)):
            for pingpong in [0, 1]:
                row_base = tile_row_block * rows_per_block + pingpong * rows_per_block // 2
                num_tile_rows = min([rows_per_block // 2, M_div_m - row_base])
                if num_tile_rows <= 0:
                    break
                tgs.append(rt.task_group())
                for tile_row in range(num_tile_rows):
                    tile_offset = (row_base + tile_row) % len(A_tiles)
                    rt.fill(inA.prod(), A, tap=A_tiles[tile_offset], task_group=tgs[-1])
                    rt.fill(inB.prod(), B, tap=b_tap, task_group=tgs[-1])
                rt.drain(outC.cons(), C, tap=C_tiles[c_index], task_group=tgs[-1], wait=True)
                c_index += 1
                if tile_row_block > 0 or (tile_row_block == 0 and pingpong > 0):
                    rt.finish_task_group(tgs[-2])
                    del tgs[-2]
        rt.finish_task_group(tgs[-1])
        del tgs[-1]

    return rt

def main():
    argparser = argparse.ArgumentParser(
        prog="AIE Matrix Multiplication MLIR Design (Single Core)",
        description="Emits MLIR code for a matrix multiplication design of the given input size",
    )
    argparser.add_argument("-M", type=int, default=256)
    argparser.add_argument("-K", type=int, default=256)
    argparser.add_argument("-N", type=int, default=256)
    argparser.add_argument("-m", type=int, default=64)
    argparser.add_argument("-k", type=int, default=64)
    argparser.add_argument("-n", type=int, default=32)
    argparser.add_argument(
        "--dtype_in", type=str, choices=["bf16", "i8", "i16"], default="i16"
    )
    argparser.add_argument(
        "--dtype_out",
        type=str,
        choices=["bf16", "i8", "i16", "f32", "i32"],
        default="i32",
    )
    argparser.add_argument("--b-col-maj", type=int, choices=[0, 1], default=0)
    argparser.add_argument("--emulate-bf16-mmul-with-bfp16", type=bool, default=False)
    argparser.add_argument("--trace_size", type=int, default=0)
    argparser.add_argument(
        "--generate-taps",
        action="store_true",
        help="Generate TensorAccessPatterns, a Python object to represent each data transfer"
        "of the input/output matrices. These objects can be used for visualization.",
    )
    args = argparser.parse_args()

    # Get device dynamically
    dev_ty = iron.get_current_device()
    dev_str = "npu2" if isinstance(dev_ty, iron.device.NPU2) else "npu"

    dtype_in = str_to_dtype(args.dtype_in)
    dtype_out = str_to_dtype(args.dtype_out)

    # Compute microkernel dimensions
    dtype_in_str = dtype_name_to_key(np.dtype(dtype_in).name)
    mac_dimensions = microkernel_mac_dim_map[dev_str][dtype_in_str]
    if dev_str == "npu2" and dtype_in_str == "bf16":
        r, s, t = mac_dimensions[args.emulate_bf16_mmul_with_bfp16]
    else:
        r, s, t = mac_dimensions

    if args.generate_taps:
        device_str = "npu"
        input0 = iron.randint(0, 256, (args.M, args.K), dtype=dtype_in, device=device_str)
        input1 = iron.randint(0, 256, (args.K, args.N), dtype=dtype_in, device=device_str)
        output = iron.zeros((args.M, args.N), dtype=dtype_out, device=device_str)
        maybe_module = my_matmul(
            dev_str,
            args.M,
            args.K,
            args.N,
            args.m,
            args.k,
            args.n,
            r,
            s,
            t,
            args.dtype_in,
            args.dtype_out,
            args.b_col_maj,
            args.emulate_bf16_mmul_with_bfp16,
            args.trace_size,
            input0,
            input1,
            output,
            args.generate_taps,
        )
        print(maybe_module)
        return

    device_str = "npu"
    input0 = iron.randint(0, 256, (args.M, args.K), dtype=dtype_in, device=device_str)
    input1 = iron.randint(0, 256, (args.K, args.N), dtype=dtype_in, device=device_str)
    output = iron.zeros((args.M, args.N), dtype=dtype_out, device=device_str)

    if np.issubdtype(dtype_in, np.integer):
        ref_vector = np.matmul(input0.numpy().astype(np.float32), input1.numpy().astype(np.float32)).astype(dtype_out)
    else:
        ref_vector = np.matmul(input0.numpy(), input1.numpy()).astype(dtype_out)

    rt = jit_matrix_multiplication(
        args.M, args.K, args.N, args.m, args.k, args.n, r, s, t,
        dtype_in, dtype_out, args.b_col_maj, args.emulate_bf16_mmul_with_bfp16,
        args.trace_size, input0, input1, output
    )

    my_program = Program(dev_ty, rt)
    module = my_program.resolve_program(SequentialPlacer())

    e = np.equal(ref_vector, output.numpy())
    errors = np.size(e) - np.count_nonzero(e)
    if not errors:
        print("\nPASS!\n")
        sys.exit(0)
    else:
        print("\nError count: ", errors)
        print("\nFAILED!\n")
        sys.exit(1)

def my_matmul(
    dev, M, K, N, m, k, n, r, s, t, dtype_in_str, dtype_out_str, b_col_maj,
    emulate_bf16_mmul_with_bfp16, trace_size, input0, input1, output, generate_taps=False
):
    assert M == input0.shape[0], f"Expected M ({M}) to match input0 rows ({input0.shape[0]})"
    assert K == input0.shape[1] == input1.shape[0], f"Expected K ({K}) to match input0 cols ({input0.shape[1]}) and input1 rows ({input1.shape[0]})"
    assert N == input1.shape[1], f"Expected N ({N}) to match input1 cols ({input1.shape[1]})"
    assert M % m == 0
    assert K % k == 0
    assert N % n == 0
    assert m % r == 0
    assert k % s == 0
    assert n % t == 0

    dev_ty = iron.get_current_device()  # Use dynamic device
    dev_str = "npu2" if isinstance(dev_ty, iron.device.NPU2) else "npu"
    vectorized = True
    enable_tracing = True if trace_size > 0 else False

    dtype_in = str_to_dtype(dtype_in_str)
    dtype_out = str_to_dtype(dtype_out_str)

    assert np.issubdtype(dtype_in, np.integer) == np.issubdtype(
        dtype_out, np.integer
    ), f"Input dtype ({dtype_in}) and output dtype ({dtype_out}) must either both be integral or both be float"
    assert (
        np.dtype(dtype_out).itemsize >= np.dtype(dtype_in).itemsize
    ), f"Output dtype ({dtype_out}) must be equal or larger to input dtype ({dtype_in})"

    M_div_m = M // m
    K_div_k = K // k
    N_div_n = N // n
    tiles = M_div_m * N_div_n

    A_taps = []
    B_taps = []
    C_taps = []

    A_ty = np.ndarray[(M * K,), np.dtype[dtype_in]]
    B_ty = np.ndarray[(K * N,), np.dtype[dtype_in]]
    C_ty = np.ndarray[(M * N,), np.dtype[dtype_out]]
    a_ty = np.ndarray[(m, k), np.dtype[dtype_in]]
    b_ty = np.ndarray[(k, n), np.dtype[dtype_in]]
    c_ty = np.ndarray[(m, n), np.dtype[dtype_out]]

    func_type = "" if vectorized else "scalar_"
    matmul_kernel = ExternalFunction(
        "matmul_and_zero",
        source_file=os.path.join(os.path.dirname(__file__), "my_matrix_multiplication.cc"),
        arg_types=[a_ty, b_ty, c_ty],
        include_dirs=[cxx_header_path()],
    )

    inA = ObjectFifo(a_ty, name="inA")
    a_dims = None
    if vectorized:
        a_dims = [(m // r, r * k), (k // s, s), (r, k), (s, 1)]
    memA = inA.cons().forward(name="memA", dims_to_stream=a_dims)

    inB = ObjectFifo(b_ty, name="inB")
    b_dims = None
    if vectorized:
        if b_col_maj:
            b_dims = [(n // t, t * k), (k // s, s), (t, k), (s, 1)]
        else:
            b_dims = [(k // s, s * n), (n // t, t), (s, n), (t, 1)]
    memB = inB.cons().forward(name="memB", dims_to_stream=b_dims)

    memC = ObjectFifo(c_ty, name="memC")
    c_dims = None
    if vectorized:
        c_dims = [(m // r, r * n), (r, t), (n // t, r * t), (t, 1)]
    outC = memC.cons().forward(name="outC", dims_to_stream=c_dims)

    def core_fn(of_a, of_b, of_c, matmul):
        for _ in range_(tiles) if tiles > 1 else range(1):
            elem_out = of_c.acquire(1)
            for _ in range_(K_div_k) if K_div_k > 1 else range(1):
                elem_in_a = of_a.acquire(1)
                elem_in_b = of_b.acquire(1)
                matmul(elem_in_a, elem_in_b, elem_out)
                of_a.release(1)
                of_b.release(1)
            of_c.release(1)

    worker = Worker(
        core_fn, [memA.cons(), memB.cons(), memC.prod(), matmul_kernel], stack_size=0xD00
    )

    rows_per_block = 4

    A_tiles = TensorTiler2D.group_tiler(
        (M, K), (m, k), (1, K_div_k), pattern_repeat=N_div_n
    )
    if b_col_maj:
        b_tap = TensorTiler2D.group_tiler((N, K), (n, k), (N_div_n, K_div_k))[0]
    else:
        b_tap = TensorTiler2D.group_tiler(
            (K, N), (k, n), (K_div_k, N_div_n), tile_group_col_major=True
        )[0]

    C_tiles = TensorTiler2D.group_tiler((M, N), (m, n), (rows_per_block // 2, N_div_n))
    c_index = 0

    rt = Runtime()
    with rt.sequence(A_ty, B_ty, C_ty) as (A, B, C):
        rt.enable_trace(trace_size, workers=[worker])
        rt.start(worker)

        tgs = []
        for tile_row_block in range(ceildiv(M_div_m, rows_per_block)):
            for pingpong in [0, 1]:
                row_base = tile_row_block * rows_per_block + pingpong * rows_per_block // 2
                num_tile_rows = min([rows_per_block // 2, M_div_m - row_base])
                if num_tile_rows <= 0:
                    break
                tgs.append(rt.task_group())
                for tile_row in range(num_tile_rows):
                    tile_offset = (row_base + tile_row) % len(A_tiles)
                    rt.fill(inA.prod(), A, tap=A_tiles[tile_offset], task_group=tgs[-1])
                    A_taps.append(A_tiles[tile_offset])
                    rt.fill(inB.prod(), B, tap=b_tap, task_group=tgs[-1])
                    B_taps.append(b_tap)
                rt.drain(outC.cons(), C, tap=C_tiles[c_index], task_group=tgs[-1], wait=True)
                C_taps.append(C_tiles[c_index])
                c_index += 1
                if tile_row_block > 0 or (tile_row_block == 0 and pingpong > 0):
                    rt.finish_task_group(tgs[-2])
                    del tgs[-2]
        rt.finish_task_group(tgs[-1])
        del tgs[-1]

    if generate_taps:
        return (
            TensorAccessSequence.from_taps(A_taps),
            TensorAccessSequence.from_taps(B_taps),
            TensorAccessSequence.from_taps(C_taps),
        )

    return rt

if __name__ == "__main__":
    main()