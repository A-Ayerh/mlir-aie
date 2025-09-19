//===- my_matrix_multiplication.cc ----------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>

#if defined(__chess__)
#define AIE_PREPARE_FOR_PIPELINING [[chess::prepare_for_pipelining]]
#define AIE_LOOP_MIN_ITERATION_COUNT(x) [[chess::min_loop_count(x)]]
#elif defined(__AIECC__)
#ifndef __STRINGIFY
#define __STRINGIFY(a) #a
#endif
#define AIE_LOOP_MIN_ITERATION_COUNT(x)                                        \
  _Pragma(__STRINGIFY(clang loop min_iteration_count(x)))
#define AIE_PREPARE_FOR_PIPELINING
#else
#define AIE_LOOP_MIN_ITERATION_COUNT(x)
#define AIE_PREPARE_FOR_PIPELINING
#endif

// Tile and intrinsic sizes must match Python script defaults
constexpr unsigned m = 64;
constexpr unsigned k = 64;
constexpr unsigned n = 32;
constexpr unsigned r = 8;
constexpr unsigned s = 2;
constexpr unsigned t = 8;

// Default accumulator (acc48 for int16 x int16)
using MMUL = aie::mmul<r, s, t, int16, int16>;

extern "C" {

void matmul_and_zero(const int16 *__restrict A, const int16 *__restrict B,
                     int16 *__restrict C) {
  // Zero C (int16)
  constexpr unsigned vector_size = 16;  // 256 bits / 16 bits = 16 int16 elements
  constexpr unsigned elements = m * n;
  aie::vector<int16, vector_size> zero_vec = aie::zeros<int16, vector_size>();

  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(elements / vector_size)
  for (unsigned i = 0; i < elements; i += vector_size) {
    aie::store_v(C + i, zero_vec);
  }

  // Matrix multiplication with accumulation into C
  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(m / r / 2)  // Adjusted for row loop step=2
  for (unsigned row = 0; row < m / r; row += 2) {
    AIE_LOOP_MIN_ITERATION_COUNT(n / t / 2)  // Adjusted for col loop step=2
    for (unsigned col = 0; col < n / t; col += 2) {
      // Pointers to start of two rows of A and two columns of B
      const int16 *__restrict A0_ptr =
          A + ((row + 0) * (k / s) + 0) * MMUL::size_A;
      const int16 *__restrict A1_ptr =
          A + ((row + 1) * (k / s) + 0) * MMUL::size_A;
      const int16 *__restrict B0_ptr =
          B + (0 * (n / t) + (col + 0)) * MMUL::size_B;
      const int16 *__restrict B1_ptr =
          B + (0 * (n / t) + (col + 1)) * MMUL::size_B;

      // Load current C values (int16, compatible with default accumulator)
      const aie::vector<int16, MMUL::size_C> C00_in = aie::load_v<MMUL::size_C>(
          C + ((row + 0) * (n / t) + (col + 0)) * MMUL::size_C);
      const aie::vector<int16, MMUL::size_C> C01_in = aie::load_v<MMUL::size_C>(
          C + ((row + 0) * (n / t) + (col + 1)) * MMUL::size_C);
      const aie::vector<int16, MMUL::size_C> C10_in = aie::load_v<MMUL::size_C>(
          C + ((row + 1) * (n / t) + (col + 0)) * MMUL::size_C);
      const aie::vector<int16, MMUL::size_C> C11_in = aie::load_v<MMUL::size_C>(
          C + ((row + 1) * (n / t) + (col + 1)) * MMUL::size_C);
      MMUL C00(C00_in);
      MMUL C01(C01_in);
      MMUL C10(C10_in);
      MMUL C11(C11_in);

      // Accumulate over k dimension
      for (unsigned i = 0; i < k / s; i += 1, A0_ptr += MMUL::size_A,
                    A1_ptr += MMUL::size_A, B0_ptr += (n / t) * MMUL::size_B,
                    B1_ptr += (n / t) * MMUL::size_B) {
        const aie::vector<int16, MMUL::size_A> A0 =
            aie::load_v<MMUL::size_A>(A0_ptr);
        const aie::vector<int16, MMUL::size_A> A1 =
            aie::load_v<MMUL::size_A>(A1_ptr);
        const aie::vector<int16, MMUL::size_B> B0 =
            aie::load_v<MMUL::size_B>(B0_ptr);
        const aie::vector<int16, MMUL::size_B> B1 =
            aie::load_v<MMUL::size_B>(B1_ptr);
        C00.mac(A0, B0);
        C01.mac(A0, B1);
        C10.mac(A1, B0);
        C11.mac(A1, B1);
      }
      // Store results back to C (int16)
      aie::store_v(C + ((row + 0) * (n / t) + (col + 0)) * MMUL::size_C,
                   C00.template to_vector<int16>());
      aie::store_v(C + ((row + 0) * (n / t) + (col + 1)) * MMUL::size_C,
                   C01.template to_vector<int16>());
      aie::store_v(C + ((row + 1) * (n / t) + (col + 0)) * MMUL::size_C,
                   C10.template to_vector<int16>());
      aie::store_v(C + ((row + 1) * (n / t) + (col + 1)) * MMUL::size_C,
                   C11.template to_vector<int16>());
    }
  }
}

}  // extern "C"