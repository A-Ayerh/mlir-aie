#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>
#include <aie_api/aie.hpp>

void internal_add_from_file(int* input_a, int* input_b, int* output_c, int tile_size) {
    for (int i = 0; i < tile_size; i++) {
        output_c[i] =  input_a[i] + input_b[i];
    }
}

template <typename T_in, typename T_out, int rowA, int colA, int colB>
void matmul_scalar_cascade_put_only(T_in *a, T_in *b) {
  event0();
  for (int row = 0; row < rowA; row++) {
    for (int col = 0; col < colB; col++) {
      T_out running_sum = 0;
      for (int i = 0; i < colA; i++) {
        running_sum += a[row * colA + i] * b[i * colB + col];
      }
      v16int32 v16 = undef_v16int32();
      v16 = upd_elem(v16, 0, (int)running_sum);
      put_mcd(v16);
    }
  }
  event1();
}

template <typename T_in, typename T_out, int rowA, int colA, int colB>
void matmul_scalar_cascade_get_only(T_in *a, T_in *b, T_out *c) {
  event0();
  for (int row = 0; row < rowA; row++) {
    for (int col = 0; col < colB; col++) {
      T_out running_sum = 0;
      for (int i = 0; i < colA; i++) {
        running_sum += a[row * colA + i] * b[i * colB + col];
      }
      v16int32 v16 = get_scd_v16int32();
      running_sum += ext_elem(v16, 0U);
      c[row * colB + col] += running_sum;
    }
  }
  event1();
}

template <typename T_in, typename T_out, int rowA, int colA, int colB>
void matmul_scalar_cascade_put_get(T_in *a, T_in *b) {
  event0();
  for (int row = 0; row < rowA; row++) {
    for (int col = 0; col < colB; col++) {
      T_out running_sum = 0;
      for (int i = 0; i < colA; i++) {
        running_sum += a[row * colA + i] * b[i * colB + col];
      }
      v16int32 v16 = get_scd_v16int32();
      running_sum += ext_elem(v16, 0U);
      v16 = upd_elem(v16, 0, (int)running_sum);
      put_mcd(v16);
    }
  }
  event1();
}


extern "C" {
    #define matmul_scalar_cascade_get_only_c_func(                                 \
    ctype_in, mlir_type_in, ctype_out, mlir_type_out, r, s, t)                 \
  void matmul_scalar_cascade_get_only_##mlir_type_in##_##mlir_type_out(        \
      ctype_in *a_in, ctype_in *b_in, ctype_out *c_out) {                      \
    matmul_scalar_cascade_get_only<ctype_in, ctype_out, DIM_M, DIM_K, DIM_N>(  \
        a_in, b_in, c_out);                                                    \
  }

#define matmul_scalar_cascade_put_only_c_func(                                 \
    ctype_in, mlir_type_in, ctype_out, mlir_type_out, r, s, t)                 \
  void matmul_scalar_cascade_put_only_##mlir_type_in##_##mlir_type_out(        \
      ctype_in *a_in, ctype_in *b_in, ctype_out *c_out) {                      \
    matmul_scalar_cascade_put_only<ctype_in, ctype_out, DIM_M, DIM_K, DIM_N>(  \
        a_in, b_in, c_out);                                                    \
  }

#define matmul_scalar_cascade_put_get_c_func(                                  \
    ctype_in, mlir_type_in, ctype_out, mlir_type_out, r, s, t)                 \
  void matmul_scalar_cascade_put_get_##mlir_type_in##_##mlir_type_out(         \
      ctype_in *a_in, ctype_in *b_in, ctype_out *c_out) {                      \
    matmul_scalar_cascade_put_get<ctype_in, ctype_out, DIM_M, DIM_K, DIM_N>(   \
        a_in, b_in, c_out);                                                    \
  }
}