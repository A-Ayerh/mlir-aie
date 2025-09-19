// kernels/vec_add_scalar.cc
// Performs plain vector add using scalar compute unit: c[i] = a[i] + b[i]
// Performs vectorized vector addition
#include <stdint.h>
#include <aie_api/aie.hpp>

extern "C" {
  void vec_add_scalar(DTYPE* __restrict a,
                      DTYPE* __restrict b,
                      DTYPE* __restrict c,
                      DTYPE N) {
    for (int i = 0; i < N; ++i) {
      c[i] = a[i] + b[i];
    }
  }

  void vec_add_vectorized(DTYPE* __restrict a,
                          DTYPE* __restrict b,
                          DTYPE* __restrict c,
                          int32_t N) {
    event0();

    constexpr int VEC = LANE_SIZE;
    DTYPE* __restrict pa = a;
    DTYPE* __restrict pb = b;
    DTYPE* __restrict pc = c;

    const int F = N / VEC;
    const int T = N % VEC;

    [[chess::prepare_for_pipelining, chess::min_loop_count(MIN_LOOP_ITERATIONS)]]
    for (int i = 0; i < F; ++i) {
      aie::vector<DTYPE, VEC> va = aie::load_v<VEC>(pa); pa += VEC;
      aie::vector<DTYPE, VEC> vb = aie::load_v<VEC>(pb); pb += VEC;
      aie::vector<DTYPE, VEC> vc = aie::add(va, vb);
      aie::store_v(pc, vc); pc += VEC;
    }

    for (int t = 0; t < T; ++t) {
      pc[t] = pa[t] + pb[t];
    }

    event1();
  }
} // extern "C"