#include <aie_api/aie.hpp>

extern "C" {
    void eltwise_add_bf16_vector(bfloat16* inputA, bfloat16* inputB, bfloat16* output) {
        // Vector type for 16 bfloat16 elements (512-bit vector)
        using vector_t = aie::vector<bfloat16, 16>;
        
        constexpr int VEC_SIZE = 16;
        constexpr int CHUNK_SIZE = 256;  // Match your chunk_ty size
        
        // Process entire chunk in vectorized fashion
        for (int i = 0; i < CHUNK_SIZE; i += VEC_SIZE) {
            // Load vectors from both inputs
            vector_t vecA = aie::load_v<16>(inputA + i);
            vector_t vecB = aie::load_v<16>(inputB + i);
            
            // Add vectors element-wise
            vector_t vecResult = aie::add(vecA, vecB);
            
            // Store result vector
            aie::store_v(output + i, vecResult);
        }
    }
} // extern "C"