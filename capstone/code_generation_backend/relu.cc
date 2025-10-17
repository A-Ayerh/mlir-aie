#include <aie_api/aie.hpp>

extern "C" {
    void bf16_relu(bfloat16* input, bfloat16* output) {
        // Vector type for 16 bfloat16 elements (512-bit vector)
        using vector_t = aie::vector<bfloat16, 16>;
        
        constexpr int VEC_SIZE = 16;
        constexpr int CHUNK_SIZE = 256;  // Match your chunk_ty size
        
        // Create zero vector by broadcasting scalar 0 to 16 elements
        vector_t zero_vec = aie::broadcast<bfloat16, 16>(0);
        
        // Process entire chunk in vectorized fashion
        for (int i = 0; i < CHUNK_SIZE; i += VEC_SIZE) {
            // Load input vector
            vector_t vecIn = aie::load_v<16>(input + i);
            
            // Apply ReLU: max(input, 0)
            vector_t vecResult = aie::max(vecIn, zero_vec);
            
            // Store result vector
            aie::store_v(output + i, vecResult);
        }
    }
} // extern "C"