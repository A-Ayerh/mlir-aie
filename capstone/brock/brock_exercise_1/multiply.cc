#include <aie_api/aie.hpp>

extern "C"{
    void internal_multiply(int* input, int* output, int tile_size, int multiplier){
        // Vector type is for 16 int32 elements (512-bit vector)
        using vector_t = aie::vector<int32_t, 16>;

        // Take 16 elements from input
        vector_t vector_in = aie::load_v<16>(input);

        //Multiply vector by scalar
        auto acc = aie::mul(vector_in, multiplier);

        //Convert accum type to vector_t
        vector_t vector_out = acc.to_vector<int32_t>();

        // Store resulting vector
        aie::store_v(output, vector_out);
    }

} // extern "C"