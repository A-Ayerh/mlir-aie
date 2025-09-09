extern "C" {
            void internal_add_from_file(int* input, int* output, int tile_size) {
                for (int i = 0; i < tile_size; i++) {
                    output[i] = input[i] + 100;
                }
            }
} // extern "C"
