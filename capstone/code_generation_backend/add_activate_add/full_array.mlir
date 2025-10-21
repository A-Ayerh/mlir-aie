module {
  aie.device(npu1) {
    %tile_0_5 = aie.tile(0, 5)
    %tile_0_3 = aie.tile(0, 3)
    %tile_1_5 = aie.tile(1, 5)
    %tile_1_3 = aie.tile(1, 3)
    %tile_2_5 = aie.tile(2, 5)
    %tile_2_3 = aie.tile(2, 3)
    %tile_3_5 = aie.tile(3, 5)
    %tile_3_3 = aie.tile(3, 3)
    %tile_0_4 = aie.tile(0, 4)
    %tile_0_2 = aie.tile(0, 2)
    %tile_1_4 = aie.tile(1, 4)
    %tile_1_2 = aie.tile(1, 2)
    %tile_2_4 = aie.tile(2, 4)
    %tile_2_2 = aie.tile(2, 2)
    %tile_3_4 = aie.tile(3, 4)
    %tile_3_2 = aie.tile(3, 2)
    %mem_tile_2_1 = aie.tile(2, 1)
    %mem_tile_0_1 = aie.tile(0, 1)
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %shim_noc_tile_1_0 = aie.tile(1, 0)
    %mem_tile_1_1 = aie.tile(1, 1)
    %mem_tile_3_1 = aie.tile(3, 1)
    %shim_noc_tile_2_0 = aie.tile(2, 0)
    %shim_noc_tile_3_0 = aie.tile(3, 0)
    aie.objectfifo @L1_L1_elwiseadd_relu_6(%tile_2_3, {%tile_2_2}, 2 : i32) : !aie.objectfifo<memref<1024xbf16>> 
    aie.objectfifo @MEM_L1_L2_D6_col2(%tile_2_2, {%mem_tile_2_1}, 2 : i32) : !aie.objectfifo<memref<1024xbf16>> 
    aie.objectfifo @MEM_L1_L2_D5_col2(%tile_2_4, {%mem_tile_2_1}, 2 : i32) : !aie.objectfifo<memref<1024xbf16>> 
    aie.objectfifo @SHIM_L2_L3_D5D6_col2(%mem_tile_2_1, {%shim_noc_tile_2_0}, 2 : i32) : !aie.objectfifo<memref<2048xbf16>> 
    aie.objectfifo.link [@MEM_L1_L2_D5_col2, @MEM_L1_L2_D6_col2] -> [@SHIM_L2_L3_D5D6_col2]([4096, 5120] [])
    aie.objectfifo @MEM_L2_L1_A5_col2(%mem_tile_2_1, {%tile_2_5}, 2 : i32) : !aie.objectfifo<memref<1024xbf16>> 
    aie.objectfifo @SHIM_L3_L2_A5A6_col2(%shim_noc_tile_2_0, {%mem_tile_2_1}, 2 : i32) : !aie.objectfifo<memref<2048xbf16>> 
    aie.objectfifo @MEM_L2_L1_A6_col2(%mem_tile_2_1, {%tile_2_3}, 2 : i32) : !aie.objectfifo<memref<1024xbf16>> 
    aie.objectfifo.link [@SHIM_L3_L2_A5A6_col2] -> [@MEM_L2_L1_A5_col2, @MEM_L2_L1_A6_col2]([] [4096, 5120])
    aie.objectfifo @SHIM_L2_L3_D1D2_col0(%mem_tile_0_1, {%shim_noc_tile_0_0}, 2 : i32) : !aie.objectfifo<memref<2048xbf16>> 
    aie.objectfifo @MEM_L1_L2_D1_col0(%tile_0_4, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<1024xbf16>> 
    aie.objectfifo @MEM_L1_L2_D2_col0(%tile_0_2, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<1024xbf16>> 
    aie.objectfifo.link [@MEM_L1_L2_D1_col0, @MEM_L1_L2_D2_col0] -> [@SHIM_L2_L3_D1D2_col0]([0, 1024] [])
    aie.objectfifo @MEM_L2_L1_B5_col2(%mem_tile_2_1, {%tile_2_5}, 2 : i32) : !aie.objectfifo<memref<1024xbf16>> 
    aie.objectfifo @SHIM_L3_L2_B5B6_col2(%shim_noc_tile_2_0, {%mem_tile_2_1}, 2 : i32) : !aie.objectfifo<memref<2048xbf16>> 
    aie.objectfifo @MEM_L2_L1_B6_col2(%mem_tile_2_1, {%tile_2_3}, 2 : i32) : !aie.objectfifo<memref<1024xbf16>> 
    aie.objectfifo.link [@SHIM_L3_L2_B5B6_col2] -> [@MEM_L2_L1_B5_col2, @MEM_L2_L1_B6_col2]([] [4096, 5120])
    aie.objectfifo @L1_L1_elwiseadd_relu_5(%tile_2_5, {%tile_2_4}, 2 : i32) : !aie.objectfifo<memref<1024xbf16>> 
    aie.objectfifo @SHIM_L3_L2_A3A4_col1(%shim_noc_tile_1_0, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<2048xbf16>> 
    aie.objectfifo @MEM_L2_L1_A3_col1(%mem_tile_1_1, {%tile_1_5}, 2 : i32) : !aie.objectfifo<memref<1024xbf16>> 
    aie.objectfifo @MEM_L2_L1_A4_col1(%mem_tile_1_1, {%tile_1_3}, 2 : i32) : !aie.objectfifo<memref<1024xbf16>> 
    aie.objectfifo.link [@SHIM_L3_L2_A3A4_col1] -> [@MEM_L2_L1_A3_col1, @MEM_L2_L1_A4_col1]([] [2048, 3072])
    aie.objectfifo @L1_L1_elwiseadd_relu_7(%tile_3_5, {%tile_3_4}, 2 : i32) : !aie.objectfifo<memref<1024xbf16>> 
    aie.objectfifo @MEM_L1_L2_D7_col3(%tile_3_4, {%mem_tile_3_1}, 2 : i32) : !aie.objectfifo<memref<1024xbf16>> 
    aie.objectfifo @MEM_L1_L2_D8_col3(%tile_3_2, {%mem_tile_3_1}, 2 : i32) : !aie.objectfifo<memref<1024xbf16>> 
    aie.objectfifo @SHIM_L2_L3_D7D8_col3(%mem_tile_3_1, {%shim_noc_tile_3_0}, 2 : i32) : !aie.objectfifo<memref<2048xbf16>> 
    aie.objectfifo.link [@MEM_L1_L2_D7_col3, @MEM_L1_L2_D8_col3] -> [@SHIM_L2_L3_D7D8_col3]([6144, 7168] [])
    aie.objectfifo @SHIM_L2_L3_D3D4_col1(%mem_tile_1_1, {%shim_noc_tile_1_0}, 2 : i32) : !aie.objectfifo<memref<2048xbf16>> 
    aie.objectfifo @MEM_L1_L2_D3_col1(%tile_1_4, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<1024xbf16>> 
    aie.objectfifo @MEM_L1_L2_D4_col1(%tile_1_2, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<1024xbf16>> 
    aie.objectfifo.link [@MEM_L1_L2_D3_col1, @MEM_L1_L2_D4_col1] -> [@SHIM_L2_L3_D3D4_col1]([2048, 3072] [])
    aie.objectfifo @L1_L1_elwiseadd_relu_8(%tile_3_3, {%tile_3_2}, 2 : i32) : !aie.objectfifo<memref<1024xbf16>> 
    aie.objectfifo @MEM_L2_L1_A7_col3(%mem_tile_3_1, {%tile_3_5}, 2 : i32) : !aie.objectfifo<memref<1024xbf16>> 
    aie.objectfifo @SHIM_L3_L2_A7A8_col3(%shim_noc_tile_3_0, {%mem_tile_3_1}, 2 : i32) : !aie.objectfifo<memref<2048xbf16>> 
    aie.objectfifo @MEM_L2_L1_A8_col3(%mem_tile_3_1, {%tile_3_3}, 2 : i32) : !aie.objectfifo<memref<1024xbf16>> 
    aie.objectfifo.link [@SHIM_L3_L2_A7A8_col3] -> [@MEM_L2_L1_A7_col3, @MEM_L2_L1_A8_col3]([] [6144, 7168])
    aie.objectfifo @MEM_L2_L1_B7_col3(%mem_tile_3_1, {%tile_3_5}, 2 : i32) : !aie.objectfifo<memref<1024xbf16>> 
    aie.objectfifo @SHIM_L3_L2_B7B8_col3(%shim_noc_tile_3_0, {%mem_tile_3_1}, 2 : i32) : !aie.objectfifo<memref<2048xbf16>> 
    aie.objectfifo @MEM_L2_L1_B8_col3(%mem_tile_3_1, {%tile_3_3}, 2 : i32) : !aie.objectfifo<memref<1024xbf16>> 
    aie.objectfifo.link [@SHIM_L3_L2_B7B8_col3] -> [@MEM_L2_L1_B7_col3, @MEM_L2_L1_B8_col3]([] [3072, 3584])
    aie.objectfifo @SHIM_L3_L2_B1B2_col0(%shim_noc_tile_0_0, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<2048xbf16>> 
    aie.objectfifo @MEM_L2_L1_B1_col0(%mem_tile_0_1, {%tile_0_5}, 2 : i32) : !aie.objectfifo<memref<1024xbf16>> 
    aie.objectfifo @MEM_L2_L1_B2_col0(%mem_tile_0_1, {%tile_0_3}, 2 : i32) : !aie.objectfifo<memref<1024xbf16>> 
    aie.objectfifo.link [@SHIM_L3_L2_B1B2_col0] -> [@MEM_L2_L1_B1_col0, @MEM_L2_L1_B2_col0]([] [0, 1024])
    aie.objectfifo @L1_L1_elwiseadd_relu_1(%tile_0_5, {%tile_0_4}, 2 : i32) : !aie.objectfifo<memref<1024xbf16>> 
    aie.objectfifo @MEM_L2_L1_A1_col0(%mem_tile_0_1, {%tile_0_5}, 2 : i32) : !aie.objectfifo<memref<1024xbf16>> 
    aie.objectfifo @SHIM_L3_L2_A1A2_col0(%shim_noc_tile_0_0, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<2048xbf16>> 
    aie.objectfifo @MEM_L2_L1_A2_col0(%mem_tile_0_1, {%tile_0_3}, 2 : i32) : !aie.objectfifo<memref<1024xbf16>> 
    aie.objectfifo.link [@SHIM_L3_L2_A1A2_col0] -> [@MEM_L2_L1_A1_col0, @MEM_L2_L1_A2_col0]([] [0, 1024])
    aie.objectfifo @L1_L1_elwiseadd_relu_2(%tile_0_3, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<1024xbf16>> 
    aie.objectfifo @SHIM_L3_L2_B3B4_col1(%shim_noc_tile_1_0, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<2048xbf16>> 
    aie.objectfifo @MEM_L2_L1_B3_col1(%mem_tile_1_1, {%tile_1_5}, 2 : i32) : !aie.objectfifo<memref<1024xbf16>> 
    aie.objectfifo @MEM_L2_L1_B4_col1(%mem_tile_1_1, {%tile_1_3}, 2 : i32) : !aie.objectfifo<memref<1024xbf16>> 
    aie.objectfifo.link [@SHIM_L3_L2_B3B4_col1] -> [@MEM_L2_L1_B3_col1, @MEM_L2_L1_B4_col1]([] [2048, 3072])
    aie.objectfifo @L1_L1_elwiseadd_relu_3(%tile_1_5, {%tile_1_4}, 2 : i32) : !aie.objectfifo<memref<1024xbf16>> 
    aie.objectfifo @L1_L1_elwiseadd_relu_4(%tile_1_3, {%tile_1_2}, 2 : i32) : !aie.objectfifo<memref<1024xbf16>> 
    func.func private @eltwise_add_bf16_scalar(memref<1024xbf16>, memref<1024xbf16>, memref<1024xbf16>)
    func.func private @bf16_relu(memref<1024xbf16>, memref<1024xbf16>)
    %core_0_5 = aie.core(%tile_0_5) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @MEM_L2_L1_A1_col0(Consume, 1) : !aie.objectfifosubview<memref<1024xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<1024xbf16>> -> memref<1024xbf16>
        %2 = aie.objectfifo.acquire @MEM_L2_L1_B1_col0(Consume, 1) : !aie.objectfifosubview<memref<1024xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<1024xbf16>> -> memref<1024xbf16>
        %4 = aie.objectfifo.acquire @L1_L1_elwiseadd_relu_1(Produce, 1) : !aie.objectfifosubview<memref<1024xbf16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<1024xbf16>> -> memref<1024xbf16>
        func.call @eltwise_add_bf16_scalar(%1, %3, %5) : (memref<1024xbf16>, memref<1024xbf16>, memref<1024xbf16>) -> ()
        aie.objectfifo.release @MEM_L2_L1_A1_col0(Consume, 1)
        aie.objectfifo.release @MEM_L2_L1_B1_col0(Consume, 1)
        aie.objectfifo.release @L1_L1_elwiseadd_relu_1(Produce, 1)
      }
      aie.end
    } {link_with = "eltwise_add_bf16_scalar.o"}
    %core_0_3 = aie.core(%tile_0_3) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @MEM_L2_L1_A2_col0(Consume, 1) : !aie.objectfifosubview<memref<1024xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<1024xbf16>> -> memref<1024xbf16>
        %2 = aie.objectfifo.acquire @MEM_L2_L1_B2_col0(Consume, 1) : !aie.objectfifosubview<memref<1024xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<1024xbf16>> -> memref<1024xbf16>
        %4 = aie.objectfifo.acquire @L1_L1_elwiseadd_relu_2(Produce, 1) : !aie.objectfifosubview<memref<1024xbf16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<1024xbf16>> -> memref<1024xbf16>
        func.call @eltwise_add_bf16_scalar(%1, %3, %5) : (memref<1024xbf16>, memref<1024xbf16>, memref<1024xbf16>) -> ()
        aie.objectfifo.release @MEM_L2_L1_A2_col0(Consume, 1)
        aie.objectfifo.release @MEM_L2_L1_B2_col0(Consume, 1)
        aie.objectfifo.release @L1_L1_elwiseadd_relu_2(Produce, 1)
      }
      aie.end
    } {link_with = "eltwise_add_bf16_scalar.o"}
    %core_1_5 = aie.core(%tile_1_5) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @MEM_L2_L1_A3_col1(Consume, 1) : !aie.objectfifosubview<memref<1024xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<1024xbf16>> -> memref<1024xbf16>
        %2 = aie.objectfifo.acquire @MEM_L2_L1_B3_col1(Consume, 1) : !aie.objectfifosubview<memref<1024xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<1024xbf16>> -> memref<1024xbf16>
        %4 = aie.objectfifo.acquire @L1_L1_elwiseadd_relu_3(Produce, 1) : !aie.objectfifosubview<memref<1024xbf16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<1024xbf16>> -> memref<1024xbf16>
        func.call @eltwise_add_bf16_scalar(%1, %3, %5) : (memref<1024xbf16>, memref<1024xbf16>, memref<1024xbf16>) -> ()
        aie.objectfifo.release @MEM_L2_L1_A3_col1(Consume, 1)
        aie.objectfifo.release @MEM_L2_L1_B3_col1(Consume, 1)
        aie.objectfifo.release @L1_L1_elwiseadd_relu_3(Produce, 1)
      }
      aie.end
    } {link_with = "eltwise_add_bf16_scalar.o"}
    %core_1_3 = aie.core(%tile_1_3) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @MEM_L2_L1_A4_col1(Consume, 1) : !aie.objectfifosubview<memref<1024xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<1024xbf16>> -> memref<1024xbf16>
        %2 = aie.objectfifo.acquire @MEM_L2_L1_B4_col1(Consume, 1) : !aie.objectfifosubview<memref<1024xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<1024xbf16>> -> memref<1024xbf16>
        %4 = aie.objectfifo.acquire @L1_L1_elwiseadd_relu_4(Produce, 1) : !aie.objectfifosubview<memref<1024xbf16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<1024xbf16>> -> memref<1024xbf16>
        func.call @eltwise_add_bf16_scalar(%1, %3, %5) : (memref<1024xbf16>, memref<1024xbf16>, memref<1024xbf16>) -> ()
        aie.objectfifo.release @MEM_L2_L1_A4_col1(Consume, 1)
        aie.objectfifo.release @MEM_L2_L1_B4_col1(Consume, 1)
        aie.objectfifo.release @L1_L1_elwiseadd_relu_4(Produce, 1)
      }
      aie.end
    } {link_with = "eltwise_add_bf16_scalar.o"}
    %core_2_5 = aie.core(%tile_2_5) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @MEM_L2_L1_A5_col2(Consume, 1) : !aie.objectfifosubview<memref<1024xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<1024xbf16>> -> memref<1024xbf16>
        %2 = aie.objectfifo.acquire @MEM_L2_L1_B5_col2(Consume, 1) : !aie.objectfifosubview<memref<1024xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<1024xbf16>> -> memref<1024xbf16>
        %4 = aie.objectfifo.acquire @L1_L1_elwiseadd_relu_5(Produce, 1) : !aie.objectfifosubview<memref<1024xbf16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<1024xbf16>> -> memref<1024xbf16>
        func.call @eltwise_add_bf16_scalar(%1, %3, %5) : (memref<1024xbf16>, memref<1024xbf16>, memref<1024xbf16>) -> ()
        aie.objectfifo.release @MEM_L2_L1_A5_col2(Consume, 1)
        aie.objectfifo.release @MEM_L2_L1_B5_col2(Consume, 1)
        aie.objectfifo.release @L1_L1_elwiseadd_relu_5(Produce, 1)
      }
      aie.end
    } {link_with = "eltwise_add_bf16_scalar.o"}
    %core_2_3 = aie.core(%tile_2_3) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @MEM_L2_L1_A6_col2(Consume, 1) : !aie.objectfifosubview<memref<1024xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<1024xbf16>> -> memref<1024xbf16>
        %2 = aie.objectfifo.acquire @MEM_L2_L1_B6_col2(Consume, 1) : !aie.objectfifosubview<memref<1024xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<1024xbf16>> -> memref<1024xbf16>
        %4 = aie.objectfifo.acquire @L1_L1_elwiseadd_relu_6(Produce, 1) : !aie.objectfifosubview<memref<1024xbf16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<1024xbf16>> -> memref<1024xbf16>
        func.call @eltwise_add_bf16_scalar(%1, %3, %5) : (memref<1024xbf16>, memref<1024xbf16>, memref<1024xbf16>) -> ()
        aie.objectfifo.release @MEM_L2_L1_A6_col2(Consume, 1)
        aie.objectfifo.release @MEM_L2_L1_B6_col2(Consume, 1)
        aie.objectfifo.release @L1_L1_elwiseadd_relu_6(Produce, 1)
      }
      aie.end
    } {link_with = "eltwise_add_bf16_scalar.o"}
    %core_3_5 = aie.core(%tile_3_5) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @MEM_L2_L1_A7_col3(Consume, 1) : !aie.objectfifosubview<memref<1024xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<1024xbf16>> -> memref<1024xbf16>
        %2 = aie.objectfifo.acquire @MEM_L2_L1_B7_col3(Consume, 1) : !aie.objectfifosubview<memref<1024xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<1024xbf16>> -> memref<1024xbf16>
        %4 = aie.objectfifo.acquire @L1_L1_elwiseadd_relu_7(Produce, 1) : !aie.objectfifosubview<memref<1024xbf16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<1024xbf16>> -> memref<1024xbf16>
        func.call @eltwise_add_bf16_scalar(%1, %3, %5) : (memref<1024xbf16>, memref<1024xbf16>, memref<1024xbf16>) -> ()
        aie.objectfifo.release @MEM_L2_L1_A7_col3(Consume, 1)
        aie.objectfifo.release @MEM_L2_L1_B7_col3(Consume, 1)
        aie.objectfifo.release @L1_L1_elwiseadd_relu_7(Produce, 1)
      }
      aie.end
    } {link_with = "eltwise_add_bf16_scalar.o"}
    %core_3_3 = aie.core(%tile_3_3) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @MEM_L2_L1_A8_col3(Consume, 1) : !aie.objectfifosubview<memref<1024xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<1024xbf16>> -> memref<1024xbf16>
        %2 = aie.objectfifo.acquire @MEM_L2_L1_B8_col3(Consume, 1) : !aie.objectfifosubview<memref<1024xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<1024xbf16>> -> memref<1024xbf16>
        %4 = aie.objectfifo.acquire @L1_L1_elwiseadd_relu_8(Produce, 1) : !aie.objectfifosubview<memref<1024xbf16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<1024xbf16>> -> memref<1024xbf16>
        func.call @eltwise_add_bf16_scalar(%1, %3, %5) : (memref<1024xbf16>, memref<1024xbf16>, memref<1024xbf16>) -> ()
        aie.objectfifo.release @MEM_L2_L1_A8_col3(Consume, 1)
        aie.objectfifo.release @MEM_L2_L1_B8_col3(Consume, 1)
        aie.objectfifo.release @L1_L1_elwiseadd_relu_8(Produce, 1)
      }
      aie.end
    } {link_with = "eltwise_add_bf16_scalar.o"}
    %core_0_4 = aie.core(%tile_0_4) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @L1_L1_elwiseadd_relu_1(Consume, 1) : !aie.objectfifosubview<memref<1024xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<1024xbf16>> -> memref<1024xbf16>
        %2 = aie.objectfifo.acquire @MEM_L1_L2_D1_col0(Produce, 1) : !aie.objectfifosubview<memref<1024xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<1024xbf16>> -> memref<1024xbf16>
        func.call @bf16_relu(%1, %3) : (memref<1024xbf16>, memref<1024xbf16>) -> ()
        aie.objectfifo.release @L1_L1_elwiseadd_relu_1(Consume, 1)
        aie.objectfifo.release @MEM_L1_L2_D1_col0(Produce, 1)
      }
      aie.end
    } {link_with = "bf16_relu.o"}
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @L1_L1_elwiseadd_relu_2(Consume, 1) : !aie.objectfifosubview<memref<1024xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<1024xbf16>> -> memref<1024xbf16>
        %2 = aie.objectfifo.acquire @MEM_L1_L2_D2_col0(Produce, 1) : !aie.objectfifosubview<memref<1024xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<1024xbf16>> -> memref<1024xbf16>
        func.call @bf16_relu(%1, %3) : (memref<1024xbf16>, memref<1024xbf16>) -> ()
        aie.objectfifo.release @L1_L1_elwiseadd_relu_2(Consume, 1)
        aie.objectfifo.release @MEM_L1_L2_D2_col0(Produce, 1)
      }
      aie.end
    } {link_with = "bf16_relu.o"}
    %core_1_4 = aie.core(%tile_1_4) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @L1_L1_elwiseadd_relu_3(Consume, 1) : !aie.objectfifosubview<memref<1024xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<1024xbf16>> -> memref<1024xbf16>
        %2 = aie.objectfifo.acquire @MEM_L1_L2_D3_col1(Produce, 1) : !aie.objectfifosubview<memref<1024xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<1024xbf16>> -> memref<1024xbf16>
        func.call @bf16_relu(%1, %3) : (memref<1024xbf16>, memref<1024xbf16>) -> ()
        aie.objectfifo.release @L1_L1_elwiseadd_relu_3(Consume, 1)
        aie.objectfifo.release @MEM_L1_L2_D3_col1(Produce, 1)
      }
      aie.end
    } {link_with = "bf16_relu.o"}
    %core_1_2 = aie.core(%tile_1_2) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @L1_L1_elwiseadd_relu_4(Consume, 1) : !aie.objectfifosubview<memref<1024xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<1024xbf16>> -> memref<1024xbf16>
        %2 = aie.objectfifo.acquire @MEM_L1_L2_D4_col1(Produce, 1) : !aie.objectfifosubview<memref<1024xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<1024xbf16>> -> memref<1024xbf16>
        func.call @bf16_relu(%1, %3) : (memref<1024xbf16>, memref<1024xbf16>) -> ()
        aie.objectfifo.release @L1_L1_elwiseadd_relu_4(Consume, 1)
        aie.objectfifo.release @MEM_L1_L2_D4_col1(Produce, 1)
      }
      aie.end
    } {link_with = "bf16_relu.o"}
    %core_2_4 = aie.core(%tile_2_4) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @L1_L1_elwiseadd_relu_5(Consume, 1) : !aie.objectfifosubview<memref<1024xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<1024xbf16>> -> memref<1024xbf16>
        %2 = aie.objectfifo.acquire @MEM_L1_L2_D5_col2(Produce, 1) : !aie.objectfifosubview<memref<1024xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<1024xbf16>> -> memref<1024xbf16>
        func.call @bf16_relu(%1, %3) : (memref<1024xbf16>, memref<1024xbf16>) -> ()
        aie.objectfifo.release @L1_L1_elwiseadd_relu_5(Consume, 1)
        aie.objectfifo.release @MEM_L1_L2_D5_col2(Produce, 1)
      }
      aie.end
    } {link_with = "bf16_relu.o"}
    %core_2_2 = aie.core(%tile_2_2) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @L1_L1_elwiseadd_relu_6(Consume, 1) : !aie.objectfifosubview<memref<1024xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<1024xbf16>> -> memref<1024xbf16>
        %2 = aie.objectfifo.acquire @MEM_L1_L2_D6_col2(Produce, 1) : !aie.objectfifosubview<memref<1024xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<1024xbf16>> -> memref<1024xbf16>
        func.call @bf16_relu(%1, %3) : (memref<1024xbf16>, memref<1024xbf16>) -> ()
        aie.objectfifo.release @L1_L1_elwiseadd_relu_6(Consume, 1)
        aie.objectfifo.release @MEM_L1_L2_D6_col2(Produce, 1)
      }
      aie.end
    } {link_with = "bf16_relu.o"}
    %core_3_4 = aie.core(%tile_3_4) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @L1_L1_elwiseadd_relu_7(Consume, 1) : !aie.objectfifosubview<memref<1024xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<1024xbf16>> -> memref<1024xbf16>
        %2 = aie.objectfifo.acquire @MEM_L1_L2_D7_col3(Produce, 1) : !aie.objectfifosubview<memref<1024xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<1024xbf16>> -> memref<1024xbf16>
        func.call @bf16_relu(%1, %3) : (memref<1024xbf16>, memref<1024xbf16>) -> ()
        aie.objectfifo.release @L1_L1_elwiseadd_relu_7(Consume, 1)
        aie.objectfifo.release @MEM_L1_L2_D7_col3(Produce, 1)
      }
      aie.end
    } {link_with = "bf16_relu.o"}
    %core_3_2 = aie.core(%tile_3_2) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @L1_L1_elwiseadd_relu_8(Consume, 1) : !aie.objectfifosubview<memref<1024xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<1024xbf16>> -> memref<1024xbf16>
        %2 = aie.objectfifo.acquire @MEM_L1_L2_D8_col3(Produce, 1) : !aie.objectfifosubview<memref<1024xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<1024xbf16>> -> memref<1024xbf16>
        func.call @bf16_relu(%1, %3) : (memref<1024xbf16>, memref<1024xbf16>) -> ()
        aie.objectfifo.release @L1_L1_elwiseadd_relu_8(Consume, 1)
        aie.objectfifo.release @MEM_L1_L2_D8_col3(Produce, 1)
      }
      aie.end
    } {link_with = "bf16_relu.o"}
    aiex.runtime_sequence @sequence(%arg0: memref<2048xbf16>, %arg1: memref<2048xbf16>, %arg2: memref<2048xbf16>) {
      %0 = aiex.dma_configure_task_for @SHIM_L3_L2_A1A2_col0 {
        aie.dma_bd(%arg0 : memref<2048xbf16>, 0, 2048, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 2048, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%0)
      %1 = aiex.dma_configure_task_for @SHIM_L3_L2_A3A4_col1 {
        aie.dma_bd(%arg0 : memref<2048xbf16>, 0, 2048, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 2048, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%1)
      %2 = aiex.dma_configure_task_for @SHIM_L3_L2_A5A6_col2 {
        aie.dma_bd(%arg0 : memref<2048xbf16>, 0, 2048, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 2048, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%2)
      %3 = aiex.dma_configure_task_for @SHIM_L3_L2_A7A8_col3 {
        aie.dma_bd(%arg0 : memref<2048xbf16>, 0, 2048, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 2048, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%3)
      %4 = aiex.dma_configure_task_for @SHIM_L3_L2_B1B2_col0 {
        aie.dma_bd(%arg1 : memref<2048xbf16>, 0, 2048, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 2048, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%4)
      %5 = aiex.dma_configure_task_for @SHIM_L3_L2_B3B4_col1 {
        aie.dma_bd(%arg1 : memref<2048xbf16>, 0, 2048, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 2048, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%5)
      %6 = aiex.dma_configure_task_for @SHIM_L3_L2_B5B6_col2 {
        aie.dma_bd(%arg1 : memref<2048xbf16>, 0, 2048, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 2048, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%6)
      %7 = aiex.dma_configure_task_for @SHIM_L3_L2_B7B8_col3 {
        aie.dma_bd(%arg1 : memref<2048xbf16>, 0, 2048, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 2048, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%7)
      %8 = aiex.dma_configure_task_for @SHIM_L2_L3_D1D2_col0 {
        aie.dma_bd(%arg2 : memref<2048xbf16>, 0, 2048, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 2048, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%8)
      aiex.dma_await_task(%8)
      aiex.dma_free_task(%0)
      aiex.dma_free_task(%1)
      aiex.dma_free_task(%2)
      aiex.dma_free_task(%3)
      aiex.dma_free_task(%4)
      aiex.dma_free_task(%5)
      aiex.dma_free_task(%6)
      aiex.dma_free_task(%7)
      %9 = aiex.dma_configure_task_for @SHIM_L2_L3_D3D4_col1 {
        aie.dma_bd(%arg2 : memref<2048xbf16>, 0, 2048, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 2048, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%9)
      aiex.dma_await_task(%9)
      %10 = aiex.dma_configure_task_for @SHIM_L2_L3_D5D6_col2 {
        aie.dma_bd(%arg2 : memref<2048xbf16>, 0, 2048, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 2048, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%10)
      aiex.dma_await_task(%10)
      %11 = aiex.dma_configure_task_for @SHIM_L2_L3_D7D8_col3 {
        aie.dma_bd(%arg2 : memref<2048xbf16>, 0, 2048, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 2048, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%11)
      aiex.dma_await_task(%11)
    }
  }
}
