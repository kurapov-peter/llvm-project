// RUN: mlir-opt -xegpu-distribute-to-wi -split-input-file %s | FileCheck %s

func.func @test_load_nd_distribution(%src: memref<24x32xf32>) -> () {
  %1 = xegpu.create_nd_tdesc %src[0, 0] : memref<24x32xf32> ->
    !xegpu.tensor_desc<8x16xf32, #xegpu.sg_map<wi_layout = [1, 16], wi_data = [1, 1]>>
// CHECK: %[[R1:.*]] = xegpu.load_nd %[[R0]] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>, packed}> : !xegpu.tensor_desc<8x16xf32> -> vector<4x1x2xf32>
  %2 = xegpu.load_nd %1 <{packed, l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}>
       : !xegpu.tensor_desc<8x16xf32, #xegpu.sg_map<wi_layout = [1, 16], wi_data = [1, 1]>> -> vector<4x16x2xf32>
  return
}