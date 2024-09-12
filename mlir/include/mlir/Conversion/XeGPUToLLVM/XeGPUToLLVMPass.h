//===- XeGPUToLLVMPass.h - Convert XeGPU ops to LLVM operations *- C++ ---*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_XEGPUTOLLVMPASS_XEGPUTOLLVMPASS_H_
#define MLIR_CONVERSION_XEGPUTOLLVMPASS_XEGPUTOLLVMPASS_H_

#include "mlir/Pass/Pass.h"
namespace mlir {

#define GEN_PASS_DECL_CONVERTXEGPUTOLLVM
#include "mlir/Conversion/Passes.h.inc"

} // namespace mlir

#endif // MLIR_CONVERSION_XEGPUTOLLVMPASS_XEGPUTOLLVMPASS_H_
