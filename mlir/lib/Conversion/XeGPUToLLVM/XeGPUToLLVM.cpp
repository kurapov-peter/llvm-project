//===- XeGPUToLLVM.cpp - XeGPU to LLVM patterns ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements patterns to convert XeGPU dialect to LLVM dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/XeGPUToLLVM/XeGPUToLLVM.h"

#define DEBUG_TYPE "xegpu-to-llvm-pattern"

namespace mlir {
void populateXeGPUToLLVMConversionPatterns(LLVMTypeConverter &converter,
                                           RewritePatternSet &patterns) {}
} // namespace mlir
