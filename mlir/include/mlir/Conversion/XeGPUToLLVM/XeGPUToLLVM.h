//===- XEGPUToLLVM.h - XeGPU to LLVM Patterns -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Provides patterns to convert XeGPU dialect to LLVM dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_XEGPUTOLLVM_XEGPUTOLLVM_H
#define MLIR_CONVERSION_XEGPUTOLLVM_XEGPUTOLLVM_H

namespace mlir {

class LLVMTypeConverter;
class RewritePatternSet;

void populateXeGPUToLLVMConversionPatterns(LLVMTypeConverter &converter,
                                           RewritePatternSet &patterns);
} // namespace mlir

#endif // MLIR_CONVERSION_XEGPUTOLLVM_XEGPUTOLLVM_H
