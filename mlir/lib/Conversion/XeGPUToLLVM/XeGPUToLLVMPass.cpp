//===- XeGPUToLLVMPass.cpp - XeGPU to LLVM dialect conversion -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to convert XeGPU dialect into the LLVM IR
// dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/XeGPUToLLVM/XeGPUToLLVMPass.h"

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/XeGPU/IR/XeGPU.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTXEGPUTOLLVM
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

#define PASS_NAME "convert-xegpu-to-llvm"

namespace {
struct ConvertXeGPUToLLVMPass
    : public impl::ConvertXeGPUToLLVMBase<ConvertXeGPUToLLVMPass> {
  using Base::Base;
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp op = getOperation();
    // TODO
  }
};
} // namespace
