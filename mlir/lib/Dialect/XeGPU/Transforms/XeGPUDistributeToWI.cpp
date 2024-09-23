//===- XeGPUDistributeToWI.cpp - XeGPU ditribute SIMD to WI -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/XeGPU/Transforms/Passes.h"

#include "mlir/Dialect/XeGPU/IR/XeGPU.h"
#include "mlir/Dialect/XeGPU/Transforms/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

namespace mlir {
namespace xegpu {
#define GEN_PASS_DEF_XEGPUDISTRIBUTETOWI
#include "mlir/Dialect/XeGPU/Transforms/Passes.h.inc"
} // namespace xegpu
} // namespace mlir

#define DEBUG_TYPE "xegpu-distribute-to-wi"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")

using namespace mlir;

namespace {
bool divisible(APInt lhs, APInt rhs) { return !lhs.urem(rhs); }
struct LoadDistributor final : public OpRewritePattern<xegpu::LoadNdOp> {
  using OpRewritePattern<xegpu::LoadNdOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(xegpu::LoadNdOp loadOp,
                                PatternRewriter &rewriter) const override;
};
struct StoreDistributor final : public OpRewritePattern<xegpu::StoreNdOp> {
  using OpRewritePattern<xegpu::StoreNdOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(xegpu::StoreNdOp storeOp,
                                PatternRewriter &rewriter) const override;
};

// TODO: smth like this
// template <class T>
// llvm::FailureOr<llvm::SmallVector<int64_t>>
// get_distributed_shape(T op, PatternRewriter &rewriter,
//                       ArrayRef<int64_t> shape) {
//   auto origType = op.getTensorDescType();
//   xegpu::SGMapAttr sgMap = origType.getSGMapAttr();
//   if (!sgMap) {
//     rewriter.notifyMatchFailure(
//         op, "the source tensor descriptor lacks sg_map attribute");
//     return failure();
//   }

//   auto origShape = origType.getShape();
//   if (origShape.size() != 2) {
//     rewriter.notifyMatchFailure(op, "unsupported shape");
//     return failure();
//   }

//   llvm::SmallVector<int64_t> distributedShape;
//   auto layout = sgMap.getWiLayout();

//   for (const auto [l, o] : llvm::zip(layout, shape)) {
//     if (!divisible(APInt(64, o), APInt(64, l))) {
//       rewriter.notifyMatchFailure(
//           op,
//           "the tensor dimentions are not divisible by the distribution
//           factor");
//       return failure();
//     }
//     distributedShape.push_back(o / l);
//   }
//   return distributedShape;
// }
} // namespace

LogicalResult
LoadDistributor::matchAndRewrite(xegpu::LoadNdOp loadOp,
                                 PatternRewriter &rewriter) const {
  xegpu::TensorDescType origType = loadOp.getTensorDescType();
  xegpu::SGMapAttr sgMap = origType.getSGMapAttr();
  if (!sgMap)
    return rewriter.notifyMatchFailure(
        loadOp, "the source tensor descriptor lacks sg_map attribute");

  auto origShape = origType.getShape();
  if (origShape.size() != 2)
    return rewriter.notifyMatchFailure(loadOp, "unsupported shape");

  llvm::SmallVector<int64_t, 2> distributedResultShape;
  auto layout = sgMap.getWiLayout();
  auto outputVectorShape = loadOp.getType().getShape();

  for (const auto [l, o, v] : llvm::zip(layout, origShape, outputVectorShape)) {
    if (!divisible(APInt(64, o), APInt(64, l)))
      return rewriter.notifyMatchFailure(
          loadOp,
          "the tensor dimentions are not divisible by the distribution factor");
    if (!divisible(APInt(64, v), APInt(64, l)))
      return rewriter.notifyMatchFailure(
          loadOp, "the output vector dimentions are not divisible by the "
                  "distribution factor");
    distributedResultShape.push_back(v / l);
  }

  if (loadOp.getPacked())
    distributedResultShape.push_back(outputVectorShape.back());

  auto distributedVectorType = mlir::VectorType::get(
      distributedResultShape, loadOp.getType().getElementType(),
      loadOp.getType().getScalableDims());

  rewriter.replaceOpWithNewOp<xegpu::LoadNdOp>(
      loadOp, distributedVectorType, loadOp.getTensorDesc(),
      loadOp.getPackedAttr(), loadOp.getTransposeAttr(), loadOp.getL1HintAttr(),
      loadOp.getL2HintAttr(), loadOp.getL3HintAttr());
  return success();
}

LogicalResult
StoreDistributor::matchAndRewrite(xegpu::StoreNdOp storeOp,
                                  PatternRewriter &rewriter) const {
  auto origType = storeOp.getTensorDescType();
  xegpu::SGMapAttr sgMap = origType.getSGMapAttr();
  if (!sgMap)
    return rewriter.notifyMatchFailure(
        storeOp, "the source tensor descriptor lacks sg_map attribute");

  auto origShape = origType.getShape();
  if (origShape.size() != 2)
    return rewriter.notifyMatchFailure(storeOp, "unsupported shape");

  llvm::SmallVector<int64_t, 2> distributedShape;
  auto layout = sgMap.getWiLayout();
  auto inputVectorShape = storeOp.getValueType().getShape();

  for (const auto [l, o] : llvm::zip(layout, inputVectorShape)) {
    if (!divisible(APInt(64, o), APInt(64, l)))
      return rewriter.notifyMatchFailure(
          storeOp,
          "the tensor dimentions are not divisible by the distribution factor");
    distributedShape.push_back(o / l);
  }

  auto storeValue = storeOp.getValue();
  auto newVectorType = mlir::VectorType::get(
      distributedShape, storeOp.getValueType().getElementType(),
      storeOp.getValueType().getScalableDims());
  storeValue.setType(newVectorType);

  // ::mlir::Value value, ::mlir::Value TensorDesc,
  // /*optional*/::mlir::xegpu::CachePolicyAttr l1_hint,
  // /*optional*/::mlir::xegpu::CachePolicyAttr l2_hint,
  // /*optional*/::mlir::xegpu::CachePolicyAttr l3_hint
  rewriter.replaceOpWithNewOp<xegpu::StoreNdOp>(
      storeOp, storeValue, storeOp.getTensorDesc(), storeOp.getL1HintAttr(),
      storeOp.getL2HintAttr(), storeOp.getL3HintAttr());
  return success();
}

void xegpu::populateXeGPUDistributeToWIPatterns(RewritePatternSet &patterns) {
  patterns.add<LoadDistributor>(patterns.getContext());
}

namespace {
struct XeGPUDistributeToWIPass final
    : public xegpu::impl::XeGPUDistributeToWIBase<XeGPUDistributeToWIPass> {
  void runOnOperation() override;
};

} // namespace

void XeGPUDistributeToWIPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  xegpu::populateXeGPUDistributeToWIPatterns(patterns);
  (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
}
