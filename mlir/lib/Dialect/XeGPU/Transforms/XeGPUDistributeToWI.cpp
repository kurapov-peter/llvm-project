//===- XeGPUDistributeToWI.cpp - XeGPU ditribute SIMD to WI -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/XeGPU/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/XeGPU/IR/XeGPU.h"
#include "mlir/Dialect/XeGPU/Transforms/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

#include <deque>

#define DEBUG_TYPE "xegpu-distribute-to-wi"

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

namespace {
struct XeGPUDistributeToWIPass final
    : public xegpu::impl::XeGPUDistributeToWIBase<XeGPUDistributeToWIPass> {
  void runOnOperation() override;
};

bool isXeGPUDescProducer(Operation *op) {
  return isa<xegpu::CreateDescOp>(op) || isa<xegpu::CreateNdDescOp>(op);
}

struct FuncDistributor final : public OpRewritePattern<func::FuncOp> {
  FuncDistributor(MLIRContext *ctx, const unsigned subgroup_size = 16)
      : OpRewritePattern(ctx), subgroupSize(subgroup_size) {}

  LogicalResult matchAndRewrite(func::FuncOp funcOp,
                                PatternRewriter &rewriter) const override;

private:
  unsigned subgroupSize;
};
} // namespace

LogicalResult
FuncDistributor::matchAndRewrite(func::FuncOp funcOp,
                                 PatternRewriter &rewriter) const {
  // llvm::errs() << "matched function: " << &funcOp << "\n";
  // llvm::errs().flush();
  // llvm::errs() << "matched function: " << funcOp.getName() << "\n";
  // llvm::errs() << "before replacing:\n" << funcOp << "\n";

  // this breaks
  // auto clonedFuncOp = rewriter.cloneWithoutRegions(funcOp);
  // std::string clonedFuncOpName = funcOp.getName().str();
  llvm::errs() << "Matched a function at" << &funcOp << "\n";
  std::string clonedFuncOpName = "a";
  auto clonedFuncOp = rewriter.create<func::FuncOp>(
      funcOp->getLoc(), clonedFuncOpName, funcOp.getFunctionType());
  SymbolTable::setSymbolVisibility(clonedFuncOp,
                                   SymbolTable::getSymbolVisibility(funcOp));
  // for (const auto &namedAttr : funcOp->getAttrs()) {
  //   clonedFuncOp->setAttr(namedAttr.getName(), namedAttr.getValue());
  // }
  assert(clonedFuncOp.getBlocks().size() == 0);
  Block *entry = clonedFuncOp.addEntryBlock();
  rewriter.setInsertionPointToStart(entry);
  Location loc = clonedFuncOp.getLoc();
  auto c16 = rewriter.create<arith::ConstantIndexOp>(loc, subgroupSize);
  auto tid = rewriter.create<gpu::ThreadIdOp>(loc, rewriter.getIndexType(),
                                              gpu::Dimension::x);
  auto ret = rewriter.create<func::ReturnOp>(loc);
  rewriter.replaceOp(funcOp, clonedFuncOp);
  llvm::errs() << "after replacing:\n" << clonedFuncOp << "\n";
}

void xegpu::populateXeGPUDistributeToWIPatterns(RewritePatternSet &patterns) {
  // patterns.add<LoadDistributor>(patterns.getContext());
  patterns.add<FuncDistributor>(patterns.getContext());
}

// FunctionOpInterface funcOp = getOperation();
//   IRRewriter builder(funcOp);

//   // we don't have a thread id yet, so create it.
//   // warp_execute_on_lane0 consumes it's result, so creating it before moving
//   // the function body into the vector region.
//   const int64_t subgroupSize = 16; // FIXME
//   // TODO: move the block instead
//   // IRMapping mapping;
//   // funcOp.getFunctionBody().cloneInto(&warpOp.getWarpRegion(), mapping);
//   // llvm::errs() << "number of blocks in warp region: "
//   //              << warpOp.getBodyRegion().getBlocks().size() << "\n";
//   auto clonedFuncOp = funcOp.clone();
//   clonedFuncOp.eraseBody();
//   Location loc = clonedFuncOp->getLoc();
//   Block *entry = clonedFuncOp.addEntryBlock();
//   builder.setInsertionPointToStart(entry);
//   auto c16 = builder.create<arith::ConstantIndexOp>(loc, subgroupSize);
//   auto tid = builder.create<gpu::ThreadIdOp>(loc, builder.getIndexType(),
//                                              gpu::Dimension::x);
//   auto warpOp = builder.create<vector::WarpExecuteOnLane0Op>(
//       loc, TypeRange(), tid.getResult(), subgroupSize);
//   SmallVector<Type> argsTypes(warpOp.getBody()->getArgumentTypes());
//   unsigned numArgs = warpOp.getBody()->getNumArguments();
//   auto ret = builder.create<func::ReturnOp>(loc);

//   llvm::errs() << "deduced argument types: " << argsTypes << "\n";

//   // Restore the block and its arguments for the warp region
//   // builder.mergeBlocks(&funcOp.getFunctionBody().getBlocks().front(),
//   //                     warpOp.getBody());
//   // auto argsTypes = warpOp.getBody()->getArgumentTypes();
//   // warpOp.getBodyRegion().getBlocks().pop_front();
//   // assert(warpOp.getBody() && "getBody() returned null");
//   // unsigned numArgs = warpOp.getBody()->getNumArguments();
//   // warpOp.getBody()->eraseArguments(0, numArgs);
//   // warpOp.getBody()->addArguments(argsTypes,
//   //                                SmallVector<Location>(argsTypes.size(),
//   //                                loc));
//   llvm::errs() << "before cloning: " << clonedFuncOp << "\n";
//   llvm::errs() << "before cloning: " << warpOp << "\n";

//   warpOp.getWarpRegion().takeBody(funcOp.getFunctionBody());
//   Block *body = &funcOp.getFunctionBody().emplaceBlock();
//   llvm::errs() << "newly created block has " << body->getArguments().size()
//                << " arguments\n";
//   body->eraseArguments(0, body->getArguments().size());
//   body->addArguments(argsTypes,
//                      SmallVector<Location>(argsTypes.size(),
//                      warpOp.getLoc()));

//   llvm::errs() << "after cloning: " << clonedFuncOp << "\n";
//   llvm::errs() << "after cloning: " << warpOp << "\n";

//   warpOp.getBodyRegion().getBlocks().back().back().erase();
//   //
//   builder.setInsertionPointToEnd(&warpOp.getWarpRegion().getBlocks().back());
//   // builder.create<vector::YieldOp>(loc);
//   // builder.setInsertionPointToEnd(body);
//   // builder.create<vector::YieldOp>(loc);

//   // builder.replaceOpWithNewOp<vector::YieldOp>(&body->back());

//   // llvm::errs() << "after replacing terminator: " << clonedFuncOp << "\n";
//   // llvm::errs() << "after replacing terminator: " << warpOp << "\n";

//   // llvm::errs() << "after restoring the vector block: " << warpOp << "\n";

//   // Restore the function body
//   // funcOp.eraseBody();
//   // Block *entry = funcOp.addEntryBlock();
//   // llvm::errs() << "created a new entry block: " << *entry << "\n";

//   // builder.setInsertionPointToStart(entry);
//   // auto c16 = builder.create<arith::ConstantIndexOp>(loc, subgroupSize);

//   // tid->moveAfter(c16);
//   // warpOp->moveAfter(tid);
//   // tid->moveBefore(&entry, entry.end());
//   // warpOp->moveBefore(&entry, entry.end());

//   llvm::errs() << "---------------------" << "\n";
//   llvm::errs() << "IR before hoisting: " << clonedFuncOp << "\n";
//   // Hoist all the non-distributable ops out of the vector distribution
//   region

//   builder.replaceOp(funcOp, clonedFuncOp);

void XeGPUDistributeToWIPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  xegpu::populateXeGPUDistributeToWIPatterns(patterns);
  auto config = GreedyRewriteConfig();
  config.maxIterations = 2;
  (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                     config);
  llvm::errs() << "Finished applying the pattern\n";
  llvm::errs() << getOperation() << "\n";
  // todo: another option is walking:
  // PatternRewriter rewriter(&getContext());

  // // 1. Get all the producers of tensor_desc.
  // //    - add all uses to the distribution list.
  // // 2. Walk through def-use chains and distribute all the ops (?)
  // auto root = getOperation();
  // std::deque<Operation *> distributionQueue;
  // root->walk([&](Operation *op) {
  //   if (isXeGPUDescProducer(op)) {
  //     distributionQueue.push_back(op);
  //   }
  // });

  // while (!distributionQueue.empty()) {
  //   Operation *op = distributionQueue.front();
  //   distributionQueue.pop_front();

  //   assert(op && "op is null");

  //   if (isXeGPUDescProducer(op)) {
  //     distributionQueue.emplace_back(op->getUses());
  //     continue;
  //   }
  // }
}
