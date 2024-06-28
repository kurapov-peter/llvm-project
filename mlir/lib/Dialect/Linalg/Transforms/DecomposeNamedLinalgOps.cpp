
#include "mlir/Dialect/Linalg/Passes.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

namespace mlir {
#define GEN_PASS_DEF_LINALGDECOMPOSENAMEDOPSPASS
#include "mlir/Dialect/Linalg/Passes.h.inc"
} // namespace mlir

#define DEBUG_TYPE "linalg-decompose-named-ops"

using namespace mlir;
using namespace mlir::linalg;

namespace {
TypedAttr createInitValueForReduceMaxOp(Type type, PatternRewriter &rewriter) {
  if (isa<FloatType>(type))
    return rewriter.getFloatAttr(
        type, APFloat::getSmallest(cast<FloatType>(type).getFloatSemantics()));
  if (isa<IntegerType>(type))
    return rewriter.getIntegerAttr(
        type, APInt::getSignedMinValue(type.getIntOrFloatBitWidth()));
  return {};
}

TypedAttr createInitValueForReduceSumOp(Type type, PatternRewriter &rewriter) {
  if (isa<FloatType>(type))
    return rewriter.getFloatAttr(
        type, APFloat::getZero(cast<FloatType>(type).getFloatSemantics()));
  if (isa<IntegerType>(type))
    return rewriter.getIntegerAttr(
        type, APInt::getZero(type.getIntOrFloatBitWidth()));
  return {};
}

Value createLinalgReduceMaxBody(PatternRewriter &rewriter, Location loc,
                                ValueRange args, Type elementTy) {
  if (isa<FloatType>(elementTy))
    return rewriter.create<arith::MaxNumFOp>(loc, args[0], args[1]);
  if (isa<IntegerType>(elementTy))
    return rewriter.create<arith::MaxSIOp>(loc, args[0], args[1]);
  return {};
}

Value createLinalgReduceSumBody(PatternRewriter &rewriter, Location loc,
                                ValueRange args, Type elementTy) {
  if (isa<FloatType>(elementTy))
    return rewriter.create<arith::AddFOp>(loc, args[0], args[1]);
  if (isa<IntegerType>(elementTy))
    return rewriter.create<arith::AddIOp>(loc, args[0], args[1]);
  return {};
}

struct DecomposeSoftmaxPattern : public OpRewritePattern<SoftmaxOp> {
  using OpRewritePattern<SoftmaxOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SoftmaxOp op,
                                PatternRewriter &rewriter) const override {
    // Decompose softmax(x) into tmp = exp(x - max(x)); tmp / sum(tmp)
    auto loc = op.getLoc();
    auto inputTy = op.getInputOperandType();
    auto resultTy = op.getOutputOperandType();
    auto dim = op.getDimension();
    auto elementTy = resultTy.getElementType();

    SmallVector<int64_t> reduceShape;
    SmallVector<Value> dynDims;
    for (unsigned i = 0; i < inputTy.getRank(); i++) {
      if (dim != i) {
        reduceShape.push_back(inputTy.getDimSize(i));
        if (inputTy.isDynamicDim(i))
          dynDims.push_back(
              rewriter.create<tensor::DimOp>(loc, op.getInput(), i));
      }
    }
    auto emptyTensor =
        rewriter
            .create<tensor::EmptyOp>(loc, reduceShape,
                                     resultTy.getElementType(), dynDims)
            .getResult();
    auto fillValAttr = createInitValueForReduceMaxOp(elementTy, rewriter);
    if (!fillValAttr)
      return rewriter.notifyMatchFailure(
          op, "No initial value found for reduction operation");

    auto fillValue = rewriter.create<arith::ConstantOp>(loc, fillValAttr);
    auto filledTensor = rewriter
                            .create<linalg::FillOp>(loc, ValueRange{fillValue},
                                                    ValueRange{emptyTensor})
                            .result();
    auto reduceOp = rewriter.create<linalg::ReduceOp>(
        loc, op.getInput(), filledTensor, dim,
        [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
          auto result =
              createLinalgReduceMaxBody(rewriter, nestedLoc, args, elementTy);
          nestedBuilder.create<linalg::YieldOp>(nestedLoc, result);
        });

    auto bcastOutput = rewriter.create<tensor::EmptyOp>(
        loc, op.getOutputOperandType().getShape(), elementTy);
    auto broadcastedReduction = rewriter.create<linalg::BroadcastOp>(
        loc, reduceOp.getResult(0), bcastOutput, reduceOp.getDimensionsAttr());
    Value broadcastedReductionTensor = broadcastedReduction.getResults()[0];

    auto subOp = rewriter.create<linalg::SubOp>(
        loc, ValueRange{op.getInput(), broadcastedReductionTensor},
        ValueRange{broadcastedReductionTensor});
    auto expOp =
        rewriter.create<linalg::ExpOp>(loc, ValueRange{subOp.getResult(0)},
                                       ValueRange{broadcastedReductionTensor});

    auto sumEmptyTensor =
        rewriter
            .create<tensor::EmptyOp>(loc, reduceShape,
                                     resultTy.getElementType(), dynDims)
            .getResult();
    auto sumFillValAttr = createInitValueForReduceSumOp(elementTy, rewriter);
    if (!sumFillValAttr)
      return rewriter.notifyMatchFailure(
          op, "No initial value found for reduction operation");

    auto sumFillValue = rewriter.create<arith::ConstantOp>(loc, sumFillValAttr);
    auto sumFilledTensor =
        rewriter
            .create<linalg::FillOp>(loc, ValueRange{sumFillValue},
                                    ValueRange{sumEmptyTensor})
            .result();
    auto reduceSumOp = rewriter.create<linalg::ReduceOp>(
        loc, expOp.getResults(), sumFilledTensor, dim,
        [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
          auto result =
              createLinalgReduceSumBody(rewriter, nestedLoc, args, elementTy);
          nestedBuilder.create<linalg::YieldOp>(nestedLoc, result);
        });
    auto sumBcastOutput = rewriter.create<tensor::EmptyOp>(
        loc, op.getOutputOperandType().getShape(), elementTy);
    auto sumBroadcastedReduction = rewriter.create<linalg::BroadcastOp>(
        loc, reduceSumOp.getResult(0), sumBcastOutput,
        reduceSumOp.getDimensionsAttr());
    Value sumBroadcastedReductionTensor =
        sumBroadcastedReduction.getResults()[0];
    auto divOp = rewriter.create<linalg::DivOp>(
        loc, ValueRange{expOp.getResult(0), sumBroadcastedReductionTensor},
        ValueRange{op.getOutput()});
    rewriter.replaceOp(op, divOp.getResults());
    return success();
  }
};

} // namespace

struct LinalgDecomposeNamedOpsPass
    : public impl::LinalgDecomposeNamedOpsPassBase<
          LinalgDecomposeNamedOpsPass> {
  using impl::LinalgDecomposeNamedOpsPassBase<
      LinalgDecomposeNamedOpsPass>::LinalgDecomposeNamedOpsPassBase;

  void runOnOperation() override;
};

void LinalgDecomposeNamedOpsPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  populateDecomposeNamedOpsPatterns(patterns);
  (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
}

void mlir::linalg::populateDecomposeNamedOpsPatterns(
    RewritePatternSet &patterns) {
  patterns.insert<DecomposeSoftmaxPattern>(patterns.getContext());
}
