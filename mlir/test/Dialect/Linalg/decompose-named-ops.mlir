// RUN: mlir-opt %s -split-input-file -linalg-decompose-named-ops | FileCheck %s

func.func @softmax(%arg0: tensor<2x16x32xf32>) -> tensor<2x16x32xf32> {
  %0 = tensor.empty() : tensor<2x16x32xf32>
  %1 = linalg.softmax dimension(2) ins(%arg0 : tensor<2x16x32xf32>) outs(%0: tensor<2x16x32xf32>) -> tensor<2x16x32xf32>
  return %1 : tensor<2x16x32xf32>
}

// CHECK:      func.func @softmax(%[[ARG0:[a-zA-Z0-9_]+]]: tensor<2x16x32xf32>) -> tensor<2x16x32xf32> {
// CHECK-NEXT %[[RED:.+]] = linalg.reduce(%[[ARG0]])
// CHECK-NEXT %[[CST:.+]] = linalg.broadcast(%[[RED]])
// CHECK-NEXT %[[SUB:.+]] = linalg.sub(%[[ARG0]], %[[CST]])
// CHECK-NEXT %[[EXP:.+]] = linalg.exp(%[[SUB]])
// CHECK-NEXT %[[SUM:.+]] = linalg.reduce(%[[EXP]])
// CHECK-NEXT %[[CST2:.+]] = linalg.broadcast(%[[SUM]])
// CHECK-NEXT %[[DIV:.+]] = linalg.div(%[[%EXP]], %[[CST2]])
// CHECK return %[[DIV]]
