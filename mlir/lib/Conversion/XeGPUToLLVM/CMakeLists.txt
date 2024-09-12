add_mlir_conversion_library(MLIRXeGPUToLLVM
  XeGPUToLLVM.cpp
  XeGPUToLLVMPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/XeGPUToLLVM

  DEPENDS
  MLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRXeGPUDialect
  MLIRLLVMDialect
)
