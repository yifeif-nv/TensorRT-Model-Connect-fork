# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_sam2 SHARED
  plugin.cpp
  sam2_bbox_postprocess.cpp
  sam2_device_workspace.cpp
  sam2_native_video_processor.cpp
  sam2_pipeline.cpp
  sam2_preprocess.cpp
  sam2_mask_postprocess_cuda.cu
  sam2_preprocess_cuda.cu
)

set_source_files_properties(
  sam2_mask_postprocess_cuda.cu
  sam2_preprocess_cuda.cu
  PROPERTIES LANGUAGE CUDA
)

target_include_directories(trtmc_model_sam2
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_include_directories(trtmc_model_sam2 SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
)
target_link_libraries(trtmc_model_sam2 PRIVATE
  trtmc_core
  ${TRTMC_CUDART_LIBRARY}
)
target_compile_options(trtmc_model_sam2 PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
  "$<$<COMPILE_LANGUAGE:CUDA>:-Wall;-Wextra>"
)
set_target_properties(trtmc_model_sam2 PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_sam2
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
