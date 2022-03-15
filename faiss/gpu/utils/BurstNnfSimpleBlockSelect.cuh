/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/gpu/utils/DeviceTensor.cuh>
#include <faiss/gpu/utils/HostTensor.cuh>

namespace faiss {
  namespace gpu {
    void runBurstNnfSimpleBlockSelect(
         Tensor<float, 2, true>& inVals,
         Tensor<float, 2, true>& outVals,
         Tensor<int, 2, true>& outKeys,
         cudaStream_t stream);
  }
}