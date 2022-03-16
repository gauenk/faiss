/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/gpu/utils/Tensor.cuh>

namespace faiss {
namespace gpu {


  void fillBurstPatches(Tensor<float, 4, true>& burst,
                        Tensor<float, 6, true>& patches,
                        Tensor<int, 2, true>& inds,
                        int queryStart, int queryStride,
                        int ws, int wb, int wf,
                        cudaStream_t stream);


  void fillBurstPatches(Tensor<half, 4, true>& burst,
                        Tensor<half, 6, true>& patches,
                        Tensor<int, 2, true>& inds,
                        int queryStart, int queryStride,
                        int ws, int wb, int wf,
                        cudaStream_t stream);

} // namespace gpu
} // namespace faiss
