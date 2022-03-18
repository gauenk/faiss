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


void runBurstNnfL2Norm(Tensor<float, 4, true>& burst,
                       Tensor<float, 4, true> fflow,
                       Tensor<float, 4, true> bflow,
                       int queryStart, int queryStride,
                       Tensor<float, 2, true>& vals,
                       Tensor<int, 2, true>& inds,
                       int srch_start, int numSearch, int ps, int pt,
                       int ws, int wf, int wb, float bmax,
                       cudaStream_t stream);

void runBurstNnfL2Norm(Tensor<half, 4, true>& burst,
                       Tensor<half, 4, true> fflow,
                       Tensor<half, 4, true> bflow,
                       int queryStart, int queryStride,
                       Tensor<float, 2, true>& vals,
                       Tensor<int, 2, true>& inds,
                       int srch_start, int numSearch, int ps, int pt,
                       int ws, int wf, int wb, float bmax,
                       cudaStream_t stream);

} // namespace gpu
} // namespace faiss
