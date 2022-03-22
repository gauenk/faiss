/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/Index.h>
#include <faiss/gpu/GpuDistance.h>

namespace faiss {
namespace gpu {

class GpuResourcesProvider;


enum class TinyEighFxnName {
    SYM = 1,
};


struct GpuTinyEighParams {
  GpuTinyEighParams()
    : num(0),dim(0),rank(0),
      fxn_name(TinyEighFxnName::SYM),
      covMat(nullptr),
      eigVecs(nullptr),
      eigVals(nullptr),
      vectorType(DistanceDataType::F64) {}

    //
    // Matrix Size
    //
    TinyEighFxnName fxn_name;
    int num,dim,rank;

    //
    // Covariance Matrix
    //

    const void* covMat;
    const void* eigVecs;
    const void* eigVals;
    DistanceDataType vectorType;

};

/// A wrapper for gpu/impl/Distance.cuh to expose direct brute-force k-nearest
/// neighbor searches on an externally-provided region of memory (e.g., from a
/// pytorch tensor).
/// The data (vectors, queries, outDistances, outIndices) can be resident on the
/// GPU or the CPU, but all calculations are performed on the GPU. If the result
/// buffers are on the CPU, results will be copied back when done.
///
/// All GPU computation is performed on the current CUDA device, and ordered
/// with respect to resources->getDefaultStreamCurrentDevice().
///
/// For each vector in `queries`, searches all of `vectors` to find its k
/// nearest neighbors with respect to the given metric
void tinyEigh(GpuResourcesProvider* resources, const GpuTinyEighParams& args);

} // namespace gpu
} // namespace faiss
