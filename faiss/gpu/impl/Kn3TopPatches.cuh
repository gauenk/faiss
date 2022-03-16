/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/gpu/impl/GeneralDistance.cuh>
#include <faiss/gpu/utils/DeviceTensor.cuh>
#include <faiss/gpu/utils/Float16.cuh>

namespace faiss {
namespace gpu {

class GpuResources;

/// Calculates brute-force L2 distance between `vectors` and
/// `queries`, returning the k closest results seen
void runKn3TopPatches(
        GpuResources* resources,
        cudaStream_t stream,
        int ps, int pt, int wf, int wb, int ws,
        int queryStart, int queryStride,
        Tensor<float, 4, true>& srch_burst,
        Tensor<float, 4, true>& fflow,
        Tensor<float, 4, true>& bflow,
        Tensor<float, 2, true>& outDistances,
        Tensor<int, 2, true>& outIndices);

void runKn3TopPatches(
        GpuResources* resources,
        cudaStream_t stream,
        int ps, int pt, int wf, int wb, int ws,
        int queryStart, int queryStride,
        Tensor<half, 4, true>& srch_burst,
        Tensor<half, 4, true>& fflow,
        Tensor<half, 4, true>& bflow,
        Tensor<float, 2, true>& outDistances,
        Tensor<int, 2, true>& outIndices);


template <typename T>
void bfKn3TopPatches(
        GpuResources* resources,
        int device,
        cudaStream_t stream,
        int ps, int pt, int wf, int wb, int ws,
        int queryStart, int queryStride,
        Tensor<T, 4, true>& srch_burst,
        Tensor<T, 6, true>& patches,
        Tensor<T, 4, true>& fflow,
        Tensor<T, 4, true>& bflow,
        Tensor<float, 2, true>& outDistances,
        Tensor<int, 2, true>& outIndices){
    DeviceScope ds(device);
    // L2 and IP are specialized to use GEMM and an optimized L2 + selection or
    // pure k-selection kernel.
    fprintf(stdout,"This code needs the _fill_ component added.\n");
    // runKn3TopPatches(resources,
    //                  stream,
    //                  ps,pt,wf,wb,ws,
    //                  queryStart,queryStride,
    //                  srch_burst,
    //                  fflow,bflow,
    //                  outDistances,
    //                  outIndices);
}

}
}
