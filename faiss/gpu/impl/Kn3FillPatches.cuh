/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// #include <faiss/gpu/impl/GeneralDistance.cuh>
// #include <faiss/gpu/utils/DeviceTensor.cuh>
// #include <faiss/gpu/utils/Float16.cuh>

namespace faiss {
namespace gpu {

class GpuResources;

/// Calculates brute-force L2 distance between `vectors` and
/// `queries`, returning the k closest results seen
void runKn3FillPatches(
        GpuResources* resources,
        cudaStream_t stream,
        int ps, int pt, int wf, int wb, int ws,
        int queryStart, int queryStride,
        Tensor<float, 4, true>& srch_burst,
        Tensor<float, 6, true>& patches);

void runKn3FillPatches(
        GpuResources* resources,
        cudaStream_t stream,
        int ps, int pt, int wf, int wb, int ws,
        int queryStart, int queryStride,
        Tensor<half, 4, true>& srch_burst,
        Tensor<half, 6, true>& patches);


template <typename T>
void bfKn3FillPatches(
        GpuResources* resources,
        int device,
        cudaStream_t stream,
        int ps, int pt, int wf, int wb, int ws,
        int queryStart, int queryStride,
        Tensor<T, 4, true>& srch_burst,
        Tensor<T, 6, true>& patches){
    DeviceScope ds(device);
    // L2 and IP are specialized to use GEMM and an optimized L2 + selection or
    // pure k-selection kernel.
    runKn3FillPatches(resources,
                      stream,
                      ps,pt,wf,wb,ws,
                      queryStart,queryStride,
                      srch_burst,
                      patches);

}

}
}
