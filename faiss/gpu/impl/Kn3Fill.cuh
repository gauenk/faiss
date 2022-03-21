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
void runKn3Fill(
        GpuResources* resources,cudaStream_t stream,
        int direction, int ps, int pt, int wf, int wb, int ws,
        int queryStart, int queryStride,
        Tensor<float, 4, true>& srch_burst,
        Tensor<float, 6, true>& patches,
        Tensor<int, 2, true>& inds);        

void runKn3Fill(
        GpuResources* resources, cudaStream_t stream,
        int direction, int ps, int pt, int wf, int wb, int ws,
        int queryStart, int queryStride,
        Tensor<half, 4, true>& srch_burst,
        Tensor<half, 6, true>& patches,
        Tensor<int, 2, true>& inds);


template <typename T>
void fillPatches(
        GpuResources* resources,
        int device,
        cudaStream_t stream,
        int ps, int pt, int wf, int wb, int ws,
        int queryStart, int queryStride,
        Tensor<T, 4, true>& srch_burst,
        Tensor<T, 6, true>& patches,
        Tensor<int, 2, true>& inds){
    DeviceScope ds(device);
    int direction = 0; // burst fills patches
    runKn3Fill(resources,stream,direction,
               ps,pt,wf,wb,ws,queryStart,queryStride,
               srch_burst,patches,inds);

}

template <typename T>
void fillBurst(
        GpuResources* resources,
        int device,
        cudaStream_t stream,
        int ps, int pt, int wf, int wb, int ws,
        int queryStart, int queryStride,
        Tensor<T, 4, true>& srch_burst,
        Tensor<T, 6, true>& patches,
        Tensor<int, 2, true>& inds){
    DeviceScope ds(device);
    int direction = 1; // patches fill burst
    runKn3Fill(resources,stream,direction,
               ps,pt,wf,wb,ws,queryStart,queryStride,
               srch_burst,patches,inds);

}


}
}
