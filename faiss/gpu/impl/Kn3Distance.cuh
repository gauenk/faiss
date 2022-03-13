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
void runL2Distance(
        GpuResources* resources,
        cudaStream_t stream,
        Tensor<float, 4, true>& srch_burst,
        Tensor<int, 2, true>& queries,
        Tensor<float, 2, true>& outDistances,
        Tensor<int, 2, true>& outIndices);

void runL2Distance(
        GpuResources* resources,
        cudaStream_t stream,
        Tensor<half, 4, true>& srch_burst,
        Tensor<int, 2, true>& queries,
        Tensor<float, 2, true>& outDistances,
        Tensor<int, 2, true>& outIndices);

//
// General distance implementation, assumes that all arguments are on the
// device. This is the top-level internal distance function to call to dispatch
// based on metric type.
//
template <typename T>
void bfKn3OnDevice(
        GpuResources* resources,
        int device,
        cudaStream_t stream,
        Tensor<T, 4, true>& srch_burst,
        Tensor<int, 2, true>& queries,
        Tensor<float, 2, true>& outDistances,
        Tensor<int, 2, true>& outIndices) {
    DeviceScope ds(device);

    // L2 and IP are specialized to use GEMM and an optimized L2 + selection or
    // pure k-selection kernel.
    fprintf(stdout,"standard search.\n");
    runL2Distance(resources,
                  stream,
                  srch_burst,
                  queries,
                  outDistances,
                  outIndices);
}

template <typename T>
void bfKn3FillOnDevice(
        GpuResources* resources,
        int device,
        cudaStream_t stream,
        Tensor<T, 4, true>& srch_burst,
        Tensor<int, 2, true>& queries,
        Tensor<float, 2, true>& outDistances,
        Tensor<int, 2, true>& outIndices){
    DeviceScope ds(device);
    // L2 and IP are specialized to use GEMM and an optimized L2 + selection or
    // pure k-selection kernel.
    fprintf(stdout,"This code needs the _fill_ component added.\n");
    runL2Distance(resources,
                  stream,
                  srch_burst,
                  queries,
                  outDistances,
                  outIndices);
}



//
// General distance implementation, assumes that all arguments are on the
// device. This is the top-level internal distance function to call to dispatch
// based on metric type.
//
/*

  Tests to fill out Matrices


 */


template <typename T>
void kn3FillTestPatches(
        GpuResources* resources,
        int device,
        cudaStream_t stream,
        Tensor<T, 6, true>& patches,
        T fill_val) {
    DeviceScope ds(device);
    // test filling patches
    thrust::fill(thrust::cuda::par.on(stream),
                 patches.data(),
                 patches.end(),fill_val);
    return;
}

template <typename T>
void kn3FillOutMats(
        GpuResources* resources,
        int device,
        cudaStream_t stream,
        Tensor<T, 4, true>& srch_burst,
        Tensor<int, 2, true>& queries,
        Tensor<float, 2, true>& outDistances,
        Tensor<int, 2, true>& outIndices,
        T fill_dists, int fill_inds) {
    DeviceScope ds(device);
    // test filling output bufs
    thrust::fill(thrust::cuda::par.on(stream),
                 outDistances.data(),
                 outDistances.end(),fill_dists);
    thrust::fill(thrust::cuda::par.on(stream),
                 outIndices.data(),
                 outIndices.end(),fill_inds);

    return;
}

template <typename T>
void kn3FillInMats(
        GpuResources* resources,
        int device,
        cudaStream_t stream,
        Tensor<T, 4, true>& srch_burst,
        Tensor<int, 2, true>& queries,
        T fill_burst, int fill_query) {
    DeviceScope ds(device);
    // We are guaranteed that all data arguments are resident on our preferred
    // `device` here

    thrust::fill(thrust::cuda::par.on(stream),
                 srch_burst.data(),
                 srch_burst.end(),fill_burst);
    thrust::fill(thrust::cuda::par.on(stream),
                 queries.data(),
                 queries.end(),fill_query);

    return;
}


} // namespace gpu
} // namespace faiss
