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
        int ps, int pt, int wf, int wb, int ws,
        int queryStart, int queryStride,
        Tensor<float, 4, true>& srch_burst,
        Tensor<float, 4, true>& fflow,
        Tensor<float, 4, true>& bflow,
        Tensor<float, 2, true>& outDistances,
        Tensor<int, 2, true>& outIndices);

void runL2Distance(
        GpuResources* resources,
        cudaStream_t stream,
        int ps, int pt, int wf, int wb, int ws,
        int queryStart, int queryStride,
        Tensor<half, 4, true>& srch_burst,
        Tensor<half, 4, true>& fflow,
        Tensor<half, 4, true>& bflow,
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
        int ps, int pt, int wf, int wb, int ws,
        int queryStart, int queryStride,
        Tensor<T, 4, true>& srch_burst,
        Tensor<T, 4, true>& fflow,
        Tensor<T, 4, true>& bflow,
        Tensor<float, 2, true>& outDistances,
        Tensor<int, 2, true>& outIndices) {
    DeviceScope ds(device);

    // L2 and IP are specialized to use GEMM and an optimized L2 + selection or
    // pure k-selection kernel.
    runL2Distance(resources,
                  stream,
                  ps,pt,wf,wb,ws,
                  queryStart,queryStride,
                  srch_burst,
                  fflow,bflow,
                  outDistances,
                  outIndices);
}

template <typename T>
void bfKn3FillOnDevice(
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
    runL2Distance(resources,stream,
                  ps,pt,wf,wb,ws,
                  queryStart,queryStride,
                  srch_burst,
                  fflow,bflow,
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
        T fill_burst, int fill_query) {
    DeviceScope ds(device);
    // We are guaranteed that all data arguments are resident on our preferred
    // `device` here

    thrust::fill(thrust::cuda::par.on(stream),
                 srch_burst.data(),
                 srch_burst.end(),fill_burst);
    // thrust::fill(thrust::cuda::par.on(stream),
    //              queries.data(),
    //              queries.end(),fill_query);

    return;
}

// helper

// We want to tile the search space by splitting ONLY "ws".
// We don't want to split the "wf,wb" because we accumulate
// "flow" offsets inside of the kernel....
// but this means we share information across kernels which is not true.
// one thread is ONE (proposed location,search location) pair
// so actually this comment about splitting over only "ws" doesn't
// make much sense.

inline void chooseKn3TileSize(int numQueries,
                              int numSearch,
                              int elementSize,
                              int& tileQueries,
                              int& tileSearch) {
    // The matrix multiplication should be large enough to be efficient, but if
    // it is too large, we seem to lose efficiency as opposed to
    // double-streaming. Each tile size here defines 1/2 of the memory use due
    // to double streaming.
    // Target 8 GB total useage 
    auto totalMem = getCurrentDeviceProperties().totalGlobalMem;
    int nstreams = 2;
    long long targetUsage = 1024 * 1024 * 1024;
    // fprintf(stdout,"targetUsage: %lld\n",targetUsage);
    targetUsage /= nstreams * elementSize; // usage per stream
    targetUsage *= 2;
    // fprintf(stdout,"targetUsage: %lld\n",targetUsage);

    // fix default queries to specific size
    int preferredTileQueries = 20;
    tileQueries = std::min(preferredTileQueries, numQueries);

    // tileCols is the remainder size
    tileSearch = std::min((int)(targetUsage / tileQueries), numSearch);
}


} // namespace gpu
} // namespace faiss
