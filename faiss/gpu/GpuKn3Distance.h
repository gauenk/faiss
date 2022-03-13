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

// // Scalar type of the vector data
// enum class DistanceDataType {
//     F32 = 1,
//     F16,
// };

enum class Kn3FxnName {
    KDISTS = 1,
    KFILL = 2,
    FILLOUT = 3,
    FILLIN = 4,
    PFILLTEST = 5,
};

/// Arguments to brute-force GPU k-nearest neighbor searching
struct GpuKn3DistanceParams {
    GpuKn3DistanceParams()
            : metric(faiss::MetricType::METRIC_L2),
              metricArg(0),
              fxn_name(Kn3FxnName::KDISTS),
              k(0),
              ps(0),
              pt(0),
              nchnls(0),
              nframes(0),
              height(0),
              width(0),
              srch_burst(nullptr),
              fill_burst(nullptr),
              patches(nullptr),
              vectorType(DistanceDataType::F32),
              queries(nullptr),
              queriesRowMajor(true),
              numQueries(0),
              outDistances(nullptr),
              ignoreOutDistances(false),
              outIndices(nullptr),
              fill_a(0.),
              fill_b(0) {}

    //
    // Search parameters
    //

    // function to run
    Kn3FxnName fxn_name;

    /// Search parameter: distance metric
    faiss::MetricType metric;

    /// Search parameter: distance metric argument (if applicable)
    /// For metric == METRIC_Lp, this is the p-value
    float metricArg;

    /// Search parameter: return k nearest neighbors
    /// If the value provided is -1, then we report all pairwise distances
    /// without top-k filtering
    int k;

    /// Patchsize dimensionality
    int ps,pt;

    //
    // Vectors being queried
    //

    /// If vectorsRowMajor is true, this is
    /// numVectors x dims, with dims innermost; otherwise,
    /// dims x numVectors, with numVectors innermost
    const void* srch_burst;
    const void* fill_burst;
    const void* patches;
    int nchnls,nframes,height,width;
    DistanceDataType vectorType;

    /// Precomputed L2 norms for each vector in `vectors`, which can be
    /// optionally provided in advance to speed computation for METRIC_L2

    //
    // The query vectors (i.e., find k-nearest neighbors in `vectors` for each
    // of the `queries`
    //

    /// If queriesRowMajor is true, this is
    /// numQueries x dims, with dims innermost; otherwise,
    /// dims x numQueries, with numQueries innermost
    const void* queries;
    bool queriesRowMajor;
    int numQueries;

    //
    // Output results
    //

    /// A region of memory size numQueries x k, with k
    /// innermost (row major) if k > 0, or if k == -1, a region of memory of
    /// size numQueries x numVectors
    float* outDistances;

    /// Do we only care about the indices reported, rather than the output
    /// distances? Not used if k == -1 (all pairwise distances)
    bool ignoreOutDistances;

    /// A region of memory size numQueries x k, with k
    /// innermost (row major). Not used if k == -1 (all pairwise distances)
    void* outIndices;

    // value for filling
    float fill_a;
    int fill_b;
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
void bfKn3(GpuResourcesProvider* resources, const GpuKn3DistanceParams& args);

} // namespace gpu
} // namespace faiss
