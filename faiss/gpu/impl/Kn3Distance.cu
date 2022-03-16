/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/gpu/impl/BroadcastSum.cuh>
#include <faiss/gpu/impl/Kn3Distance.cuh>
#include <faiss/gpu/impl/BurstPatchSearch.cuh>
#include <faiss/gpu/impl/DistanceUtils.cuh>
#include <faiss/gpu/impl/L2Norm.cuh>
#include <faiss/gpu/impl/L2Select.cuh>
#include <faiss/gpu/utils/BlockSelectKernel.cuh>
#include <faiss/gpu/utils/BurstNnfSimpleBlockSelect.cuh>
#include <faiss/gpu/utils/DeviceDefs.cuh>
#include <faiss/gpu/utils/Limits.cuh>
#include <faiss/gpu/utils/MatrixMult.cuh>

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <algorithm>
#include <memory>

namespace faiss {
namespace gpu {

template <typename T>
void runKn3Distance(GpuResources* res,cudaStream_t stream,
                    int ps, int pt, int wf, int wb, int ws,
                    int queryStart, int queryStride,
                    Tensor<T, 4, true>& srch_burst,
                    Tensor<T, 4, true>& fflow,
                    Tensor<T, 4, true>& bflow,
                    Tensor<float, 2, true>& outDistances,
                    Tensor<int, 2, true>& outIndices) {

    // The size of the image burst
    auto nchnls = srch_burst.getSize(0);
    auto nframes = srch_burst.getSize(1);
    auto height = srch_burst.getSize(2);
    auto width = srch_burst.getSize(3);

    // The # of queries; we batch over these twice
    // auto numQueries = queries.getSize(0);
    // auto qdim = queries.getSize(1);

    // The "k" of the knn
    auto k = outDistances.getSize(1);
    auto numQueries = outDistances.getSize(0);

    // The dimensions of the vectors to consider
    // FAISS_ASSERT(qdim == 3);
    FAISS_ASSERT(outDistances.getSize(0) == numQueries);
    FAISS_ASSERT(outIndices.getSize(0) == numQueries);

    // If we're querying against a 0 sized set, just return empty results
    // thrust::fill(thrust::cuda::par.on(stream),
    //              outDistances.data(),
    //              outDistances.end(),
    //              Limits<float>::getMax());
    // thrust::fill(thrust::cuda::par.on(stream),
    //              outIndices.data(),
    //              outIndices.end(),-1);

    // By default, aim to use up to 512 MB of memory for the processing, with
    // both number of queries and number of centroids being at least 512.
    int numSearch = ws*ws;
    int timeWindowSize = wf*wb+1;
    int tileQueries,tileSearch;
    chooseKn3TileSize(numQueries,numSearch,sizeof(T),tileQueries,tileSearch);
    tileQueries = 512;
    tileSearch = numSearch;
    int numQueryTiles = utils::divUp(numQueries, tileQueries);
    int numSearchTiles = utils::divUp(numSearch, tileSearch);
    // fprintf(stdout,"numQueries,numSearch: %d,%d\n",numQueries,numSearch);
    // fprintf(stdout,"tileQueries,tileSearch: %d,%d\n",tileQueries,tileSearch);
    // fprintf(stdout,"numQueryTiles,numSearchTiles: %d,%d\n",numQueryTiles,numSearchTiles);

    //
    // --> Allocate a frame offsets <--
    //
    // DeviceTensor<float, 2, true> f(
    //         res, makeTempAlloc(AllocType::Other, stream),
    //         {tileQueries, tileSearch * timeWindowSize});
    // DeviceTensor<float, 2, true> distanceBuf2(
    //         res, makeTempAlloc(AllocType::Other, stream),
    //         {tileQueries, tileSearch * timeWindowSize});
    // DeviceTensor<float, 2, true>* distanceBufs[2] = {
    //         &distanceBuf1, &distanceBuf2};


    // We can have any number of vectors to query against, even less than k, in
    // which case we'll return -1 for the index
    FAISS_ASSERT(k <= GPU_MAX_SELECTION_K); // select limitation

    // Temporary output memory space we'll use
    DeviceTensor<float, 2, true> distanceBuf1(
            res, makeTempAlloc(AllocType::Other, stream),
            {tileQueries, tileSearch * timeWindowSize});
    DeviceTensor<float, 2, true> distanceBuf2(
            res, makeTempAlloc(AllocType::Other, stream),
            {tileQueries, tileSearch * timeWindowSize});
    DeviceTensor<float, 2, true>* distanceBufs[2] = {
            &distanceBuf1, &distanceBuf2};

    // DeviceTensor<float, 2, true> outDistanceBuf1(
    //         res, makeTempAlloc(AllocType::Other, stream),
    //         {tileQueries, numSearchTiles * k});
    // DeviceTensor<float, 2, true> outDistanceBuf2(
    //         res, makeTempAlloc(AllocType::Other, stream),
    //         {tileQueries, numSearchTiles * k});
    // DeviceTensor<float, 2, true>* outDistanceBufs[2] = {
    //         &outDistanceBuf1, &outDistanceBuf2};

    // DeviceTensor<int, 2, true> outIndexBuf1(
    //         res, makeTempAlloc(AllocType::Other, stream),
    //         {tileQueries, numSearchTiles * k});
    // DeviceTensor<int, 2, true> outIndexBuf2(
    //         res, makeTempAlloc(AllocType::Other, stream),
    //         {tileQueries, numSearchTiles * k});
    // DeviceTensor<int, 2, true>* outIndexBufs[2] = {
    //         &outIndexBuf1, &outIndexBuf2};

    auto streams = res->getAlternateStreamsCurrentDevice();
    streamWait(streams, {stream});

    int curStream = 0;
    bool interrupt = false;

    // Tile over input queries 1
    for (int i = 0; i < numQueries; i += tileQueries) {
        if (interrupt || InterruptCallback::is_interrupted()) {
            interrupt = true;
            break;
        }
        // fprintf(stdout,"i-loop:[%d]\n",i);

        /*

          -- select correct data view --
          
        */
        int curQuerySize = std::min(tileQueries, numQueries - i);
        auto queryStart_i = queryStart + i;
        auto outDistanceView = outDistances.narrow(0, i, curQuerySize);
        auto outIndexView = outIndices.narrow(0, i, curQuerySize);
        // auto outDistanceBufRowView =
        //         outDistanceBufs[curStream]->narrow(0, 0, curQuerySize);
        // auto outIndexBufRowView =
        //         outIndexBufs[curStream]->narrow(0, 0, curQuerySize);

        // Tile over search-space
        for (int j = 0; j < numSearch; j += tileSearch) {
            if (InterruptCallback::is_interrupted()) {
                interrupt = true;
                break;
            }
            // fprintf(stdout,"j-loop:[%d]\n",j);

            /*

              -- select correct data view --

            */

            int curSearchSize = std::min(tileSearch, numSearch - j);
            int curSearchTile = j / tileSearch;

            int fullSearchSize = curSearchSize * timeWindowSize;
            auto distanceBufView = distanceBufs[curStream]
                                           ->narrow(0, 0, curQuerySize)
                                           .narrow(1, 0, fullSearchSize);
            // auto outDistanceBufColView =
            //         outDistanceBufRowView.narrow(1, k * curSearchTile, k);
            // auto outIndexBufColView =
            //         outIndexBufRowView.narrow(1, k * curSearchTile, k);


            /*

              -- exec kernel --

            */
            
            if (curSearchSize == numSearch){ // we search all at once
              
              // thrust::fill(thrust::cuda::par.on(stream),
              //              distanceBufView.data(),
              //              distanceBufView.end(),
              //              Limits<float>::getMax());
              // thrust::fill(thrust::cuda::par.on(stream),
              //              outDistanceView.data(),
              //              outDistanceView.end(),
              //              Limits<float>::getMax());

              runBurstNnfL2Norm(srch_burst,fflow,bflow,
                                queryStart_i,queryStride,
                                distanceBufView,outIndexView,
                                j,curSearchSize,ps,pt,ws,wf,wb,stream);
              runBurstNnfSimpleBlockSelect(distanceBufView,
                                           outDistanceView,
                                           outIndexView,stream);

            }else{ // store in temp bufs

            }

            /*
            // L2: distance is ||c||^2 - 2qc + ||q||^2, we compute -2qc
            // IP: just compute qc
            // (query id x dim) x (centroid id, dim)' = (query id, centroid id)
            runMatrixMult(
                    distanceBufView,
                    false, // not transposed
                    queryView,
                    !queriesRowMajor, // transposed MM if col major
                    centroidsView,
                    centroidsRowMajor, // transposed MM if row major
                    computeL2 ? -2.0f : 1.0f,
                    0.0f,
                    res->getBlasHandleCurrentDevice(),
                    streams[curStream]);

            // For L2 distance, we use this fused kernel that performs both
            // adding ||c||^2 to -2qc and k-selection, so we only need two
            // passes (one write by the gemm, one read here) over the huge
            // region of output memory
            //
            // If we aren't tiling along the number of centroids, we can
            // perform the output work directly
            if (tileCols == numCentroids) {
                // Write into the final output
                runL2SelectMin(
                        distanceBufView,
                        *centroidNorms,
                        outDistanceView,
                        outIndexView,
                        k,
                        streams[curStream]);

                if (!ignoreOutDistances) {
                    // expand (query id) to (query id, k) by duplicating
                    // along rows top-k ||c||^2 - 2qc + ||q||^2 in the form
                    // (query id, k)
                    runSumAlongRows(
                            queryNormNiew,
                            outDistanceView,
                            true, // L2 distances should not go below zero
                                  // due to roundoff error
                            streams[curStream]);
                }
            } else {
                auto centroidNormsView =
                        centroidNorms->narrow(0, j, curCentroidSize);

                // Write into our intermediate output
                runL2SelectMin(
                        distanceBufView,
                        centroidNormsView,
                        outDistanceBufColView,
                        outIndexBufColView,
                        k,
                        streams[curStream]);

                if (!ignoreOutDistances) {
                    // expand (query id) to (query id, k) by duplicating
                    // along rows top-k ||c||^2 - 2qc + ||q||^2 in the form
                    // (query id, k)
                    runSumAlongRows(
                            queryNormNiew,
                            outDistanceBufColView,
                            true, // L2 distances should not go below zero
                                  // due to roundoff error
                            streams[curStream]);
                }
            */
            }


        // As we're finished with processing a full set of centroids, perform
        // the final k-selection
        /*
        if (tileCols != numCentroids) {
            // The indices are tile-relative; for each tile of k, we need to add
            // tileCols to the index
            fprintf(stdout,"here.\n");

            runIncrementIndex(
                    outIndexBufRowView, k, tileCols, streams[curStream]);

            runBlockSelectPair(
                    outDistanceBufRowView,
                    outIndexBufRowView,
                    outDistanceView,
                    outIndexView,
                    computeL2 ? false : true,
                    k,
                    streams[curStream]);
        }
        */


        // curStream = (curStream + 1) % 2;
    }

    // Have the desired ordering stream wait on the multi-stream
    streamWait({stream}, streams);

    if (interrupt) {
        FAISS_THROW_MSG("interrupted");
    }

}

template <typename T>
void runL2Distance(
        GpuResources* res, cudaStream_t stream,
        int ps, int pt, int wf, int wb, int ws,
        int queryStart, int queryStride,
        Tensor<T, 4, true>& srch_burst,
        Tensor<T, 4, true>& fflow,
        Tensor<T, 4, true>& bflow,
        Tensor<float, 2, true>& outDistances,
        Tensor<int, 2, true>& outIndices) {
    runKn3Distance<T>(res,stream,
                      ps,pt,wf,wb,ws,
                      queryStart,queryStride,
                      srch_burst,fflow,bflow,
                      outDistances,outIndices);
}

//
// Instantiations of the distance templates
//

void runL2Distance(
        GpuResources* res, cudaStream_t stream,
        int ps, int pt, int wf, int wb, int ws,
        int queryStart, int queryStride,
        Tensor<float, 4, true>& srch_burst,
        Tensor<float, 4, true>& fflow,
        Tensor<float, 4, true>& bflow,
        Tensor<float, 2, true>& outDistances,
        Tensor<int, 2, true>& outIndices){
    runL2Distance<float>(res,stream,
                         ps,pt,wf,wb,ws,
                         queryStart,queryStride,
                         srch_burst,fflow,bflow,
                         outDistances,outIndices);
}

void runL2Distance(
        GpuResources* res, cudaStream_t stream,
        int ps, int pt, int wf, int wb, int ws,
        int queryStart, int queryStride,
        Tensor<half, 4, true>& srch_burst,
        Tensor<half, 4, true>& fflow,
        Tensor<half, 4, true>& bflow,
        Tensor<float, 2, true>& outDistances,
        Tensor<int, 2, true>& outIndices) {
    runL2Distance<half>(
            res,stream,
            ps,pt,wf,wb,ws,
            queryStart,queryStride,
            srch_burst,fflow,bflow,
            outDistances,outIndices);
}

} // namespace gpu
} // namespace faiss
