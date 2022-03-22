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
#include <faiss/gpu/impl/Kn3Fill.cuh>
#include <faiss/gpu/impl/FillPatches2Burst.cuh>
#include <faiss/gpu/impl/FillBurst2Patches.cuh>
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
void runKn3Fill(GpuResources* res,cudaStream_t stream,
                int direction, int ps, int pt, int wf, int wb, int ws,
                int queryStart, int queryStride,
                Tensor<T, 4, true>& fill_burst,
                Tensor<T, 6, true>& patches,
                Tensor<int, 2, true>& inds) {

    // The size of the image burst
    auto nframes = fill_burst.getSize(0);
    auto nchnls = fill_burst.getSize(1);
    auto height = fill_burst.getSize(2);
    auto width = fill_burst.getSize(3);
    auto numQueries = patches.getSize(0);
    auto k = patches.getSize(1);
    auto burst_npix = height * width * nframes;
    auto queryMax = ((int)(burst_npix-1) / queryStride)+1;
    fprintf(stdout,"queryMax,numQueries: %d,%d\n",queryMax,numQueries);

    // Size checking
    FAISS_ASSERT(numQueries <= queryMax);

    // Tiling 
    int numSearch = ws*ws;
    int timeWindowSize = wf+wb+1;
    int tileQueries,tileSearch;
    // chooseKn3TileSize(numQueries,numSearch,sizeof(T),tileQueries,tileSearch);
    tileQueries = 8*1024;
    tileSearch = numSearch;
    // theirs was 512 x 40960
    int numQueryTiles = utils::divUp(numQueries, tileQueries);
    int numSearchTiles = utils::divUp(numSearch, tileSearch);
    // fprintf(stdout,"numQueries,numSearch: %d,%d\n",numQueries,numSearch);
    // fprintf(stdout,"tileQueries,tileSearch: %d,%d\n",tileQueries,tileSearch);
    // fprintf(stdout,"numQueryTiles,numSearchTiles: %d,%d\n",numQueryTiles,numSearchTiles);

    // streams 
    auto streams = res->getAlternateStreamsCurrentDevice();
    streamWait(streams, {stream});

    // looping params
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
        auto patchesView = patches.narrow(0, i, curQuerySize);
        auto indsView = inds;
        if (direction == 0){
          indsView = inds.narrow(0, i, curQuerySize);
        }

        // Tile over search-space
        for (int j = 0; j < numSearch; j += tileSearch) {
            if (InterruptCallback::is_interrupted()) {
                interrupt = true;
                break;
            }
            // fprintf(stdout,"j-loop:[%d]\n",j);

            // select correct data view

            int curSearchSize = std::min(tileSearch, numSearch - j);
            int curSearchTile = j / tileSearch;
            int fullSearchSize = curSearchSize * timeWindowSize;

            // exec kernel 

            if (curSearchSize == numSearch){ // we search all at once

              if (direction == 0){  // burst fills patches
                // fprintf(stdout,"fill patches with a burst\n");
                // FAISS_ASSERT(inds.getSize(0) == curQuerySize);
                fill_burst2patches(fill_burst,patchesView,indsView,queryStart_i,
                                   queryStride,ws,wb,wf,stream);
              }else if(direction == 1){ // patches fill burst
                fill_patches2burst(fill_burst,patchesView,queryStart_i,
                                   queryStride,ws,wb,wf,stream);
              }else{
                FAISS_THROW_MSG("[Kn3Fill.cu]: fill direction invalid");
              }

            }else{ // store in temp bufs
              FAISS_THROW_MSG("[Kn3Fill.cu]: bad tiling");
            }
        }

        // curStream = (curStream + 1) % 2;
    }

    // Have the desired ordering stream wait on the multi-stream
    streamWait({stream}, streams);

    if (interrupt) {
        FAISS_THROW_MSG("interrupted");
    }

}



//
// Instantiations of the distance templates
//

void runKn3Fill(
        GpuResources* res, cudaStream_t stream,
        int direction,
        int ps, int pt, int wf, int wb, int ws,
        int queryStart, int queryStride,
        Tensor<float, 4, true>& fill_burst,
        Tensor<float, 6, true>& patches,
        Tensor<int, 2, true>& inds){
  runKn3Fill<float>(res,stream,direction,
                    ps,pt,wf,wb,ws,
                    queryStart,queryStride,
                    fill_burst,patches,inds);
}

void runKn3Fill(
        GpuResources* res, cudaStream_t stream,
        int direction,
        int ps, int pt, int wf, int wb, int ws,
        int queryStart, int queryStride,
        Tensor<half, 4, true>& fill_burst,
        Tensor<half, 6, true>& patches,
        Tensor<int, 2, true>& inds){
  runKn3Fill<half>(res,stream,direction,
                   ps,pt,wf,wb,ws,
                   queryStart,queryStride,
                   fill_burst,patches,inds);
}

} // namespace gpu
} // namespace faiss
