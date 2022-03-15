/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/gpu/utils/StaticUtils.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/gpu/impl/BurstPatchSearch.cuh>
#include <faiss/gpu/utils/ConversionOperators.cuh>
#include <faiss/gpu/utils/DeviceDefs.cuh>
#include <faiss/gpu/utils/Float16.cuh>
#include <faiss/gpu/utils/MathOperators.cuh>
#include <faiss/gpu/utils/PtxUtils.cuh>
#include <faiss/gpu/utils/Reductions.cuh>

#include <algorithm>

namespace faiss {
namespace gpu {

#define inline_min(a, b) ( (a) < (b) ? (a) : (b) )
// #define macro_max(a, b) (a > b)? a: b 
#define legal_access(a, b, c, d) ((((a) >= (c)) && (((a)+(b)) < (d))) ? true : false)

template <typename T,int BatchQueries,int BatchSpace,bool NormLoop>
__global__ void burstPatchSearchKernel(
	Tensor<T, 4, true, int> burst,
	Tensor<T, 4, true, int> fflow,
	Tensor<T, 4, true, int> bflow,
	Tensor<int, 2, true, int> query,
	Tensor<float, 2, true, int> vals,
	Tensor<int, 2, true, int> inds,
    int spaceStartInput, int ps, int pt, int ws, int wf, int wb){
    extern __shared__ char smemByte[]; // #warps * BatchQueries * BatchSearch * nelements
    float* smem = (float*)smemByte;

    // get the cuda vars 
    int numWarps = utils::divUp(blockDim.x, kWarpSize); 
    int laneId = getLaneId();
    int threadId = threadIdx.x;
    int warpId = threadId / kWarpSize;

    // Warp Id to Frame [assuming patchsize = 7]
    int spatialSize = ps*ps;
    int numWarpsPerFrame = (spatialSize-1) / (kWarpSize+1) + 1; // "divUp"
    int frame_index = threadId / (numWarpsPerFrame * kWarpSize); // proposal frame index
    int spatialId = threadId % (numWarpsPerFrame * kWarpSize); // raster of spatial location

    // Unpack Shapes [Burst]
    int nframes = burst.getSize(0);
    int nftrs = 1;//burst.getSize(1); // fixed to "1" since hsv searc
    int height = burst.getSize(2);
    int width = burst.getSize(3);
    int wsHalf = (ws-1)/2;
    // if ((laneId == 0) && (blockIdx.x == 0)){
    //   // printf("frame_index: %d, frame_index - wb: %d\n",frame_index,frame_index-wb);
    //   printf("nframes: %d, height: %d, width: %d\n",nframes,height,width);
    // }

    // Unpack Shapes [Vals]
    int k = vals.getSize(1);

    // Unpack Shapes [Queries]
    int numQueries = query.getSize(0);
    int numQueryBlocks = numQueries / BatchQueries;

    // Blocks to Indices
    int numPerThread = ps*ps;
    int queryIndexStart = BatchQueries*(blockIdx.x); // batch size is 1 
    int spaceStart = BatchSpace*(blockIdx.y)+spaceStartInput; // batch size is 1
    int timeWindowSize = wf + wb + 1;
    float Z = ps * ps;

    // accumulation location for norm
    float pixNorm[BatchQueries][BatchSpace];

    /*
    // determine if our batchsize is too big for the location;
    // the batchsize at compile time might not be a multiple of batchsize at compute time.
    bool lastRowTile = (rowStart + RowTileSize - 1) >= vals.getSize(0);
    bool lastColTile = (colStart + ColTileSize - 1) >= vals.getSize(1);
    bool lastBlockTile = (blockStart + BlockTileSize - 1) >= vals.getSize(2);
    // bool lastTile = lastBlockTile || lastRowTile || lastColTile;
    */

    bool lastTile = false;

    if (lastTile){
    }else{
      
      if (NormLoop){

      }else{
        // A block of threads is the exact size of the vector
#pragma unroll
        for (int qidx = 0; qidx < BatchQueries; ++qidx) {
#pragma unroll
          for (int sidx = 0; sidx < BatchSpace; ++sidx) {

            //
            //  Get Locations from Offset
            //

            // Unpack Query Index [center pix of patch]
            int queryIndex = queryIndexStart + qidx;
            int r_query_row = query[queryIndex][1];
            int r_query_col = query[queryIndex][2];
              
            // Reference Location [top-left of patch]
            int r_frame = query[queryIndex][0];
            int r_rowTop = r_query_row - ps/2;
            int r_colLeft = r_query_col - ps/2;

            // Unpack Search Index [offset in search space]
            int spaceIndex = spaceStart + sidx;
            int space_row = spaceIndex / ws - wsHalf;
            int space_col = spaceIndex % ws - wsHalf;

            // Proposed Location [top-left of search patch]
            int p_frame = r_frame - wb + frame_index;
            int p_rowTop = r_rowTop + space_row;
            int p_colLeft = r_colLeft + space_col;
              
            //
            // Spatial Index -> (c,h,w) location of patch delta
            //

            // features 
            int ftr = spatialId % nftrs;
            int fDiv = spatialId / nftrs;
    		  
            // width
            int wIdx = fDiv % ps;
            int wDiv = fDiv / ps;
    		  
            // height
            int hIdx = wDiv % ps;
            int hDiv = wDiv / ps;

            //
            // Location [top-left] + Offsets [threadIdx] = Patch Index
            //
    		  
            // Ref Locs
            int r_row = (r_rowTop + hIdx) % height;
            int r_col = (r_colLeft + wIdx) % width;
            r_row = (r_row >= 0) ? r_row : (-r_row);
            r_col = (r_col >= 0) ? r_col : (-r_col);
            
            // Proposed Locs
            int p_row = (p_rowTop + hIdx) % height;
            int p_col = (p_colLeft + wIdx) % width;
            p_row = (p_row >= 0) ? p_row : (-p_row);
            p_col = (p_col >= 0) ? p_col : (-p_col);

            // Check Legal Access [Proposed Location]
            bool flegal = (p_frame >= 0) && (p_frame < nframes);
            bool rlegal = (spatialId >= 0) && (spatialId < spatialSize);
            bool legal = flegal && rlegal;
    		  
            // image values
            T ref_val = burst[r_frame][ftr][r_row][r_col];
            T tgt_val = flegal ? burst[p_frame][ftr][p_row][p_col] : (T)0;
            T diff = Math<T>::sub(ref_val,tgt_val);
            diff = Math<T>::mul(diff,diff);
            diff = flegal ? diff : (T)1e5;
            diff = rlegal ? diff : (T)0.;
            pixNorm[qidx][sidx] = diff;
          }
        }
      }

      // Sum up all parts within each warp
#pragma unroll
        for (int qidx = 0; qidx < BatchQueries; ++qidx) {
#pragma unroll
          for (int sidx = 0; sidx < BatchSpace; ++sidx) {
            pixNorm[qidx][sidx] = warpReduceAllSum(pixNorm[qidx][sidx]);
          }
        }

        // Write each warp's first value into the shared "kernel" memory
        if (laneId == 0) {
#pragma unroll
          for (int qidx = 0; qidx < BatchQueries; ++qidx) {
#pragma unroll
            for (int sidx = 0; sidx < BatchSpace; ++sidx) {
              int smemQueryIndex = qidx;
              int smemSpaceIndex = sidx * BatchQueries;
              int smemBatchIdx = smemQueryIndex + smemSpaceIndex;
              int smemIdx = smemBatchIdx * numWarps + warpId;
              smem[smemIdx] = pixNorm[qidx][sidx];
            }
          }
        }

    }

    // sync across all threads 
    __syncthreads();

    if (warpId == 0) {
      int frame_index = threadIdx.x;
      for(int widx = 0; widx < numWarpsPerFrame; ++widx){
#pragma unroll
        for (int qidx = 0; qidx < BatchQueries; ++qidx) {
#pragma unroll
          for (int sidx = 0; sidx < BatchSpace; ++sidx) {
            int smemQueryIndex = qidx;
            int smemSpaceIndex = sidx * BatchQueries;
            int smemBatchIdx = smemQueryIndex + smemSpaceIndex;
            int smemIdx = smemBatchIdx * numWarps + numWarpsPerFrame * frame_index + widx;
            float val = 0;

            bool flegal = (frame_index < timeWindowSize);
            int outRow = queryIndexStart + qidx;
            int outCol = (spaceStart + sidx) * timeWindowSize + frame_index;
            if (flegal){
              val = smem[smemIdx];
              if (widx  == 0){
                vals[outRow][outCol] = val;
              }else{
                vals[outRow][outCol] += val;
                vals[outRow][outCol] /= Z;
              }
            }
          }
        }
      }
    }
}

template <typename T>
void runBurstNnfL2Norm(Tensor<T, 4, true>& burst,
                       Tensor<T, 4, true>& fflow,
                       Tensor<T, 4, true>& bflow,
                       Tensor<int, 2, true>& query,
                       Tensor<float, 2, true>& vals,
                       Tensor<int, 2, true>& inds,
                       int start, int numSearch,
                       int ps, int pt, int ws, int wf, int wb,
                       cudaStream_t stream){

  int maxThreads = (int)getMaxThreadsCurrentDevice();
  constexpr int batchQueries = 1;
  constexpr int batchSpace = 1;
  bool normLoop = false;

#define RUN_BURST_PATCH_SEARCH(TYPE_T)			\
    do {                                                                      \
         if (normLoop) {                                                       \
         burstPatchSearchKernel<TYPE_T,batchQueries,batchSpace,true>  \
           <<<grid, block, smem, stream>>>(burst, fflow, bflow, query, vals, inds, \
                                           start, ps, pt ,ws ,wf, wb);  \
         } else {                                                              \
         burstPatchSearchKernel<TYPE_T,batchQueries,batchSpace,false>  \
           <<<grid, block, smem, stream>>>(burst, fflow, bflow, query, vals, inds, \
                                           start, ps, pt ,ws ,wf, wb);  \
         }                                                                     \
     } while (0)

//     // compute numThreads
//     int nframes = burst.getSize(0);
//     int nftrs = burst.getSize(1);
//     int dim = patchsize*patchsize*nftrs*nframes;
//     bool normLoop = dim > maxThreads;

    int numQueries = query.getSize(0);
    int numComps = batchQueries * batchSpace;

    int timeWindowSize = wb + wf + 1;
    int warpsPerFrame = (ps * ps - 1) / (kWarpSize + 1) + 1; // "divUp"
    //int dim = timeWindowSize * ps * ps;// dim must include "gaps"; "ps x ps" fits into warp
    int dim = warpsPerFrame * timeWindowSize * kWarpSize;
    FAISS_ASSERT(dim < maxThreads);
    int numThreads = std::min(dim, maxThreads);
    int nWarps = utils::divUp(numThreads, kWarpSize);
//     // numThreads = utils::roundUp(numThreads,kWarpSize); // round-up for warp reduce.
    FAISS_ASSERT(ps == 7);

//     // compute number of Grids
//     int height = vals.getSize(0);
//     int width = vals.getSize(1);
//     int blockBatchSize = blocks.getSize(1);
//     int numToComp = height * width * blockBatchSize;
//     int numToCompPerKernel = rowTileSize * colTileSize * blockTileSize;
//     int numHeightBlocks = utils::divUp(height, rowTileSize);
//     int numWidthBlocks = utils::divUp(width, colTileSize);
//     int numBlockBlocks = utils::divUp(blockBatchSize, blockTileSize);
//     int nBlocks = utils::divUp(numToComp,numToCompPerKernel);

    // get grids and threads 
    int numQueryBlocks = numQueries / batchQueries;
    // fprintf(stdout,"numSearch: %d\n",numSearch);
    auto grid = dim3(numQueryBlocks,numSearch);
    auto block = dim3(numThreads);
    auto smem = sizeof(float) * timeWindowSize * numComps * nWarps;
    // auto grid = dim3(numHeightBlocks,numWidthBlocks,numBlockBlocks);
    // auto block = dim3(numThreads);
    // auto smem = sizeof(float) * numToCompPerKernel * nWarps;

//     // weird convserion for me... ... idk
//     float* tmpTVec;
//     float tmp[1];
//     tmp[0] = 100.;
//     tmpTVec = reinterpret_cast<float*>(tmp);
//     float TVecMax = tmpTVec[0];
    RUN_BURST_PATCH_SEARCH(T);

// #undef RUN_NNF_L2
    CUDA_TEST_ERROR();
}

void runBurstNnfL2Norm(Tensor<float, 4, true>& burst,
                       Tensor<float, 4, true> fflow,
                       Tensor<float, 4, true> bflow,
                       Tensor<int, 2, true>& query,
                       Tensor<float, 2, true>& vals,
                       Tensor<int, 2, true>& inds,
                       int srch_start, int numSearch,
                       int ps, int pt, int ws, int wf, int wb,
                       cudaStream_t stream){
  runBurstNnfL2Norm<float>(burst, fflow, bflow, query, vals, inds,
                           srch_start, numSearch, ps, pt, ws, wf, wb, stream);
}

void runBurstNnfL2Norm(Tensor<half, 4, true>& burst,
                       Tensor<half, 4, true> fflow,
                       Tensor<half, 4, true> bflow,
                       Tensor<int, 2, true>& query,
                       Tensor<float, 2, true>& vals,
                       Tensor<int, 2, true>& inds,
                       int srch_start, int numSearch,
                       int ps, int pt, int ws, int wf, int wb,
                       cudaStream_t stream){
  runBurstNnfL2Norm<half>(burst, fflow, bflow, query, vals, inds,
                          srch_start, numSearch, ps, pt, ws, wf, wb, stream);
}

} // namespace gpu
} // namespace faiss
