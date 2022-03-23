/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/gpu/utils/StaticUtils.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/gpu/impl/FillBurst2Patches.cuh>
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
#define inline_max(a, b) ( (a) > (b) ? (a) : (b) )
// #define macro_max(a, b) (a > b)? a: b 
#define legal_access(a, b, c, d) ((((a) >= (c)) && (((a)+(b)) < (d))) ? true : false)

template <typename T, int BatchQueries>
__global__ void burstPatchFillKernel(
	Tensor<T, 4, true, int> burst,
	Tensor<T, 6, true, int> patches,
	Tensor<int, 2, true, int> inds,
    int queryStart, int queryStride,
    int ws, int wb, int wf,
    int dimPerThread, int numPerQuery, int queriesPerBlock){

    // get the cuda vars 
    int numWarps = utils::divUp(blockDim.x, kWarpSize); 
    int laneId = getLaneId();
    int threadId = threadIdx.x;

    // Unpack Shapes
    int numQueries = patches.getSize(0);
    int k = patches.getSize(1);
    int pt = patches.getSize(2);
    int c = patches.getSize(3);
    int ps = patches.getSize(4);
    int ws2 = ws * ws;
    int nWt = wb + wf + 1;

    // Unpack Shapes [Burst]
    int nframes = burst.getSize(0);
    int height = burst.getSize(2);
    int width = burst.getSize(3);
    int wsHalf = (ws-1)/2;
    int npix = height * width;

    for (int d_index = 0; d_index < dimPerThread; ++d_index){
      for (int qpb = 0; qpb < queriesPerBlock; ++qpb){
#pragma unroll
        for (int qidx = 0; qidx < BatchQueries; ++qidx){

          // compute start of copies 
          int queryIndexStart = queriesPerBlock*BatchQueries*(blockIdx.x);
          // [Query Index]
          int queryIndex = (queryIndexStart + qidx + qpb);
          int queryPix = queryStride*(queryIndex + queryStart);

          // [Patch Index]
          int pindex = dimPerThread * threadIdx.x + d_index;

          if ((queryIndex < numQueries) && (pindex < numPerQuery)){
    
              //
              // [Patch] Indices
              //

              int denom = 1;
              int kIndex = (pindex) % k;
              denom = k;
              int ptIndex = (pindex / denom) % pt;
              denom  = k * pt;
              int cIndex = (pindex / denom) % c;
              denom  = k * pt * c;
              int hIndex = (pindex / denom) % ps;
              denom  = k * pt * c * ps;
              int wIndex = (pindex / denom) % ps;
              // printf("pi,k,pt,c,h,w: %d,%d,%d,%d,%d,%d\n",
              //        pindex,kIndex,ptIndex,cIndex,hIndex,wIndex);

              // [Ref] Location
              int r_frame = queryPix / npix;
              int r_query_row = (queryPix % npix) / width;
              int r_query_col = (queryPix % npix) % width;
              int r_rowTop = r_query_row - (ps/2);
              int r_colLeft = r_query_col - (ps/2);
    
              // Frame Offsets
              int shift_t_min = inline_min(0,r_frame - wb);
              int shift_t_max = inline_max(0,r_frame + wf - nframes + pt);
              int shift_t = shift_t_min + shift_t_max;
              int frame_min = inline_max(r_frame - wb - shift_t,0);
              int frame_min_shift = r_frame - frame_min;
    
              // Search Space Offsets
              int spaceIndex = inds[queryIndex][kIndex];
              int frame_index = spaceIndex % nWt;
              int space_row = ((spaceIndex / nWt) / ws) - wsHalf;
              int space_col = ((spaceIndex / nWt) % ws) - wsHalf;
    
              // [Proposed] Location [top-left of search patch]
              int p_frame = r_frame + frame_index - frame_min_shift;
              int p_rowTop = r_rowTop + space_row;
              int p_colLeft = r_colLeft + space_col;
    
              //
              // [Burst] Indices
              //
    
              int tIndex = p_frame;
              int b_hIndex = p_rowTop + hIndex;
              int b_wIndex = p_colLeft + wIndex;
              b_hIndex = (b_hIndex < height) ?  b_hIndex : (2*height - b_hIndex - 1);
              b_wIndex = (b_wIndex < width) ? b_wIndex : (2*width - b_wIndex - 1);
              b_hIndex = (b_hIndex >= 0) ? b_hIndex : (-b_hIndex-1);
              b_wIndex = (b_wIndex >= 0) ? b_wIndex : (-b_wIndex-1);
              // printf("b_hIndex,b_wIndex: %d,%d\n",b_hIndex,b_wIndex);

              // [Fill] Patches with "val"
    
              T val = burst[tIndex][cIndex][b_hIndex][b_wIndex];
              // T val = burst[0][0][0][0];
              // T val = burst[tIndex][cIndex][0][0];
              // T val = burst[0][cIndex][b_hIndex][b_wIndex];
              // T val = burst[0][0][b_hIndex][b_wIndex];
              // T val = threadId * 1.0;
              // T val = (T)pindex;
              patches[queryIndex][kIndex][ptIndex][cIndex][hIndex][wIndex] = val;
              // patches[queryIndex][kIndex][ptIndex][cIndex][hIndex][wIndex]=spaceIndex;
          }
        }
      }
    }
}

template <typename T>
void fill_burst2patches(Tensor<T, 4, true>& burst,
                        Tensor<T, 6, true>& patches,
                        Tensor<int, 2, true>& inds,
                        int queryStart, int queryStride,
                        int ws, int wb, int wf,
                        cudaStream_t stream){

  // batching 
  constexpr int batchQueries = 8;

  // unpack shapes 
  int maxThreads = (int)getMaxThreadsCurrentDevice();
  int numQueries = inds.getSize(0); // == numPatches
  int k = inds.getSize(1);
  int ps = patches.getSize(5);
  int c = burst.getSize(1);

  // compute num threads
  int numPerQuery = c*ps*ps*k;
  int dimPerThread = 1; // how much does each thread handle
  int threadsPerPatch = ((numPerQuery-1)/(dimPerThread)) + 1;
  int queriesPerBlock = 1;// a function of "k"; smaller "k" -> greater "qpb"
  int numTotalThreads = threadsPerPatch * queriesPerBlock;
  int numThreads = ((numTotalThreads-1) / dimPerThread) + 1;
  while(numThreads > 1024){
    dimPerThread += 1;
    numThreads = ((numTotalThreads-1) / dimPerThread) + 1;
  }
  // fprintf(stdout,"numPerQuery: %d\n",numPerQuery);
  // fprintf(stdout,"dimPerThread: %d\n",dimPerThread);
  // fprintf(stdout,"threadsPerPatch: %d\n",threadsPerPatch);
  // fprintf(stdout,"queriesPerBlock: %d\n",queriesPerBlock);
  // fprintf(stdout,"numThreads: %d\n",numThreads);

  // unpack shape of queries
  int nq = patches.getSize(0);
  int pk = patches.getSize(1);
  int pt = patches.getSize(2);
  int pc = patches.getSize(3);
  int ps1 = patches.getSize(4);
  int ps2 = patches.getSize(5);
  // fprintf(stdout,"pshape = (%d,%d,%d,%d,%d,%d)\n",nq,pk,pt,pc,ps1,ps2);

  int height = burst.getSize(2);
  int width = burst.getSize(3);
  // fprintf(stdout,"bshape = (%d,%d)\n",height,width);


  // get grids and threads 
  int numQueryBlocks = (numQueries-1) / (batchQueries*queriesPerBlock) + 1;
  auto grid = dim3(numQueryBlocks);
  auto block = dim3(numThreads);
  // fprintf(stdout,"numQueryBlocks,numThreads,maxThreads: %d,%d,%d\n",
  //         numQueryBlocks,numThreads,maxThreads);
  // fprintf(stdout,"k,ps: %d,%d\n",k,ps);
  FAISS_ASSERT(numThreads < maxThreads);


  burstPatchFillKernel<T,batchQueries>
    <<<grid, block, 0, stream>>>(burst, patches, inds,
                                 queryStart, queryStride,
                                 ws, wb, wf, dimPerThread, numPerQuery, queriesPerBlock);
    
  CUDA_TEST_ERROR();
}

void fill_burst2patches(Tensor<float, 4, true>& burst,
                        Tensor<float, 6, true>& patches,
                        Tensor<int, 2, true>& inds,
                        int queryStart, int queryStride,
                        int ws, int wb, int wf, cudaStream_t stream){
  fill_burst2patches<float>(burst,patches,inds,queryStart,queryStride,ws,wb,wf,stream);
}

void fill_burst2patches(Tensor<half, 4, true>& burst,
                        Tensor<half, 6, true>& patches,
                        Tensor<int, 2, true>& inds,
                        int queryStart, int queryStride,
                        int ws, int wb, int wf, cudaStream_t stream){
  fill_burst2patches<half>(burst,patches,inds,queryStart,queryStride,ws,wb,wf,stream);
}

} // namespace gpu
} // namespace faiss
