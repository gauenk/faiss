/**
 * Copyright (c) Kent Gauen
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/gpu/utils/StaticUtils.h>
#include <faiss/gpu/utils/MathOperators.cuh>
#include <faiss/impl/FaissAssert.h>
#include <faiss/gpu/utils/BurstNnfSimpleBlockSelect.cuh>

/****
     Select "topK" from "blockTileSize" of inVals
 ****/

#define ABS(N) (((N)<0)?(-(N)):((N)))

namespace faiss {
  namespace gpu {

    __global__ void burstNnfBlockFill(
	    Tensor<float, 2, true> inVals,
        Tensor<float, 2, true> outVals,
        Tensor<int, 2, true> outKeys){

      int queryIndex = threadIdx.x + blockDim.x * blockIdx.x;
      int numQueries = inVals.getSize(0);
      int numSearch = inVals.getSize(1);
      int k = outVals.getSize(1);
      int compStart = 0;//4000;
      bool legal = queryIndex < numQueries;
      // printf("queryIndex: %d\n",queryIndex);
      // printf("numQueries: %d\n",numQueries);

      if ( legal ) {

        int compIndex = compStart;
        for (int comp = 0; comp < k; ++comp){          
          outVals[queryIndex][comp] = (float)inVals[queryIndex][compIndex];
          outKeys[queryIndex][comp] = (int)compIndex;
          compIndex += 1;
        }
      }
    }

    __global__ void burstNnfBlockSelect(
	    Tensor<float, 2, true> inVals,
        Tensor<float, 2, true> outVals,
        Tensor<int, 2, true> outKeys){

      int queryIndex = threadIdx.x + blockDim.x * blockIdx.x;
      int numQueries = inVals.getSize(0);
      int numSearch = inVals.getSize(1);
      int k = outVals.getSize(1);
      int kidx = 0;
      bool legal = queryIndex < numQueries;
      // printf("queryIndex: %d\n",queryIndex);
      // printf("numQueries: %d\n",numQueries);

      if ( legal ) {

        float outVal_max = outVals[queryIndex][k-1];
        float outVal_curr = outVal_max;
        for (int comp = 0; comp < numSearch; ++comp){

          float inVal = inVals[queryIndex][comp];

          if (inVal < outVal_max){
            kidx = k-1;
            outVal_curr = outVal_max;
            while( inVal < outVal_curr && kidx > 0){
              kidx -= 1;
              outVal_curr = outVals[queryIndex][kidx];
            }
            if (kidx != 0){ kidx += 1; }
            else if (inVal > outVal_curr){ kidx += 1; }

            // shift values up
            for (int sidx = k-1; sidx > kidx; --sidx){
              outVals[queryIndex][sidx] = (float)outVals[queryIndex][sidx-1];
              outKeys[queryIndex][sidx] = (int)outKeys[queryIndex][sidx-1];
            }

            // assign new values
            outVals[queryIndex][kidx] = inVal;
            outKeys[queryIndex][kidx] = comp;
            outVal_max = outVals[queryIndex][k-1];

          }
        }
      }
    }
    
    void runBurstNnfSimpleBlockSelect(
	Tensor<float, 2, true>& inVals,
	Tensor<float, 2, true>& outVals,
	Tensor<int, 2, true>& outKeys,
	cudaStream_t stream){

      // assert shapes 

      // batching
      constexpr int batchQueries = 8;
      constexpr int batchSpace = 1;

      // setup kernel launch
      int maxThreads = (int) getMaxThreadsCurrentDevice();
      int numQueries = inVals.getSize(0);
      int numSearch = inVals.getSize(1);
      int k = outVals.getSize(1);
      
      int numQueriesSqrt = (int)(utils::pow(numQueries*1.0, .5)+1);
      auto grid = dim3(numQueriesSqrt);
      auto block = dim3(numQueriesSqrt);

      // launch kernel
      burstNnfBlockSelect<<<grid, block, 0, stream>>>(inVals, outVals, outKeys);
      // burstNnfBlockFill<<<grid, block, 0, stream>>>(inVals, outVals, outKeys);

      CUDA_TEST_ERROR();

    }
    
  }
}