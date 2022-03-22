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
#include <faiss/gpu/impl/TinyEigh.cuh>
#include <faiss/gpu/utils/DeviceDefs.cuh>
#include <faiss/gpu/utils/Limits.cuh>
#include <faiss/gpu/utils/MatrixMult.cuh>

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <algorithm>
#include <memory>

#include <cuda_runtime.h>
#include <cusolverDn.h>

// #include "cusolver_utils.h"

namespace faiss {
namespace gpu {

template <typename T>
void runTinyEigh(GpuResources* res,cudaStream_t stream,
                 Tensor<T, 3, true>& covMat,
                 Tensor<T, 3, true>& eigVecs,
                 Tensor<T, 2, true>& eigVals){
  
  // thrust::fill(thrust::cuda::par.on(stream),
  //              eigVecs.data(),eigVecs.end(),
  //              Limits<float>::getMax());
  // thrust::fill(thrust::cuda::par.on(stream),
  //              eigVals.data(),eigVals.end(),
  //              Limits<float>::getMax());
  int num = covMat.getSize(0);
  int dim = covMat.getSize(1);

  //
  // -- init --
  //

  syevjInfo_t syevj_params = NULL;
  cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; // compute eigenvalues and eigenvectors.
  cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
  int lda = dim;

  // int bsize = 1024*5;
  // -- error handling  --

  // -- precision --
  float residual = 0;
  int executed_sweeps = 0;
  const float tol = 1.e-7;
  const int max_sweeps = 100;
  const int sort_eig = 1;
  CUSOLVER_CHECK(cusolverDnCreateSyevjInfo(&syevj_params));
  CUSOLVER_CHECK(cusolverDnXsyevjSetTolerance(syevj_params, tol));
  // CUSOLVER_CHECK(cusolverDnXsyevjSetMaxSweeps(syevj_params, max_sweeps));
  CUSOLVER_CHECK(cusolverDnXsyevjSetSortEig(syevj_params, sort_eig));


  // CUSOLVER_CHECK(cusolverDnDsyevjBatched_bufferSize(cusolverH, jobz, uplo, dim,\
  //                                                   d_A, lda, d_W, &lwork, \
  //                                                   syevj_params,num));
  // CUDA_VERIFY(cudaMalloc(reinterpret_cast<void **>(&d_work), sizeof(float) * lwork));
  // CUSOLVER_CHECK(cusolverDnDsyevjBatched(cusolverH, jobz, uplo, dim, d_A,\
  //                                        lda, d_W, d_work, lwork, d_info,\
  //                                        syevj_params,num));

  //
  // -- Tiling --
  //

  int batchSize = covMat.getSize(0);
  int tileBatches = 5*1024;
  int nstreams = 4;

  //
  // -- Get our Streams --
  //

  auto streams = res->getAlternateStreamsCurrentDevice();
  streamWait(streams, {stream});
  int curStream = 0;


  //
  // -- nullptr --
  //

  int* d_info = nullptr;
  CUDA_VERIFY(cudaMalloc(reinterpret_cast<void **>(&d_info), batchSize*sizeof(int)));
  
  //
  // -- run pointers --
  //

  int lwork = 0;
  float *d_work = nullptr;
  float* d_A = (float*)covMat.data();  
  float* d_W = (float*)eigVals.data();
  int* d_info_ptr = d_info;

  //
  // -- set buffer --
  //

  cusolverDnHandle_t cusolverH;
  CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));
  CUSOLVER_CHECK(cusolverDnSetStream(cusolverH, streams[curStream]));
  CUSOLVER_CHECK(cusolverDnSsyevjBatched_bufferSize(cusolverH, jobz, uplo, dim, \
                                                    d_A, lda, d_W, &lwork, \
                                                    syevj_params,tileBatches));
  CUDA_VERIFY(cudaMalloc(reinterpret_cast<void **>(&d_work), sizeof(float) * lwork));
  CUSOLVER_CHECK(cusolverDnDestroy(cusolverH));


  //
  // -- batching --
  //

  for (int i = 0; i < batchSize; i += tileBatches) {

    // -- slice across batch --
    int curBatchSize = std::min(tileBatches, batchSize - i);
    auto covMatView = covMat.narrow(0,i,curBatchSize);
    auto eigValsView = eigVals.narrow(0,i,curBatchSize);

    // -- compute spectrum --
    d_A = (float*)covMatView.data();
    d_W = (float*)eigValsView.data();
    d_info_ptr = d_info + i;

    // -- solver handle --
    cusolverDnHandle_t cusolverH_i;
    CUSOLVER_CHECK(cusolverDnCreate(&cusolverH_i));
    CUSOLVER_CHECK(cusolverDnSetStream(cusolverH_i, streams[curStream]));
    CUSOLVER_CHECK(cusolverDnSsyevjBatched(cusolverH_i, jobz, uplo, dim, d_A,\
                                           lda, d_W, d_work, lwork, d_info_ptr,\
                                           syevj_params, curBatchSize));
    CUSOLVER_CHECK(cusolverDnDestroy(cusolverH_i));

    curStream = (curStream + 1) % nstreams;

  }

  // /* step 4: query working space of syevj */
  // CUSOLVER_CHECK(cusolverDnDsyevj_bufferSize(cusolverH, jobz, uplo, m,
  //                                            d_A, lda, d_W, &lwork, syevj_params));
  /* step 5: compute eigen-pair   */
  // CUSOLVER_CHECK(cusolverDnDsyevj(cusolverH, jobz, uplo, m, d_A, lda, d_W,
  //                                 d_work, lwork, devInfo,syevj_params));

  // Have the desired ordering stream wait on the multi-stream
  streamWait({stream}, streams);

  //
  // -- clean up --
  //

  CUDA_VERIFY(cudaFree(d_info));
  CUDA_VERIFY(cudaFree(d_work));
  CUSOLVER_CHECK(cusolverDnDestroySyevjInfo(syevj_params));


}


void runTinyEigh(GpuResources* res,
                 cudaStream_t stream,
                 Tensor<double, 3, true>& covMat,
                 Tensor<double, 3, true>& eigVecs,
                 Tensor<double, 2, true>& eigVals){
  runTinyEigh<double>(res,stream,covMat,eigVecs,eigVals);
}

void runTinyEigh(GpuResources* res,
                 cudaStream_t stream,
                 Tensor<float, 3, true>& covMat,
                 Tensor<float, 3, true>& eigVecs,
                 Tensor<float, 2, true>& eigVals){
  runTinyEigh<float>(res,stream,covMat,eigVecs,eigVals);
}

void runTinyEigh(GpuResources* res,
                 cudaStream_t stream,
                 Tensor<half, 3, true>& covMat,
                 Tensor<half, 3, true>& eigVecs,
                 Tensor<half, 2, true>& eigVals){
  runTinyEigh<half>(res,stream,covMat,eigVecs,eigVals);
}
  

}
}