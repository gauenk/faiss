
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

void runTinyEigh(GpuResources* res,
                 cudaStream_t stream,
                 Tensor<double, 3, true>& covMat,
                 Tensor<double, 3, true>& eigVecs,
                 Tensor<double, 2, true>& eigVals);
void runTinyEigh(GpuResources* res,
                 cudaStream_t stream,
                 Tensor<float, 3, true>& covMat,
                 Tensor<float, 3, true>& eigVecs,
                 Tensor<float, 2, true>& eigVals);
void runTinyEigh(GpuResources* res,
                 cudaStream_t stream,
                 Tensor<half, 3, true>& covMat,
                 Tensor<half, 3, true>& eigVecs,
                 Tensor<half, 2, true>& eigVals);

template <typename T>
void tinyEighOnDevice(GpuResources* resources,int device,
                      cudaStream_t stream,
                      Tensor<T, 3, true>& covMat,
                      Tensor<T, 3, true>& eigVecs,
                      Tensor<T, 2, true>& eigVals) {
  DeviceScope ds(device);
  runTinyEigh(resources,stream,covMat,eigVecs,eigVals);
  // fprintf(stdout,"runTinyEigh.\n");

}

}
}