/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/gpu/GpuTinyEigh.h>
#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/gpu/impl/TinyEigh.cuh>
#include <faiss/gpu/utils/ConversionOperators.cuh>
#include <faiss/gpu/utils/CopyUtils.cuh>
#include <faiss/gpu/utils/DeviceTensor.cuh>

namespace faiss {
namespace gpu {

template <typename T>
void tinyEighConvert(GpuResourcesProvider* prov, const GpuTinyEighParams& args) {

    // Validate the input data
    // FAISS_THROW_IF_NOT_MSG(args.num > 0, "bfKn3: k must be > 0 for top-k reduction");
    // FAISS_THROW_IF_NOT_MSG(args.dim > 0, "bfKn3: patchsize must be > 0");
    
    // Don't let the resources go out of scope
    auto resImpl = prov->getResources();
    auto res = resImpl.get();
    auto device = getCurrentDevice();
    auto stream = res->getDefaultStreamCurrentDevice();

    /*

      ---------------------------
      --> Move Data To Device <--
      ---------------------------

     */

    // Distances and Queries -> Device
    auto covMat = toDeviceTemporary<T, 3>(
            res,device,
            const_cast<T*>(reinterpret_cast<const T*>(args.covMat)),
            stream,{args.num,args.dim,args.dim});
    auto eigVecs = toDeviceTemporary<T, 3>(
            res,device,
            const_cast<T*>(reinterpret_cast<const T*>(args.eigVecs)),
            stream,{args.num,args.dim,args.dim});
    auto eigVals = toDeviceTemporary<T, 2>(
            res,device,
            const_cast<T*>(reinterpret_cast<const T*>(args.eigVals)),
            stream,{args.num,args.dim});

    // Since we've guaranteed that all arguments are
    // on device, call the implementation

    if (args.fxn_name == TinyEighFxnName::SYM){
      tinyEighOnDevice<T>(res,device,stream,covMat,eigVecs,eigVals);
    }else{
      FAISS_THROW_MSG("unknown function name.");
    }

}

void tinyEigh(GpuResourcesProvider* res, const GpuTinyEighParams& args) {

    if (args.vectorType == DistanceDataType::F64) {
        tinyEighConvert<double>(res, args);
    }else if (args.vectorType == DistanceDataType::F32) {
        tinyEighConvert<float>(res, args);
    } else if (args.vectorType == DistanceDataType::F16) {
        tinyEighConvert<half>(res, args);
    } else {
        FAISS_THROW_MSG("unknown vectorType");
    }
}

}
}
