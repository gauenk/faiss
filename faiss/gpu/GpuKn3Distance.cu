/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/gpu/GpuKn3Distance.h>
#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/gpu/impl/Kn3Distance.cuh>
#include <faiss/gpu/utils/ConversionOperators.cuh>
#include <faiss/gpu/utils/CopyUtils.cuh>
#include <faiss/gpu/utils/DeviceTensor.cuh>

namespace faiss {
namespace gpu {

template <typename T>
void bfKn3Convert(GpuResourcesProvider* prov, const GpuKn3DistanceParams& args) {

    // Validate the input data
    FAISS_THROW_IF_NOT_MSG(args.k > 0, "bfKn3: k must be > 0 for top-k reduction");
    FAISS_THROW_IF_NOT_MSG(args.ps > 0, "bfKn3: patchsize must be > 0");
    FAISS_THROW_IF_NOT_MSG(args.pt > 0, "bfKn3: temporal-patchsize must be > 0");
    FAISS_THROW_IF_NOT_MSG(args.nframes > 0, "bfKn3: num of frames must be > 0");
    FAISS_THROW_IF_NOT_MSG(args.nchnls > 0, "bfKn3: image channels must be > 0");
    FAISS_THROW_IF_NOT_MSG(args.height > 0, "bfKn3: image height must be > 0");
    FAISS_THROW_IF_NOT_MSG(args.width > 0, "bfKn3: image width must be > 0");
    FAISS_THROW_IF_NOT_MSG(
            args.srch_burst, "bfKn3: vectors must be provided (passed null)");
    FAISS_THROW_IF_NOT_MSG(
            args.numQueries > 0, "bfKn3: numQueries must be > 0");
    FAISS_THROW_IF_NOT_MSG(
            args.queries, "bfKn3: queries must be provided (passed null)");
    FAISS_THROW_IF_NOT_MSG(
            args.outDistances,
            "bfKn3: outDistances must be provided (passed null)");
    FAISS_THROW_IF_NOT_MSG(
            args.outIndices || args.k == -1,
            "bfKn3: outIndices must be provided (passed null)");
    static_assert(sizeof(int) == 4, "");
    
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
    auto srch_burst = toDeviceTemporary<T, 4>(
            res,
            device,
            const_cast<T*>(reinterpret_cast<const T*>(args.srch_burst)),
            stream,
            {args.nframes,args.nchnls,args.height,args.width});
    auto tQueries = toDeviceTemporary<int, 2>(
            res,
            device,
            const_cast<int*>(reinterpret_cast<const int*>(args.queries)),
            stream,
            {args.numQueries,3});

    // Output Distances and Inds
    auto tOutDistances = toDeviceTemporary<float, 2>(res,
                                                     device,
                                                     args.outDistances,
                                                     stream,
                                                     {args.numQueries, args.k});
    auto tOutIntIndices = toDeviceTemporary<int, 2>(res,
                                                    device,
                                                    (int*)args.outIndices,
                                                    stream,
                                                    {args.numQueries, args.k});

    // Since we've guaranteed that all arguments are
    // on device, call the implementation

    if (args.fxn_name == Kn3FxnName::KDISTS){

      // Only _Compute_ the Nearest Neighbors (no fill)
      bfKn3OnDevice<T>(res,device,stream,
                       srch_burst,tQueries,
                       args.queriesRowMajor,
                       args.k,args.metric,
                       args.metricArg,tOutDistances,
                       tOutIntIndices,args.ignoreOutDistances);

    }else if (args.fxn_name == Kn3FxnName::KFILL){

      auto patches = toDeviceTemporary<T,6>(res,device,
                     const_cast<T*>(reinterpret_cast<const T*>(args.patches)),
                     stream,
                     {args.numQueries,args.k,args.pt,args.nchnls,args.ps,args.ps});

      // Compute Nearest Neighbors AND Fill
      bfKn3FillOnDevice<T>(res,
                           device,
                           stream,
                           srch_burst,
                           tQueries,
                           args.queriesRowMajor,
                           args.k,
                           args.metric,
                           args.metricArg,
                           tOutDistances,
                           tOutIntIndices,
                           args.ignoreOutDistances);

    }else if (args.fxn_name == Kn3FxnName::PFILLTEST){

      auto patches = toDeviceTemporary<T,6>(res,device,
                     const_cast<T*>(reinterpret_cast<const T*>(args.patches)),
                     stream,
                     {args.numQueries,args.k,args.pt,args.nchnls,args.ps,args.ps});

      // Compute Nearest Neighbors AND Fill
      kn3FillTestPatches<T>(res,device,stream,patches,args.fill_a);

    }else if (args.fxn_name == Kn3FxnName::FILLOUT){
      // Output Distances and Inds
      auto tOutDistances = toDeviceTemporary<float, 2>(res,
                                                       device,
                                                       args.outDistances,
                                                       stream,
                                                       {args.numQueries, args.k});
      auto tOutIntIndices = toDeviceTemporary<int, 2>(res,
                                                      device,
                                                      (int*)args.outIndices,
                                                      stream,
                                                      {args.numQueries, args.k});
      // Fill outputs for computing nearest neighbors
      kn3FillOutMats<T>(res,
                        device,
                        stream,
                        srch_burst,
                        tQueries,
                        args.queriesRowMajor,
                        args.k,
                        args.metric,
                        args.metricArg,
                        tOutDistances,
                        tOutIntIndices,
                        args.ignoreOutDistances,
                        args.fill_a,
                        args.fill_b);

      // // Copy back if necessary
      // fromDevice<int, 2>(tOutIntIndices, (int*)args.outIndices, stream);

      // // Copy distances back if necessary
      // fromDevice<float, 2>(tOutDistances, args.outDistances, stream);

    }else if (args.fxn_name == Kn3FxnName::FILLIN){
      // Fill inputs for computing nearest neighbors
      kn3FillInMats<T>(res,
                       device,
                       stream,
                       srch_burst,
                       tQueries,
                       args.queriesRowMajor,
                       args.k,
                       args.metric,
                       args.metricArg,
                       args.ignoreOutDistances,
                       args.fill_a,
                       args.fill_b);
    }else{
      FAISS_THROW_MSG("unknown function name.");
    }

}

void bfKn3(GpuResourcesProvider* res, const GpuKn3DistanceParams& args) {

    if (args.vectorType == DistanceDataType::F32) {
        bfKn3Convert<float>(res, args);
    } else if (args.vectorType == DistanceDataType::F16) {
        bfKn3Convert<half>(res, args);
    } else {
        FAISS_THROW_MSG("unknown vectorType");
    }
}

}
}
