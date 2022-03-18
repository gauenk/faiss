

# -- faiss --
import faiss
from faiss.contrib.torch_utils import using_stream

# -- torch --
import torch as th

# -- local --
from .search_args import get_search_args
from .fill_args import get_fill_args
from .clock import Timer

def run_search(srch_img,queryStart,numElems,flows,sigma,srch_args,
               bufs,clock=None,pfill=False):
    """

    Execute the burst nearest neighbors search

    """

    # -- select [search only] or [search + fill] --
    if pfill: fxn_name = faiss.Kn3FxnName_KPATCHES
    else: fxn_name = faiss.Kn3FxnName_KDIST

    # -- faiss args --
    device = srch_img.device
    args,bufs = get_search_args(srch_img,None,queryStart,numElems,
                                srch_args,flows,bufs,fxn_name=fxn_name)
    # -- setup stream --
    res = faiss.StandardGpuResources()

    # -- exec --
    if not(clock is None): clock.tic()
    with using_stream(res):
        faiss.bfKn3(res, args)
    if not(clock is None):
        th.cuda.synchronize()
        clock.toc()
    return bufs


def run_fill(fill_img,patches,queryStart,srch_args,clock=None):
    """

    Fill the "fill_img" with the patches as the "query" locations

    """

    # -- faiss args --
    device = fill_img.device
    args,xb = get_fill_args(fill_img,patches,queryStart,srch_args)

    # -- setup stream --
    res = faiss.StandardGpuResources()
    pytorch_stream = th.cuda.current_stream()
    cuda_stream_s = faiss.cast_integer_to_cudastream_t(pytorch_stream.cuda_stream)
    res.setDefaultStream(th.cuda.current_device(), cuda_stream_s)
    th.cuda.synchronize()

    # -- exec --
    if not(clock is None): clock.tic()
    faiss.bfKn3(res, args)
    if not(clock is None):
        th.cuda.synchronize()
        clock.toc()
