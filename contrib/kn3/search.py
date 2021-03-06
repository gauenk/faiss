

# -- faiss --
import faiss
from faiss.contrib.torch_utils import using_stream

# -- torch --
import torch as th

# -- local --
from .search_args import get_search_args
from .fill_args import get_fill_args
from .clock import Timer

def run_search(srch_img,queryStart,batchQueries,flows,sigma,srch_args,
               bufs,clock=None,pfill=False):
    """

    Execute the burst nearest neighbors search

    """

    # -- select [search only] or [search + fill] --
    if pfill: fxn_name = faiss.Kn3FxnName_KPATCHES
    else: fxn_name = faiss.Kn3FxnName_KDIST

    # -- faiss args --
    device = srch_img.device
    args,bufs = get_search_args(srch_img,None,queryStart,batchQueries,
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

def run_sim(srch_img,src_img,dst_img,ref,
            queryStart,batchQueries,flows,sigma,srch_args,
            bufs,clock=None,pfill=False):
    """

    Search the "srch_img" to find indices
    to fill the "dst_img" with the "src_img".

    E.g.
    -> "srch_img" = patch matching estimate
    -> "dst_img" = image to be constructed as "similar"
    -> "src_img" = pixels to be copied to "dst_img"

    """

    # -- select [search only] or [search + fill] --
    if pfill: fxn_name = faiss.Kn3FxnName_KPATCHES
    else: fxn_name = faiss.Kn3FxnName_KDIST

    # -- faiss args --
    device = srch_img.device
    args,bufs = get_search_args(srch_img,None,queryStart,batchQueries,
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


def run_fill(img,patches,queryStart,srch_args,ftype,inds=None,clock=None):
    """

    Fill the "img" with the patches as the "query" locations

    Patches -> fill -> Image

    """

    # -- faiss args --
    device = img.device
    args,xb,D,I = get_fill_args(img,patches,queryStart,ftype,inds,srch_args)

    # -- setup stream --
    res = faiss.StandardGpuResources()

    # -- exec --
    if not(clock is None): clock.tic()
    with using_stream(res):
        faiss.bfKn3(res, args)
    # faiss.bfKn3(res, args)
    if not(clock is None):
        th.cuda.synchronize()
        clock.toc()
