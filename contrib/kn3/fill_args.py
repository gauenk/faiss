
# -- misc --
from easydict import EasyDict as edict

# -- faiss --
import faiss
from faiss.contrib.torch_utils import using_stream

# -- torch --
import torch as th

# -- local --
from .utils import get_buf,check_contiguous,get_contiguous,\
    get_float_ptr,get_int_ptr,optional,get_patches,get_flow

def get_fill_args(xb,patches,queryStart,args):

    # -- unpack args --
    assert not(patches is None)
    pshape = patches.shape
    nq,k,pt,c,ps,ps = pshape
    numQueries = nq
    ws = optional(args,'ws',27)
    wf = optional(args,'wf',6)
    wb = optional(args,'wb',6)
    bmax = optional(args,'bmax',255.)
    queryStride = optional(args,'queryStride',1)
    # patches = patches[:,:1] # DONT do this; requires copy to make contiguous

    # -- setup --
    device = xb.device
    tf32,ti32 = th.float32,th.int32

    # -- shape --
    t,c,h,w = xb.shape

    # -- check sizes --
    t,c,h,w = xb.size()

    # -- alloc/format patch --
    fxn_name = faiss.Kn3FxnName_PFILL
    patches = get_patches(patches,pshape,device,tf32)

    # --- faiss info --
    xb = get_contiguous(xb)
    xb_ptr,xb_type = get_float_ptr(xb)
    D = th.zeros((1,1),dtype=tf32,device=device)
    I = th.zeros((1,1),dtype=ti32,device=device)
    I_ptr,I_type = get_int_ptr(I)
    D_ptr,D_type = get_float_ptr(D)
    patches_ptr,_ = get_float_ptr(patches)
    # fflow_ptr,_ = get_float_ptr(fflow)
    # bflow_ptr,_ = get_float_ptr(bflow)

    # -- type checks --
    assert xb.dtype == tf32
    assert patches.dtype == tf32

    # -- create args --
    args = faiss.GpuKn3DistanceParams()
    args.fxn_name = fxn_name

    args.k = k
    args.ps = ps
    args.pt = pt
    args.ws = ws
    args.wf = wf
    args.wb = wb
    args.bmax = bmax

    args.nframes = t
    args.nchnls = c
    args.height = h
    args.width = w

    # args.fflow = fflow_ptr
    # args.bflow = bflow_ptr

    args.vectorType = xb_type
    args.srch_burst = xb_ptr
    args.patches = patches_ptr
    args.queryStart = queryStart
    args.queryStride = queryStride
    args.numQueries = nq
    args.outDistances = D_ptr
    args.outIndices = I_ptr
    # args.outIndicesType = I_type

    return args,xb
