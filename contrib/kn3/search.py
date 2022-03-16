

# -- faiss --
import faiss
from faiss.contrib.torch_utils import using_stream

# -- torch --
import torch as th

# -- local --
from .utils import get_buf,check_contiguous,get_contiguous,\
    get_float_ptr,get_int_ptr,optional,get_patches,get_flow
from .clock import Timer

def get_faiss_args(xb,xb_fill,queryStart,numQueries,args,flows,bufs,
                   fxn_name=faiss.Kn3FxnName_KDIST):

    # -- unpack args --
    k = optional(args,'k',-1)
    ps = optional(args,'ps',7)
    pt = optional(args,'pt',1)
    ws = optional(args,'ws',27)
    wf = optional(args,'wf',6)
    wb = optional(args,'wb',6)
    queryStride = optional(args,'queryStride',1)
    nq = numQueries

    # -- unpack flows --
    fflow = optional(flows,'fflow',None)
    bflow = optional(flows,'bflow',None)

    # -- unpack bufs --
    patches = optional(bufs,'patches',None)
    D = optional(bufs,'dists',None)
    I = optional(bufs,'inds',None)

    # -- setup --
    device = xb.device
    tf32,ti32 = th.float32,th.int32

    # -- shape --
    t,c,h,w = xb.shape

    # -- check sizes --
    t,c,h,w = xb.size()

    # -- alloc/format patch --
    kdist_b = fxn_name == faiss.Kn3FxnName_KDIST
    if kdist_b: pshape = (1,1,1,1,1,1) # no space needed
    else: pshape = (nq,k,pt,c,ps,ps)
    patches = get_patches(patches,pshape,device,tf32)

    # -- shape checking --
    if k == -1: k = D.shape[1]
    elif D is None: assert k != -1

    # -- alloc/format flows --
    fshape = (t,2,h,w)
    fflow = get_flow(fflow,fshape,device)
    bflow = get_flow(bflow,fshape,device,fflow)

    # -- alloc/format bufs --
    D = get_buf(D,nq,k,device,tf32)
    I = get_buf(I,nq,k,device,ti32)
    if bufs.dists is None:
        bufs.dists = D
    if bufs.inds is None:
        bufs.inds = I

    # --- faiss info --
    xb = get_contiguous(xb)
    xb_ptr,xb_type = get_float_ptr(xb)
    if xb_fill is None: xbfill_ptr = xb_ptr
    else:
        xbfill = get_contiguous(xb_fill)
        xbfill_ptr,_ = get_float_ptr(xb_fill)
    I_ptr,I_type = get_int_ptr(I)
    D_ptr,D_type = get_float_ptr(D)
    patches_ptr,_ = get_float_ptr(patches)
    fflow_ptr,_ = get_float_ptr(fflow)
    bflow_ptr,_ = get_float_ptr(bflow)

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

    args.nframes = t
    args.nchnls = c
    args.height = h
    args.width = w

    args.fflow = fflow_ptr
    args.bflow = bflow_ptr

    args.srch_burst = xb_ptr
    args.fill_burst = xbfill_ptr
    args.patches = patches_ptr
    args.vectorType = xb_type
    args.queryStart = queryStart
    args.queryStride = queryStride
    args.numQueries = nq
    args.outDistances = D_ptr
    args.outIndices = I_ptr
    args.outIndicesType = I_type

    return args

def run_search(srch_img,queryStart,numQueries,flows,sigma,srch_args,bufs):
    """

    Execute the burst nearest neighbors search

    """


    # -- faiss args --
    device = srch_img.device
    args = get_faiss_args(srch_img,None,queryStart,numQueries,
                          srch_args,flows,bufs,fxn_name=faiss.Kn3FxnName_KDIST)
    # -- setup stream --
    res = faiss.StandardGpuResources()
    pytorch_stream = th.cuda.current_stream()
    cuda_stream_s = faiss.cast_integer_to_cudastream_t(pytorch_stream.cuda_stream)
    res.setDefaultStream(th.cuda.current_device(), cuda_stream_s)

    # -- exec --
    clock = Timer()
    print("HI")
    clock.tic()
    faiss.bfKn3(res, args)
    clock.toc()
    print("by")
    print(clock)
