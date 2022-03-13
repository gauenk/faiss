

# -- faiss --
import faiss
from faiss.contrib.torch_utils import using_stream

# -- torch --
import torch as th

# -- local --
from .utils import get_buf,check_contiguous,get_float_ptr,get_int_ptr

def run_fill_output(val_dists, val_inds):

    # -- create resources --
    device = "cuda:0"
    res = faiss.StandardGpuResources()


    # -- create dummy data --
    tf32,ti32 = th.float32,th.int32
    xb = th.zeros((10,10,10,10),device=device,dtype=tf32)
    xq = th.zeros((10,3),device=device,dtype=ti32)

    # -- init params --
    ps,pt = 7,1
    k = 3
    D,I = None,None

    # -- check sizes --
    t,c,h,w = xb.size()
    nq, d = xq.size()

    # -- alloc/format bufs --
    D = get_buf(D,nq,k,device,tf32)
    I = get_buf(I,nq,k,device,ti32)

    # --- faiss info --
    xb_row_major = check_contiguous(xb)
    xq_row_major = check_contiguous(xq)
    assert xq_row_major is True,"Must be row major."
    xb_ptr,xb_type = get_float_ptr(xb)
    xq_ptr,xq_type = get_int_ptr(xq)
    I_ptr,I_type = get_int_ptr(I)
    D_ptr,D_type = get_float_ptr(D)

    # -- create args --
    args = faiss.GpuKn3DistanceParams()
    args.fxn_name = faiss.Kn3FxnName_FILLOUT
    args.k = k
    args.ps = ps
    args.pt = pt
    args.nframes = t
    args.nchnls = c
    args.height = h
    args.width = w
    args.srch_burst = xb_ptr
    args.vectorType = xb_type
    args.queries = xq_ptr
    args.queriesRowMajor = xq_row_major
    args.queryType = xq_type
    args.numQueries = nq
    args.outDistances = D_ptr
    args.outIndices = I_ptr
    args.outIndicesType = I_type
    args.fill_a = val_dists
    args.fill_b = val_inds

    # -- exec --
    with using_stream(res):
        faiss.bfKn3(res, args)

    return D,I

def run_fill_input(val_burst, val_query, bshape, qsize):

    # -- create resources --
    device = "cuda:0"
    res = faiss.StandardGpuResources()


    # -- create dummy data --
    tf32,ti32 = th.float32,th.int32
    xb = th.zeros(bshape,device=device,dtype=tf32)
    xq = th.zeros((qsize,3),device=device,dtype=ti32)

    # -- init params --
    ps,pt = 7,1
    k = 3
    D,I = None,None

    # -- check sizes --
    t,c,h,w = xb.size()
    nq, d = xq.size()

    # -- alloc/format bufs --
    D = get_buf(D,nq,k,device,tf32)
    I = get_buf(I,nq,k,device,ti32)

    # --- faiss info --
    xb_row_major = check_contiguous(xb)
    xq_row_major = check_contiguous(xq)
    assert xq_row_major is True,"Must be row major."
    xb_ptr,xb_type = get_float_ptr(xb)
    xq_ptr,xq_type = get_int_ptr(xq)
    I_ptr,I_type = get_int_ptr(I)
    D_ptr,D_type = get_float_ptr(D)

    # -- create args --
    args = faiss.GpuKn3DistanceParams()
    args.fxn_name = faiss.Kn3FxnName_FILLIN
    args.k = k
    args.ps = ps
    args.pt = pt
    args.nframes = t
    args.nchnls = c
    args.height = h
    args.width = w
    args.dims = d
    args.srch_burst = xb_ptr
    args.vectorType = xb_type
    args.queries = xq_ptr
    args.queriesRowMajor = xq_row_major
    args.queryType = xq_type
    args.numQueries = nq
    args.outDistances = D_ptr
    args.outIndices = I_ptr
    args.outIndicesType = I_type
    args.fill_a = val_burst
    args.fill_b = val_query

    # -- exec --
    with using_stream(res):
        faiss.bfKn3(res, args)

    return xb,xq

def run_fill_patches(val, pshape, t):

    # -- create resources --
    device = "cuda:0"
    res = faiss.StandardGpuResources()

    # -- create dummy data --
    qsize,k,pt,c,ps,ps = pshape
    bshape = (t,c,32,32)
    tf32,ti32 = th.float32,th.int32
    xb = th.zeros(bshape,device=device,dtype=tf32)
    xq = th.zeros((qsize,3),device=device,dtype=ti32)
    patches = th.zeros(pshape,device=device,dtype=tf32)

    # -- init params --
    D,I = None,None

    # -- check sizes --
    t,c,h,w = xb.size()
    nq, d = xq.size()

    # -- alloc/format bufs --
    D = get_buf(D,nq,k,device,tf32)
    I = get_buf(I,nq,k,device,ti32)

    # --- faiss info --
    xb_row_major = check_contiguous(xb)
    xq_row_major = check_contiguous(xq)
    assert xq_row_major is True,"Must be row major."
    xb_ptr,xb_type = get_float_ptr(xb)
    xq_ptr,xq_type = get_int_ptr(xq)
    I_ptr,I_type = get_int_ptr(I)
    D_ptr,D_type = get_float_ptr(D)
    patches_ptr,_ = get_float_ptr(patches)

    # -- create args --
    args = faiss.GpuKn3DistanceParams()
    args.fxn_name = faiss.Kn3FxnName_PFILLTEST
    args.k = k
    args.ps = ps
    args.pt = pt
    args.nframes = t
    args.nchnls = c
    args.height = h
    args.width = w
    args.dims = d
    args.srch_burst = xb_ptr
    args.patches = patches_ptr
    args.vectorType = xb_type
    args.queries = xq_ptr
    args.queriesRowMajor = xq_row_major
    args.queryType = xq_type
    args.numQueries = nq
    args.outDistances = D_ptr
    args.outIndices = I_ptr
    args.outIndicesType = I_type
    args.fill_a = val

    # -- exec --
    with using_stream(res):
        faiss.bfKn3(res, args)

    return patches


