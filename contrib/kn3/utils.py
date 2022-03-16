
# -- linalg --
import torch as th
from einops import rearrange,repeat

# -- faiss-torch --
import faiss
from faiss.contrib.torch_utils import *

# ------------------------------
#
#      FAISS Format Funcs
#
# ------------------------------

def get_buf(D,nq,k,device,dtype):
    if D is None:
        D = th.empty(nq, k, device=device, dtype=dtype)
    else:
        assert D.shape == (nq, k)
        # interface takes void*, we need to check this
        assert (D.dtype == dtype)
    return D

def get_patches(patches,pshape,device,dtype):
    if patches is None:
        patches = th.empty(pshape, device=device, dtype=dtype)
    else:
        assert patches.shape == pshape
        # interface takes void*, we need to check this
        assert (patches.dtype == dtype)
    return patches

def get_contiguous(tensor):
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    return tensor

def check_contiguous(xq):
    if xq.is_contiguous():
        xq_row_major = True
    elif xq.t().is_contiguous():
        xq = xq.t()
        xq_row_major = False
    else:
        raise TypeError('matrix should be row or column-major')
    return xq_row_major

def get_float_ptr(xb):
    if xb.dtype == th.float32:
        xb_type = faiss.DistanceDataType_F32
        xb_ptr = swig_ptr_from_FloatTensor(xb)
    elif xb.dtype == th.float16:
        xb_type = faiss.DistanceDataType_F16
        xb_ptr = swig_ptr_from_HalfTensor(xb)
    else:
        raise TypeError('xb must be f32 or f16')
    return xb_ptr,xb_type

def get_int_ptr(I):
    if I.dtype == th.int64:
        I_type = faiss.IndicesDataType_I64
        I_ptr = swig_ptr_from_IndicesTensor(I)
    elif I.dtype == I.dtype == th.int32:
        I_type = faiss.IndicesDataType_I32
        I_ptr = swig_ptr_from_IntTensor(I)
    else:
        raise TypeError('I must be i64 or i32')
    return I_ptr,I_type

def get_flow(flow,shape,device,flow_alloced=None):
    tf32 = th.float32
    if flow is None and not(flow_alloced is None):
        return flow_alloced
    elif flow is None:
        flow = th.zeros(shape,dtype=tf32,device=device)
    return flow

# ------------------------------
#
#         Get 3D Inds
#
# ------------------------------

def get_3d_inds(inds,h,w):

    # -- unpack --
    hw = h*w # no "chw" in this code-base; its silly.
    bsize,num = inds.shape
    device = inds.device

    # -- shortcuts --
    tdiv = th.div
    tmod = th.remainder

    # -- init --
    aug_inds = th.zeros((3,bsize,num),dtype=th.int64)
    aug_inds = aug_inds.to(inds.device)

    # -- fill --
    aug_inds[0,...] = tdiv(inds,hw,rounding_mode='floor') # inds // chw
    aug_inds[1,...] = tdiv(tmod(inds,hw),w,rounding_mode='floor') # (inds % hw) // w
    aug_inds[2,...] = tmod(inds,w)
    aug_inds = rearrange(aug_inds,'three b n -> (b n) three')

    return aug_inds

# ------------------------------
#
#      Misc
#
# ------------------------------

def optional(pydict,key,default):
    if pydict is None: return default
    elif not(key in pydict): return default
    else: return pydict[key]
