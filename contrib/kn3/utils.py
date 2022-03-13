
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
        D = torch.empty(nq, k, device=device, dtype=dtype)
    else:
        assert D.shape == (nq, k)
        # interface takes void*, we need to check this
        assert (D.dtype == dtype)
    return D

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
    if xb.dtype == torch.float32:
        xb_type = faiss.DistanceDataType_F32
        xb_ptr = swig_ptr_from_FloatTensor(xb)
    elif xb.dtype == torch.float16:
        xb_type = faiss.DistanceDataType_F16
        xb_ptr = swig_ptr_from_HalfTensor(xb)
    else:
        raise TypeError('xb must be f32 or f16')
    return xb_ptr,xb_type

def get_int_ptr(I):
    if I.dtype == torch.int64:
        I_type = faiss.IndicesDataType_I64
        I_ptr = swig_ptr_from_IndicesTensor(I)
    elif I.dtype == I.dtype == torch.int32:
        I_type = faiss.IndicesDataType_I32
        I_ptr = swig_ptr_from_IntTensor(I)
    else:
        raise TypeError('I must be i64 or i32')
    return I_ptr,I_type

# ------------------------------
#
#         Get 3D Inds
#
# ------------------------------

def get_3d_inds(inds,c,h,w):

    # -- unpack --
    chw,hw = c*h*w,h*w
    bsize,num = inds.shape
    device = inds.device

    # -- shortcuts --
    tdiv = th.div
    tmod = th.remainder

    # -- init --
    aug_inds = th.zeros((3,bsize,num),dtype=th.int64)
    aug_inds = aug_inds.to(inds.device)

    # -- fill --
    aug_inds[0,...] = tdiv(inds,chw,rounding_mode='floor') # inds // chw
    aug_inds[1,...] = tdiv(tmod(inds,hw),w,rounding_mode='floor') # (inds % hw) // w
    aug_inds[2,...] = tmod(inds,w)
    aug_inds = rearrange(aug_inds,'three b n -> (b n) three')

    return aug_inds
