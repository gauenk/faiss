

import torch as th
from einops import rearrange

import faiss
from faiss.contrib.torch_utils import using_stream

from .utils import get_float_ptr

def compute_cov(patches):
    patches = rearrange(patches,'b k t c h w -> b c k (t h w)')
    device = patches.device

    # -- no grad --
    with th.no_grad():

        # -- center patches
        patches /= 255.
        center = patches.mean(dim=2,keepdim=True)
        cpatches = patches - center

        # -- flat batch & color --
        cpatches = rearrange(cpatches,'b c k p -> (b c) k p')

        # -- compute covariance matrix --
        bsize,num,pdim = cpatches.shape
        covMat = th.matmul(cpatches.transpose(2,1),cpatches)/num

    return covMat

def patches2cov(patches):
    # -- unpack patches --
    tf32 = th.float32
    b,k,t,c,h,w = patches.shape
    # -- empty eigen-stuff --
    covMat = compute_cov(patches)
    return covMat

def tiny_eigh(covMat):

    # -- compute covMat --
    num,pdim,pdim = covMat.shape
    tf64 = th.float64
    tf32 = th.float32
    device = covMat.device
    tfdt = tf32

    # -- type --
    covMat = covMat.contiguous().type(tfdt)

    # -- create shells --
    eigVals = th.zeros((num,pdim),dtype=tfdt,device=device)
    eigVecs = th.zeros((num,pdim,pdim),dtype=tfdt,device=device)

    # -- get pointers --
    covMat_ptr,covMat_dtype = get_float_ptr(covMat)
    eigVals_ptr,_ = get_float_ptr(eigVals)
    eigVecs_ptr,_ = get_float_ptr(eigVecs)

    # -- faiss args --
    args = faiss.GpuTinyEighParams()
    args.num = num
    args.dim = pdim
    args.rank = -1
    args.covMat = covMat_ptr
    args.vectorType = covMat_dtype
    args.eigVals = eigVals_ptr
    args.eigVecs = eigVecs_ptr

    # -- faiss stream --
    res = faiss.StandardGpuResources()

    # -- compute eigh for cov --
    with using_stream(res):
        faiss.tinyEigh(res, args)

    # -- final formatting --
    eigVecs = covMat
    eigVecs = eigVecs.transpose(2,1)
    eigVals = th.flip(eigVals,dims=(1,))
    eigVecs = th.flip(eigVecs,dims=(2,))

    return eigVals,eigVecs
