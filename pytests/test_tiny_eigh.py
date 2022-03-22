
# -- python --
import cv2,tqdm,copy
import numpy as np
import unittest
import tempfile
import sys
from einops import rearrange
import shutil
from pathlib import Path
from easydict import EasyDict as edict

# -- vision --
from PIL import Image

# -- linalg --
import torch as th
import numpy as np

# -- package helper imports --
import vnlb
from faiss.contrib import kn3
from faiss.contrib import testing

# -- check if reordered --
from scipy import optimize
SAVE_DIR = Path("./output/tests/")

#
#
# -- Primary Testing Class --
#
#
PYTEST_OUTPUT = Path("./pytests/output/")

class TestTinyEigh(unittest.TestCase):

    #
    # -- Load Data --
    #

    def do_load_data(self,dname,sigma,device="cuda:0"):

        #  -- Read Data (Image & VNLB-C++ Results) --
        clean = testing.load_dataset(dname)
        clean = clean[:15,:,:32,:32].to(device).type(th.float32)
        # clean = th.zeros((15,3,32,32)).to(device).type(th.float32)
        clean = clean * 1.0
        noisy = clean + sigma * th.normal(0,1,size=clean.shape,device=device)
        return clean,noisy

    def do_load_flow(self,comp_flow,burst,sigma,device):
        if comp_flow:
            #  -- TV-L1 Optical Flow --
            flow_params = {"nproc":0,"tau":0.25,"lambda":0.2,"theta":0.3,
                           "nscales":100,"fscale":1,"zfactor":0.5,"nwarps":5,
                           "epsilon":0.01,"verbose":False,"testing":False,'bw':True}
            fflow,bflow = vnlb.swig.runPyFlow(burst,sigma,flow_params)
        else:
            #  -- Empty shells --
            t,c,h,w = burst.shape
            tf32,tfl = th.float32,th.long
            fflow = th.zeros(t,2,h,w,dtype=tf32,device=device)
            bflow = fflow.clone()

        # -- pack --
        flows = edict()
        flows.fflow = fflow
        flows.bflow = bflow
        return flows

    def init_topk_shells(self,bsize,k,pt,c,ps,device):
        tf32,ti32 = th.float32,th.int32
        vals = float("inf") * th.ones((bsize,k),dtype=tf32,device=device)
        inds = -th.ones((bsize,k),dtype=ti32,device=device)
        patches = -th.ones((bsize,k,pt,c,ps,ps),dtype=tf32,device=device)
        return vals,inds,patches

    def exec_kn3_search(self,K,clean,flows,sigma,args,bufs):

        # -- unpack --
        device = clean.device
        shape = clean.shape
        t,c,h,w = shape

        # -- prepare kn3 search  --
        index,BSIZE = 0,t*h*w
        args.k = K
        numQueries = (BSIZE-1) // args.queryStride + 1

        # -- search --
        kn3.run_search(clean,0,numQueries,flows,sigma,args,bufs,pfill=True)
        th.cuda.synchronize()

        # -- unpack --
        kn3_vals = bufs.dists
        kn3_inds = bufs.inds
        kn3_patches = bufs.patches

        return kn3_inds,kn3_patches

    def exec_tiny_eigh(self,in_covMat):
        # covMat = kn3.patches2cov(patches)
        clock = kn3.Timer()
        clock.tic()
        covMat = in_covMat.clone()
        eigVals,eigVecs = kn3.tiny_eigh(covMat)
        clock.toc()
        dtime = clock.diff
        return eigVals,eigVecs,dtime

    def exec_vnlb_eigh(self,in_covMat):
        # covMat = vnlb.patches2cov(patches)
        clock = kn3.Timer()
        clock.tic()
        covMat = in_covMat.clone()
        eigVals,eigVecs = vnlb.cov2eigs(covMat)
        clock.toc()
        dtime = clock.diff
        return eigVals,eigVecs,dtime

    def rec_mats(self,eigVals,eigVecs):
        # -- compute covMat --
        L = eigVals.float()
        Q = eigVecs.float()
        A_r = Q @ th.diag_embed(L) @ Q.transpose(2,1)

        # -- compute ID --
        num,dim,dim = eigVecs.shape
        I_r = Q @ Q.transpose(2,1)

        # -- compute ID ref --
        I = th.eye(dim)[None,:,:].repeat(num,1,1).to(eigVecs.device).float()
        return A_r,I_r,I


    #
    # -- [Exec] Sim Search --
    #

    def run_comparison(self,noisy,clean,sigma,flows,args):

        # -- fixed testing params --
        K = 15
        BSIZE = 50
        NBATCHES = 3
        shape = noisy.shape
        device = noisy.device

        # -- create empty bufs --
        bufs = edict()
        bufs.patches = None
        bufs.dists = None
        bufs.inds = None
        args['queryStride'] = 7
        args['stype'] = "faiss"

        # -- exec over batches --
        for index in range(NBATCHES):

            # -- get new image --
            noise = sigma * th.randn_like(clean)
            noisy = (clean + noise).type(th.float32).contiguous()

            # -- search using faiss code --
            _,patches = self.exec_kn3_search(K,noisy,flows,sigma,args,bufs)
            covMat = vnlb.patches2cov(patches/255.)
            # print("covMat[min,max]: ",covMat.min().item(),covMat.max().item())

            # -- kn3 eig  --
            kn3_eigVals,kn3_eigVecs,kn3_time = self.exec_tiny_eigh(covMat)

            # -- vnlb eig  --
            vnlb_eigVals,vnlb_eigVecs,vnlb_time = self.exec_vnlb_eigh(covMat)

            # -- cpu --
            kn3_eigVals_np = kn3_eigVals.cpu().float().numpy()
            kn3_eigVecs_np = kn3_eigVecs.cpu().float().abs().numpy()
            vnlb_eigVals_np = vnlb_eigVals.cpu().float().numpy()
            vnlb_eigVecs_np = vnlb_eigVecs.cpu().float().abs().numpy()

            # -- valid --
            assert not np.all(vnlb_eigVals_np<1e-8)
            assert not np.all(vnlb_eigVecs_np<1e-8)

            # -- test vals --
            np.testing.assert_array_almost_equal(kn3_eigVals_np,vnlb_eigVals_np)

            # -- test [kn3] vecs --
            A_r,I_r,I = self.rec_mats(kn3_eigVals,kn3_eigVecs)
            assert th.dist(A_r,covMat).item() < 1e-4
            assert th.dist(I_r,I).item() < 1e-2

            # -- test [vnlb] vecs --
            A_r,I_r,I = self.rec_mats(vnlb_eigVals,vnlb_eigVecs)
            assert th.dist(A_r,covMat).item() < 1e-4
            assert th.dist(I_r,I).item() < 1e-2
            print("kn3_time: ",kn3_time)
            print("vnlb_time: ",vnlb_time)


    def run_single_test(self,dname,sigma,comp_flow,pyargs):
        noisy,clean = self.do_load_data(dname,sigma)
        flows = self.do_load_flow(False,clean,sigma,noisy.device)
        self.run_comparison(noisy,clean,sigma,flows,pyargs)

    def test_sim_search(self):

        # -- init save path --
        np.random.seed(123)
        save_dir = SAVE_DIR
        if not save_dir.exists():
            save_dir.mkdir(parents=True)

        # -- test 1 --
        sigma = 10.
        dname = "text_tourbus_64"
        comp_flow = False
        args = edict({'ps':5,'pt':1,'c':3})
        self.run_single_test(dname,sigma,comp_flow,args)
