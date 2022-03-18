
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

# -- linalg --
import torch as th
import numpy as np

# -- package helper imports --
import vpss
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

class TestTopKPatches(unittest.TestCase):

    #
    # -- Load Data --
    #

    def do_load_data(self,dname,sigma,device="cuda:0"):

        #  -- Read Data (Image & VNLB-C++ Results) --
        clean = testing.load_dataset(dname).to(device)[:5]
        clean = th.zeros((15,3,32,32)).to(device).type(th.float32)
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

    def get_search_inds(self,index,bsize,stride,shape,device):
        t,c,h,w  = shape
        start = index * bsize
        stop = ( index + 1 ) * bsize
        ti32 = th.int32
        srch_inds = th.arange(start,stop,stride,dtype=ti32,device=device)[:,None]
        srch_inds = kn3.get_3d_inds(srch_inds,h,w)
        srch_inds = srch_inds.contiguous()
        return srch_inds

    def init_topk_shells(self,bsize,k,pt,c,ps,device):
        tf32,ti32 = th.float32,th.int32
        vals = float("inf") * th.ones((bsize,k),dtype=tf32,device=device)
        inds = -th.ones((bsize,k),dtype=ti32,device=device)
        patches = -th.ones((bsize,k,pt,c,ps,ps),dtype=tf32,device=device)
        return vals,inds,patches

    def exec_vpss_search(self,K,clean,flows,sigma,args):
        vpss_mode = args['vpss_mode']
        if vpss_mode == "exh":
            return self.exec_vpss_search_exh(K,clean,flows,sigma,args)
        elif vpss_mode == "vnlb":
            return self.exec_vpss_search_vnlb(K,clean,flows,sigma,args)
        else:
            raise ValueError(f"Uknown vpss_mode [{vpss_mode}]")

    def exec_vpss_search_vnlb(self,K,clean,flows,sigma,args):

        # -- vnlb --
        bufs = vnlb.global_search_default(clean,sigma,None,args.ps,K,pfill=True)
        vpss_patches = bufs.patches

        # -- return --
        th.cuda.synchronize()

        # -- weight floating-point issue --
        vpss_patches = vpss_patches.type(th.float32)

        return vpss_patches

    def exec_vpss_search_exh(self,K,clean,flows,sigma,args):

        # -- unpack --
        device = clean.device
        shape = clean.shape
        t,c,h,w = shape

        # -- get search inds --
        index,BSIZE,stride = 0,t*h*w,args.bstride
        srch_inds = self.get_search_inds(index,BSIZE,stride,shape,device)
        srch_inds = srch_inds.type(th.int32)

        # -- get return shells --
        numQueries = ((BSIZE - 1)//args.bstride)+1
        nq,pt,c,ps = numQueries,args.pt,args.c,args.ps
        vpss_vals,vpss_inds,vpss_patches = self.init_topk_shells(nq,K,pt,c,ps,device)

        # -- search using numba code --
        vpss.exec_sim_search_burst(clean,srch_inds,vpss_vals,
                                   vpss_inds,flows,sigma,args)

        # -- fill patches --
        vpss.get_patches_burst(clean,vpss_inds,ps,pt=pt,patches=vpss_patches,
                               fill_mode="faiss")

        # -- return --
        th.cuda.synchronize()

        # -- weight floating-point issue --
        vpss_patches = vpss_patches.type(th.float32)

        return vpss_patches

    def exec_kn3_search(self,K,clean,flows,sigma,args,bufs):

        # -- unpack --
        device = clean.device
        shape = clean.shape
        t,c,h,w = shape

        # -- prepare kn3 search  --
        index,BSIZE = 0,t*h*w
        args.k = K

        # -- search --
        kn3.run_search(clean,0,BSIZE,flows,sigma,args,bufs,pfill=True)
        th.cuda.synchronize()

        # -- unpack --
        kn3_vals = bufs.dists
        kn3_inds = bufs.inds
        kn3_patches = bufs.patches

        return kn3_patches


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
        t,c,h,w = shape
        npix = h*w
        bstride = 1

        # -- create empty bufs --
        bufs = edict()
        bufs.patches = None
        bufs.dists = None
        bufs.inds = None
        args['stype'] = "faiss"
        args['vpss_mode'] = "exh"
        args['queryStride'] = 7
        args['bstride'] = args['queryStride']
        # args['vpss_mode'] = "vnlb"

        # -- exec over batches --
        for index in range(NBATCHES):

            # -- get data --
            clean = 255.*th.rand_like(clean).type(th.float32)
            # clean /= 255.
            # clean *= 255.
            noisy = clean.clone()

            # -- search using python code --
            vpss_patches = self.exec_vpss_search(K,clean,flows,sigma,args)

            # -- search using faiss code --
            kn3_patches = self.exec_kn3_search(K,clean,flows,sigma,args,bufs)

            # -- to numpy --
            vpss_patches = vpss_patches.cpu().numpy()
            kn3_patches = kn3_patches.cpu().numpy()

            # -- allow for swapping of "close" values --
            # qindex = 3*32*32+32*16+8
            # # print(kn3_patches[qindex,0,0,0])
            # # print(vpss_patches[qindex,0,0,0])
            # np.testing.assert_array_almost_equal(kn3_patches[qindex,0,0,0],vpss_patches[qindex,0,0,0])
            # qindex = 0
            # np.testing.assert_array_almost_equal(kn3_patches[qindex,0,0,0],vpss_patches[qindex,0,0,0])
            # qindex = -1
            # print(kn3_patches[qindex,0,0,:4,:4])
            # print(vpss_patches[qindex,0,0,:4,:4])
            # np.testing.assert_array_almost_equal(kn3_patches[qindex,0,0,0],vpss_patches[qindex,0,0,0])
            # print("PASSED.")
            neq = np.where(np.abs(kn3_patches - vpss_patches) > 100.)
            if len(neq[0]) > 0:
                print(neq)
                bidx = neq[0][0]
                kidx = neq[1][0]
                print(bidx,kidx)
                bt,bh,bw = (bidx // npix),(bidx // npix)//w,(bidx // npix)%w
                print(bt,bh,bw)
                print(np.stack([kn3_patches[bidx,kidx,0],vpss_patches[bidx,kidx,0]]))
            np.testing.assert_array_equal(kn3_patches,vpss_patches)

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
        sigma = 25.
        dname = "text_tourbus_64"
        comp_flow = False
        args = edict({'ps':7,'pt':1,'c':3})
        self.run_single_test(dname,sigma,comp_flow,args)

        # -- test 2 --
        # sigma = 25.
        # dname = "text_tourbus_64"
        # comp_flow = False
        # pyargs = {'ps_x':3,'ps_t':2}
        # self.run_single_test(dname,sigma,comp_flow,pyargs)
