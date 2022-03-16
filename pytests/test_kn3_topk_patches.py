
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
        clean = th.zeros((15,3,32,32)).to(device)
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

    def get_search_inds(self,index,bsize,shape,device):
        t,c,h,w  = shape
        start = index * bsize
        stop = ( index + 1 ) * bsize
        ti32 = th.int32
        srch_inds = th.arange(start,stop,dtype=ti32,device=device)[:,None]
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

        # -- unpack --
        device = clean.device
        shape = clean.shape
        t,c,h,w = shape

        # -- get search inds --
        index,BSIZE = 0,t*h*w
        srch_inds = self.get_search_inds(index,BSIZE,shape,device)
        srch_inds = srch_inds.type(th.int32)

        # -- get return shells --
        pt,c,ps = args.pt,args.c,args.ps
        vpss_vals,vpss_inds,vpss_patches = self.init_topk_shells(BSIZE,K,pt,c,ps,device)

        # -- search using numba code --
        vpss.exec_sim_search_burst(clean,srch_inds,vpss_vals,
                                   vpss_inds,flows,sigma,args)

        # -- fill patches --
        vpss.get_patches_burst(clean,vpss_inds,ps,pt=pt,patches=vpss_patches)

        # -- return --
        th.cuda.synchronize()
        return vpss_patches

    def exec_kn3_search(self,K,clean,flows,sigma,args,bufs):

        # -- unpack --
        device = clean.device
        shape = clean.shape
        t,c,h,w = shape

        # -- get search inds --
        index,BSIZE = 0,t*h*w

        # -- get return shells --
        pt,c,ps = args.pt,args.c,args.ps
        kn3_vals,kn3_inds,kn3_patches = self.init_topk_shells(BSIZE,K,pt,c,ps,device)

        # -- unpack --
        bufs.dists = kn3_vals
        bufs.inds = kn3_inds
        bufs.patches = kn3_patches

        # -- search --
        kn3.run_search(clean/255.,0,BSIZE,flows,sigma/255.,args,bufs,pfill=True)
        th.cuda.synchronize()

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

        # -- create empty bufs --
        bufs = edict()
        bufs.patches = None
        bufs.dists = None
        bufs.inds = None
        clean = 255.*th.rand_like(clean).type(th.float32)
        noisy = 255.*th.rand_like(clean).type(th.float32)
        args['stype'] = "faiss"

        # -- exec over batches --
        for index in range(NBATCHES):

            # -- search using python code --
            vpss_patches = self.exec_vpss_search(K,clean,flows,sigma,args)

            # -- search using faiss code --
            kn3_patches = self.exec_kn3_search(K,clean,flows,sigma,args,bufs)

            # -- to numpy --
            vpss_patches = vpss_patches.cpu().numpy()
            kn3_patches = kn3_patches.cpu().numpy()

            # -- allow for swapping of "close" values --
            np.testing.assert_array_equal(kn3_patches,vpss_patches)
            # np.testing.assert_array_almost_equal(kn3_vals,vpss_vals)

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
