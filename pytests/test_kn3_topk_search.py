
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

class TestTopKSearch(unittest.TestCase):

    #
    # -- Load Data --
    #

    def do_load_data(self,dname,sigma,device="cuda:0"):

        #  -- Read Data (Image & VNLB-C++ Results) --
        clean = testing.load_dataset(dname).to(device)
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
        srch_inds = th.arange(start,stop,device=device)[:,None]
        srch_inds = kn3.get_3d_inds(srch_inds,c,h,w)
        srch_inds = srch_inds.contiguous()
        return srch_inds

    def init_topk_shells(self,bsize,k,device):
        tf32,ti32 = th.float32,th.int32
        vals = float("inf") * th.ones((bsize,k),dtype=tf32,device=device)
        inds = -th.ones((bsize,k),dtype=ti32,device=device)
        return vals,inds

    #
    # -- [Exec] Sim Search --
    #

    def run_comparison(self,noisy,clean,sigma,flows,args):

        # -- fixed testing params --
        K = 10
        BSIZE = 50
        NBATCHES = 2
        shape = noisy.shape
        device = noisy.device
        print("shape: ",shape)
        print("clean.shape: ",clean.shape)
        print("noisy.shape: ",noisy.shape)


        # -- create empty bufs --
        bufs = edict()
        bufs.patches = None
        bufs.dists = None
        bufs.inds = None
        clean = 255.*th.rand_like(clean).type(th.float32)
        noisy = 255.*th.rand_like(clean).type(th.float32)
        print(clean.max())
        print(noisy.max())

        # -- exec over batches --
        for index in range(NBATCHES):

            # -- get testing inds --
            srch_inds = self.get_search_inds(index,BSIZE,shape,device)
            srch_inds = srch_inds.type(th.int32)

            # -- get return sizes --
            vpss_vals,vpss_inds = self.init_topk_shells(BSIZE,K,device)
            kn3_vals,kn3_inds = self.init_topk_shells(BSIZE,K,device)

            # -- search using numba code --
            vpss.exec_sim_search_burst(clean,srch_inds,vpss_vals,
                                       vpss_inds,flows,sigma,args)

            # -- search using faiss code --
            # vpss.exec_sim_search_burst(clean,srch_inds,kn3_vals,
            #                            kn3_inds,flows,sigma,args)
            print(srch_inds.shape,kn3_vals.shape,kn3_inds.shape)
            bufs.dists = kn3_vals
            bufs.inds = kn3_inds
            print("clean.max().item(): ",clean.max().item())
            kn3.run_search(clean/255.,srch_inds,flows,sigma/255.,args,bufs)
            print(bufs.dists)
            print(bufs.inds)

            # -- to numpy --
            vpss_vals = vpss_vals.cpu().numpy()
            vpss_inds = vpss_inds.cpu().numpy()
            kn3_vals = kn3_vals.cpu().numpy()
            kn3_inds = kn3_inds.cpu().numpy()

            # -- allow for swapping of "close" values --
            np.testing.assert_array_equal(kn3_vals,vpss_vals)
            np.testing.assert_array_equal(kn3_inds,vpss_inds)

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
        args = edict({'ps':7,'pt':1})
        self.run_single_test(dname,sigma,comp_flow,args)

        # -- test 2 --
        # sigma = 25.
        # dname = "text_tourbus_64"
        # comp_flow = False
        # pyargs = {'ps_x':3,'ps_t':2}
        # self.run_single_test(dname,sigma,comp_flow,pyargs)
