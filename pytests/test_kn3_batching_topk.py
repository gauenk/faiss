
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

def save_image(burst,prefix="prefix"):
    root = PYTEST_OUTPUT
    if not(root.exists()): root.mkdir()
    burst = rearrange(burst,'t c h w -> t h w c')
    burst = np.clip(burst,0,255)
    burst = burst.astype(np.uint8)
    nframes = burst.shape[0]
    for t in range(nframes):
        fn = "%s_kn3_io_%02d.png" % (prefix,t)
        img = Image.fromarray(burst[t])
        path = str(root / fn)
        img.save(path)

def get_empty_bufs(K,args,shape,device):
    ps,pt = args.ps,args.pt
    stride = args.queryStride
    t,c,h,w = shape
    return init_empty_bufs(K,stride,ps,pt,t,c,h,w,device)

def init_empty_bufs(k,stride,ps,pt,t,c,h,w,device):
    nq = (t*h*w)//stride+1
    cshape = (nq,k)
    pshape = (nq,k,pt,c,ps,ps)
    tf32,ti32 = th.float32,th.int32
    bufs = edict()
    bufs.patches = th.zeros(pshape,device=device,dtype=tf32)
    bufs.dists = th.zeros(cshape,device=device,dtype=tf32)
    bufs.inds = th.zeros(cshape,device=device,dtype=ti32)
    return bufs


class TestBatchingTopKPatches(unittest.TestCase):

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

    def exec_kn3_search_exh(self,K,clean,flows,sigma,args,bufs,pfill):

        # -- unpack --
        device = clean.device
        shape = clean.shape
        t,c,h,w = shape

        # -- prepare kn3 search  --
        index,npix = 0,t*h*w
        args.k = K
        numQueryTotal = (npix-1)//args.queryStride+1

        # -- search --
        kn3.run_search(clean,0,numQueryTotal,flows,sigma,args,bufs,pfill=pfill)
        th.cuda.synchronize()

        # -- unpack --
        kn3_dists = bufs.dists
        kn3_inds = bufs.inds
        kn3_patches = bufs.patches

        return kn3_dists,kn3_patches


    def exec_kn3_search_bch(self,K,clean,flows,sigma,args,bufs,pfill):

        # -- unpack --
        device = clean.device
        shape = clean.shape
        t,c,h,w = shape

        # -- prepare kn3 search  --
        index,npix = 0,t*h*w
        args.k = K
        bsize = 1000
        numQueryTotal = (npix-1)//args.queryStride+1
        nbatches = (numQueryTotal-1)//bsize + 1

        def view_buffer(bufs,batch,bsize):
            # -- get slice --
            start = batch * bsize
            end =  (batch+1) * bsize
            bslice = slice(start,end)

            # -- apply slice --
            view_bufs = edict()
            for key,val in bufs.items():
                view_bufs[key] = val[bslice]
            return view_bufs

        # -- iterate over batches --
        for batch in range(nbatches):

            # -- view buffer --
            view_bufs = view_buffer(bufs,batch,bsize)

            # -- search --
            qstart = bsize*batch
            bsize_b = min(bsize,numQueryTotal - qstart)
            # print(view_bufs.dists.shape,bsize_b,qstart,numQueryTotal,batch,nbatches,bsize)
            assert bsize_b > 0,"strictly positive batch size."
            kn3.run_search(clean,qstart,bsize_b,flows,sigma,
                           args,view_bufs,pfill=pfill)
            th.cuda.synchronize()
        kn3_dists = bufs.dists
        kn3_patches = bufs.patches
        return kn3_dists,kn3_patches

    #
    # -- [Exec] Sim Search --
    #

    def run_comparison(self,noisy,clean,sigma,flows,args,pfill):

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
        exh_bufs = edict()
        exh_bufs.patches = None
        exh_bufs.dists = None
        exh_bufs.inds = None

        # -- setup args --
        args['stype'] = "faiss"
        args['vpss_mode'] = "exh"
        args['queryStride'] = 7
        args['bstride'] = args['queryStride']
        # args['vpss_mode'] = "vnlb"

        # -- empty bufs --
        bch_bufs = get_empty_bufs(K,args,shape,device)

        # -- exec over batches --
        for index in range(NBATCHES):

            # -- get data --
            clean = 255.*th.rand_like(clean).type(th.float32)
            # clean /= 255.
            # clean *= 255.
            noisy = clean.clone()

            # -- search using batching --
            bch_dists,bch_patches = self.exec_kn3_search_bch(K,clean,flows,sigma,
                                                             args,bch_bufs,pfill)

            # -- search using exh search --
            exh_dists,exh_patches = self.exec_kn3_search_exh(K,clean,flows,sigma,
                                                             args,exh_bufs,pfill)

            # -- to numpy --
            bch_dists = bch_dists.cpu().numpy()
            exh_dists = exh_dists.cpu().numpy()

            bch_patches = bch_patches.cpu().numpy()
            exh_patches = exh_patches.cpu().numpy()

            # -- allow for swapping of "close" values --
            np.testing.assert_array_equal(exh_dists,bch_dists)
            if pfill:
                np.testing.assert_array_equal(exh_patches,bch_patches)

    def run_single_test(self,dname,sigma,comp_flow,pyargs):
        noisy,clean = self.do_load_data(dname,sigma)
        flows = self.do_load_flow(False,clean,sigma,noisy.device)
        # -- fill patches --
        self.run_comparison(noisy,clean,sigma,flows,pyargs,True)
        # -- fill dists only --
        self.run_comparison(noisy,clean,sigma,flows,pyargs,False)

    def test_batch_sim_search(self):

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
