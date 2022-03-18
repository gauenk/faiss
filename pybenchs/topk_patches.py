"""
Comparing our GPU Patch Matching with Competition

"""

# -- sys add --
import matplotlib
matplotlib.use("agg")
import os,sys
# sys.path.append("./core/")

# -- linalg --
import numpy as np
import numpy.random as npr
import torch as th
from einops import rearrange,repeat

# -- cache io --
import cache_io
from easydict import EasyDict as edict

# -- python-kernel search --
import vpss
import vnlb

# -- faiss-tiling search --
from n2sim.sim_search import compute_sim_images,faiss_global_search
from align.nnf import compute_burst_nnf

# -- faiss-burst search --
from faiss.contrib import kn3
from faiss.contrib import testing # helper
from faiss.contrib.kn3 import Timer


# ----------------------------------------
#
#   Patch Matching Functions [Eccv 2022]
#
# ----------------------------------------

def exec_pm_faiss_tiling_eccv2022(clock,burst,ps=7,subsize=100):
    clock.tic()
    nframes = int(burst.shape[0])
    burst = rearrange(burst,'t c h w -> t 1 c h w')
    t = nframes//2
    compute_burst_nnf(burst,t,ps,subsize)
    th.cuda.synchronize()
    clock.toc()

def exec_pm_faiss_burst_eccv2022(clock,burst,ps=7,subsize=100):
    clock.tic()
    t,c,h,w = burst.shape
    npix = t*h*w
    BSIZE = npix
    flows,sigma,bufs = None,0.,None
    args = edict()
    args.k = subsize
    args.queryStride = 3
    args.ws = 2
    BSIZE = (npix-1)//args.queryStride + 1
    bufs = kn3.run_search(burst,0,BSIZE,flows,sigma/255.,args,bufs,pfill=True)
    clock.toc()

# ---------------------------
#
#   Patch Matching Functions
#
# ---------------------------

def exec_pm_numba(clock,burst,ps=7,subsize=100):
    clock.tic()

    args = vpss.exh_search_default_args()
    args.bstride = 7
    args.ps = 7
    args.k = subsize
    vpss.exh_search(burst,None,0.,subsize,args)
    # vnlb.global_search_default(burst,0.,clock,ps,subsize,pfill=True)

    th.cuda.synchronize()
    clock.toc()

def exec_pm_faiss_tiling(clock,burst,ps=7,subsize=100):
    clock.tic()
    nframes = int(burst.shape[0])
    burst = rearrange(burst,'t c h w -> t 1 c h w')
    # for t in range(nframes):
    t = nframes//2
    compute_burst_nnf(burst,t,ps,subsize)
    # faiss_global_search(burst,ps,subsize)
    th.cuda.synchronize()
    clock.toc()

def exec_pm_faiss_burst(clock,burst,ps=7,subsize=100):
    clock.tic()
    t,c,h,w = burst.shape
    npix = t*h*w
    BSIZE = npix
    flows,sigma,bufs = None,0.,None
    args = edict()
    args.k = subsize
    args.queryStride = 7
    args.wf = 6
    args.wb = 6
    args.ws = 10
    # BSIZE = (npix-1)//args.queryStride + 1
    bufs = kn3.run_search(burst,0,BSIZE,flows,sigma,args,bufs,pfill=True)
    print("bufs.patches.shape: ",bufs.patches.shape)
    th.cuda.synchronize()
    clock.toc()

def pm_select(method):
    if method == "tiling":
        time_fxn = exec_pm_faiss_tiling
    elif method == "burst":
        time_fxn = exec_pm_faiss_burst
    elif method == "burst_eccv2022":
        time_fxn = exec_pm_faiss_burst_eccv2022
    elif method == "numba":
        time_fxn = exec_pm_numba
    else:
        raise ValueError(f"Unknown method [{method}]")
    return time_fxn

# ---------------------------
#
#       Timing module
#
# ---------------------------

def timing_pm(cfg):

    # -- select method --
    time_fxn = pm_select(cfg.method)

    # -- create results --
    results = edict()
    results.times = []

    # -- burst --
    tf32 = th.float32
    device = "cuda:0"
    burst = npr.rand(cfg.t,3,cfg.hw,cfg.hw)
    burst = th.from_numpy(burst).to("cuda:0").type(tf32)

    # -- patch matching GPU --
    print(f"Patch Matching [{cfg.method}]")
    for i in range(cfg.nreps):
        th.cuda.empty_cache()
        clock = Timer()
        time_fxn(clock,burst,cfg.ps,cfg.subsize)
        results.times.append(clock.diff)
    return results

# ---------------------------
#
#       Main Function
#
# ---------------------------

def main():

    # -- (1) Init --
    pid = os.getpid()
    print("PID: ",pid)
    verbose = True
    cache_dir = ".cache_io"
    cache_name = "topk_patches"
    cache = cache_io.ExpCache(cache_dir,cache_name)
    # cache.clear()

    # -- (2) create experiments --
    exps = {"t":[10],"hw":[64,128,200],"ps":[7],"subsize":[3],
            "method":["burst","numba"],"nreps":[2]} # "numba",
    experiments = cache_io.mesh_pydicts(exps) # create mesh
    exps = {"t":[10],"hw":[64,128,256,512],"ps":[7],"subsize":[3],
            "method":["burst","tiling"],"nreps":[2]} # "numba",
    experiments += cache_io.mesh_pydicts(exps) # create mesh


    # -- (3) [Execute or Load] each Experiment --
    nexps = len(experiments)
    for exp_num,config in enumerate(experiments):

        # -- setup --
        th.cuda.empty_cache()

        # -- info --
        if verbose:
            print("-="*25+"-")
            print(f"Running exeriment number {exp_num+1}/{nexps}")
            print("-="*25+"-")
            print(config)

        # -- logic --
        uuid = cache.get_uuid(config) # assing ID to each Dict in Meshgrid
        # cache.clear_exp(uuid) # RESET VNLB.
        results = cache.load_exp(config) # possibly load result
        if results is None: # check if no result
            results = timing_pm(config)
            cache.save_exp(uuid,config,results) # save to cache

    # -- (4) print results! --
    records = cache.load_flat_records(experiments)
    fields = ['times','method','hw']
    print(records[fields])


if __name__ == "__main__":
    main()
