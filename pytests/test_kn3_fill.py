
# -- misc --
import sys,tqdm
from pathlib import Path

# -- dict data --
import copy
from easydict import EasyDict as edict

# -- vision --
import cv2

# -- testing --
import unittest
import tempfile

# -- linalg --
import torch as th
import numpy as np
from einops import rearrange

# -- package --
from faiss.contrib import kn3

# -- check if reordered --
from scipy import optimize
SAVE_DIR = Path("./output/tests/")

#
#
# -- Primary Testing Class --
#
#

class TestKn3Fill(unittest.TestCase):

    #
    # -- Load Data --
    #

    def run_fill_output_test(self,val_dists,val_inds):
        dists,inds = kn3.run_fill_output(val_dists,val_inds)
        error_vals = th.sum((dists - val_dists)**2).item()
        error_inds = th.sum((inds - val_inds)**2).item()
        assert error_vals < 1e-10
        assert error_inds < 1e-10

    def run_fill_input_test(self,val_burst,val_query):
        bsize = 10
        shape = (3,3,3,3)
        burst,query = kn3.run_fill_input(val_burst,val_query,shape,bsize)
        error_burst = th.sum((burst - val_burst)**2).item()
        error_query = th.sum((query - val_query)**2).item()
        assert error_burst < 1e-10
        assert error_query < 1e-10

    def run_fill_patches_test(self,val):
        t = 8
        pshape = (20,5,1,3,11,11) # (qsize,k,pt,c,ps,ps)
        patches = kn3.run_fill_patches(val,pshape,t)
        error = th.sum((patches - val)**2).item()
        assert error < 1e-10

    def test_sim_search(self):

        # -- test 1 --
        a,b = 3,4
        self.run_fill_input_test(a,b)
        self.run_fill_output_test(a,b)
        self.run_fill_patches_test(a)
