
import torch as th
import numpy as np
from PIL import Image
from pathlib import Path

def load_dataset(dname):

    # -- constant for limiting weird behavior --
    MAX_NFRAMES = 85

    # -- get path --
    data_root = Path("/home/gauenk/Documents/packages/faiss/pytests/data")
    if not data_root.exists():
        print(f"Error: Data patch DNE: [{str(data_root)}]")
    data_dir = data_root / dname
    if not data_dir.exists():
        print(f"Error: Data patch DNE: [{str(data_dir)}]")

    # -- load frames --
    burst = []
    for fid in range(MAX_NFRAMES):
        img_fn = data_dir / ("%05d.png" % fid)
        if not img_fn.exists(): break
        img = Image.open(img_fn).convert("RGB")
        img = np.array(img).transpose(2,0,1)
        burst.append(img)
    burst = np.stack(burst)
    burst = th.from_numpy(burst)
    return burst
