import os
import os.path as osp
import numpy as np

def check_dir(dir):
    if not osp.exists(dir):
        os.makedirs(dir)


