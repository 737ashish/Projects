import torch
import torch.nn as nn
import poly_utils as ut
import numpy as np

def runge2D(x):
    return 1/(1 + 25*x[:,0]**2 + 25*x[:,1]**2)

def runge2D_shift(x):
    return 1/(1 + 25*x[:,0]**2 + 25*x[:,1]**2) - 0.5 