import torch
import torch.nn as nn
import poly_utils as ut
import numpy as np

def runge1D(x):
    return 1/(1 + 25*x**2)

def runge1D_shift(x):
    return 1/(1 + 25*x**2) - 0.5

def sin(x):
    return np.sin(np.pi * x) 
