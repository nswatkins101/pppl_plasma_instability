from pylab import *
import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# import python scripts for each model
from gauss_fit import *
from gauss_autoencoder import *

dtype = torch.cuda.DoubleTensor if torch.cuda.is_available() else torch.DoubleTensor
device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.set_default_dtype(torch.float64)

def scipyLoss(x, model, loss_func, x_data, y_data):
	print("Printing params in loss %s" % str(x))
	model.setParams(x)
	loss = loss_func(model(x_data), y_data)
	loss.backward()
	return loss.data.numpy(), np.array([param.grad.data[0] for param in model.parameters()])