import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

from my_utils import *
from plot_data import *

#np.random.seed(1)
#torch.manual_seed(1)

n_epochs = 4   # number of optimization epochs
lr =   0.001  # learning rate 
bs = 1        # batch size 

# Physical constants (using normalized code units).
K_B = 1.0
EPSILON_0 = 1.0 													# Permittivity of free space.
MU_0 = 1.0																# Permiability of free space.
LIGHT_SPEED = 1/math.sqrt(MU_0*EPSILON_0)	# Speed of light.

ELC_MASS = 1.0 													# Electron mass.
ELC_CHARGE = -1.0												# Electron charge.
N_0 = 1.0																# initial reference density

def main():
	# specify slice of data to fit
	tslice = 199
	xslice = -15
	# load data
	mySys = System(fileloc="hotWater/hotWater", pMinFrames=tslice, pMaxFrames=tslice, pMass=ELC_MASS, pCharge=ELC_CHARGE, pN0=1, pVth=1)
	X, V, f = mySys.getField("elc", tslice)
	# create training data from x slice
	for n in range(len(X)):
		if X[n] < xslice and X[n+1] >= xslice:
			n = n+1
			break
	xsliceInd = n
	x_train = V
	print("x_train shape")
	print(x_train.shape)
	y_train = f[xsliceInd]
	# normalize data set
	y_train = y_train/y_train.max()
	
	# for fitting datasets I will set train=valid
	x_valid = x_train.copy()
	y_valid = y_train.copy()
	 # convert to pytorch tensors
	x_train, y_train, x_valid, y_valid = map( torch.tensor, (x_train, y_train, x_valid, y_valid))
	
	model, opt = get_model(lr, "Step", 2)
	
	train_ds = TensorDataset(x_train, y_train)
	valid_ds = TensorDataset(x_valid, y_valid)
	loss_func = F.mse_loss
	hist = fit(n_epochs, model, opt, loss_func, bs, train_ds, valid_ds)
	plot_acc_loss(hist)

	print("Final accuracy of model")
	print(accuracy(model, loss_func, x_valid, y_valid))
	
	x_pred = x_train.detach().clone()
	y_pred = model.getField(x_pred)
	
	x_org = x_pred.detach().clone()
	y_org = model.getField_o(x_org)
	print(y_org)
	
	figure, axes = subplots()
	plotFigFromData((x_train.detach().numpy(), y_train.detach().numpy()), figure, axes, color='blue')
	plotFigFromData((x_org.detach().numpy(), y_org.detach().numpy()), figure, axes, color='red')
	plotFigFromData((x_pred.detach().numpy(), y_pred.detach().numpy()), figure, axes, color='green')
	show()

	return

if __name__ == "__main__":
	main()