import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

from torch_utils import *
from my_utils import *
from plot_data import *

#np.random.seed(1)
#torch.manual_seed(1)

n_epochs = 50   # number of optimization epochs
lr =   0.01   # learning rate 
bs = 10        # batch size 
inverse_decay = 0.001
weight_decay = 0.0
pOrder = 1

# Physical constants (using normalized code units).
K_B = 1.0
EPSILON_0 = 1.0 													# Permittivity of free space.
MU_0 = 1.0																# Permiability of free space.
LIGHT_SPEED = 1/math.sqrt(MU_0*EPSILON_0)	# Speed of light.

ELC_MASS = 1.0 													# Electron mass.
ELC_CHARGE = -1.0												# Electron charge.
N_0 = 1.0																# initial reference density

def wrapped_mse_loss(model, xb, yb):
	# implement renormalization weight decay in the optimizer, inverse_decay here
	# inverse_decay prevents the model from fitting the data with h=0, i.e. f=0
	# which eliminates the need to cull the zero data points.
	# The inverse polynomial order determines how much the minimum
	# is shifted by the additional term. 
	#return F.mse_loss(model(xb),yb) + inverse_decay/(model.h.norm()+model.dev.norm())**pOrder + weight_decay*(model.dev.norm())**pOrder
	
	return 1000*F.mse_loss(model(xb),yb)

def main():
	# testing using Pytorch architecture to train nonlinear regression
	xSize = 200
	x_list = np.linspace(-10, 10, num=xSize)
	gaussian = GaussianSpace(1, xSize)
	x_train, y_train = gaussian.getXY(x_list)
	
	x_valid = x_train.clone()
	y_valid = y_train.clone()
	 # convert to pytorch tensors
	#x_train, y_train, x_valid, y_valid = map( torch.tensor, (x_train, y_train, x_valid, y_valid))
	
	model = get_model(lr, "Gaussian")
	opt = get_opt(lr, model)
	
	train_ds = TensorDataset(x_train, y_train)
	valid_ds = TensorDataset(x_valid, y_valid)
	
	loss_func = wrapped_mse_loss
	hist = fit(n_epochs, model, opt, loss_func, bs, train_ds, valid_ds)
	plot_acc_loss(hist)

	print("Final accuracy of model")
	print(accuracy(model, loss_func, x_valid, y_valid))
	print(model.getParams())
	
	x_pred = x_train.detach().clone()
	y_pred = model.getField(x_pred)
	
	x_org = x_pred.detach().clone()
	y_org = model.getField_o(x_org)
	
	figure, axes = subplots()
	plotFigFromData((x_pred.detach().numpy(), y_pred.detach().numpy()), figure, axes, color='green')
	plotFigFromData((x_org.detach().numpy(), y_org.detach().numpy()), figure, axes, color='red')
	plotFigFromData((x_train.detach().numpy(), y_train.detach().numpy()), figure, axes, color='blue')
	show()

	return

if __name__ == "__main__":
	main()