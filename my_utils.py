from pylab import *
import os
import math
import numpy as np
import torch

import config

def vprint(string, **kwargs):
	if config.verboseFlag:
		return print(string, **kwargs)
	else:
		return False
		
def getAverage(xb, yb, dx):
	# calculates the mean of a distribution in the domain xb
	# by weighting the values in the domain by the distribution
	# value yb. I assume the lengths are equivalent and that they
	# are flattened, regardless of dimensionality. This means
	# it works for 1d and 2d distributions
	if type(xb) != torch.Tensor:
		x = torch.Tensor(xb)
	else:
		x = xb.detach().clone()
	if type(yb) != torch.Tensor:
		y = torch.Tensor(yb)
	else:
		y = yb.detach().clone()
	# calculate l1norm
	yl1norm = torch.sum(y)*dx
	avg = yl1norm/(xb[-1]-xb[0])
	return avg
	
def getWeightedMean(xb, yb, dx):
	# calculates the mean of a distribution in the domain xb
	# by weighting the values in the domain by the distribution
	# value yb. I assume the lengths are equivalent and that they
	# are flattened, regardless of dimensionality. This means
	# it works for 1d and 2d distributions
	if type(xb) != torch.Tensor:
		x = torch.Tensor(xb)
	else:
		x = xb.detach().clone()
	if type(yb) != torch.Tensor:
		y = torch.Tensor(yb)
	else:
		y = yb.detach().clone()
	# rescale yb so that its l1 norm is 1 
	yl1norm = torch.sum(y)*dx
	y = y/yl1norm
	weightedX = torch.zeros(x.size())
	for n in range(len(x)):
		weightedX[n] = x[n]*y[n]*dx
	return torch.sum(weightedX, dim=0)
	
def getStdDev(xb,yb, dx):
	print("getStdDev...")
	if type(xb) != torch.Tensor:
		x = torch.Tensor(xb)
	else:
		x = xb.detach().clone()
	if type(yb) != torch.Tensor:
		y = torch.Tensor(yb)
	else:
		y = yb.detach().clone()

	var = getWeightedMean(torch.pow(x, 2),y, dx)-torch.pow(getWeightedMean(x,y,dx), 2)
	var = var.numpy()
	print(var)
	print(np.sqrt(var))
	return np.sqrt(var)
	
def sigmaClip(xb,yb):
	numSigmas = 2
	mean = getWeightedMean(xb,yb)
	std = getStdDev(xb,yb)
	x_clipped = []
	y_clipped = []
	for k in range(len(xb)):
		diff = xb[k]-mean
		if diff[0].norm() <= (numSigmas*std[0])**2:
			if diff[1].norm() <= (numSigmas*std[1])**2:
				x_clipped.append(xb[k])
				y_clipped.append(yb[k])
	return torch.Tensor(x_clipped), torch.Tensor(y_clipped)
		
def Func2x(x, params, dim, domainSize, func1x):
	# assumes the 2x function can be factored into 1x versions
	# assume params has the shape (h, params), where all are 2d arrays except h
	vprint("__Func2x(x,params=%s)" % str(params))
	xType = type(x)
	vprint("Running __Func2x using %s..." % func1x.__name__)
	vprint("Input to __Func2x is a %s" % str(xType))
	# checks that x is at least a list-like structure
	if xType==np.ndarray or xType==torch.Tensor:
		vprint("Input to __Func2x is a %s with shape %s" % (str(xType), str(list(x.shape))))
		# determine shape of input
		rtn = 1
		if (list(x.shape)==[dim]):
			vprint("Recognized shape [dim]")
			for n in range(dim):
				rtn = rtn * func1x(x[n], [row[n] for row in params[1:]])
			vprint("rtn has shape %s" % str(rtn.size()))
			return params[0]*rtn
		elif (list(x.shape)==[dim, domainSize]):	
			vprint("Recognized shape [dim, domainSize]")
			rtn = func1x(x[0], [row[0] for row in params[1:]])
			for n in range(dim-1):
				rtn = torch.ger(rtn, func1x(x[n+1], [row[n+1] for row in params[1:]]))	# ger implements outer product
				# returns a matrix, value on a 2d grid
			vprint("rtn has shape %s" % str(rtn.size()))
			return params[0]*rtn
		elif (list(x.shape)==[domainSize]*dim+[dim]):
			vprint("Recognized shape [domainSize,...,domainSize, dim]")
			x_reshape = torch.reshape(x, tuple([dim]+[domainSize]*dim))
			for n in range(dim):
				rtn = rtn * func1x(x_reshape[n], [row[n] for row in params[1:]])
			if xType == np.ndarray:
				rtn = rtn.numpy()
			# returns a matrix, of shape [domainSize]*dim, value on a grid
			vprint("rtn has shape %s" % str(rtn.size()))
			return params[0]*rtn
		elif (x.shape[-1] == dim):
			vprint("Recognized shape [...,dim]")
			# assume the shape for doing batches
			rtn = torch.zeros((x.shape[0]))
			x_reshape = torch.reshape(x, (dim, x.shape[0]))
			val = 1
			for n in range(dim):
				val = val * func1x(x_reshape[n], [row[n] for row in params[1:]])
			rtn = val
			if xType == np.ndarray:
				rtn = rtn.numpy()
			vprint("rtn has shape %s" % str(rtn.size()))
			return params[0]*rtn
	elif xType==list:
		if len(x) == dim:
			rtn = 1
			for n in range(dim):
				rtn = rtn * func1x(x[n], [row[n] for row in params[1:]])
			return params[0]*rtn
	else:
		print("ERROR: __Func2x wasn't passed a list-like structure")
		print("Consider using %s for 1d domains" % func1x.__name__)
		return -1
	print("Evaluation not implemented for this data structure")
	return -1
