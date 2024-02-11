from pylab import *
import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from my_utils import *

def logisticFunc1x(x, rate, mean):
	if type(x) == torch.Tensor:
		return 4/(1+torch.exp(rate*(x-mean)))
	elif type(x) == np.ndarray:
		return 4/(1+np.exp(rate*(x-mean)))
	else:
		return 4/(1+math.exp(rate*(x-mean)))
	
def logisticStepFunc1x(x, params):
	# assumes params have shape (dev, mean, rate), each is a float
	Lshifted = logisticFunc1x(x, rate=params[2], mean=2*(params[1]-params[0]))
	Rshifted = logisticFunc1x(x, rate=params[2], mean=2*(params[1]+params[0]))
	if type(Lshifted) == torch.Tensor:
		return torch.abs(Lshifted - Rshifted)
	elif type(Lshifted) == np.ndarray:
		return np.abs(Lshifted - Rshifted)
	elif type(Lshifted) == float:
		return abs(Lshifted - Rshifted)
	else:
		print("ERROR: return type of logisticFunc2x is not recognized")
		return -1

def logisticStepFunc2x(x, params, domainSize):
	# assume params has the shape (h, dev, mean, rate), where all are 2d arrays except h
	vprint("logisticStepFunc2x(x,params=%s)" % str(params))
	dim = len(params[1])										# number of dev parameters determines dimensionality
	xType = type(x)
	vprint("Running logisticStepFunc2x...")
	vprint("Input to logisticFunc2x is a %s" % str(xType))
	# checks that x is at least a list-like structure
	if xType==np.ndarray or xType==torch.Tensor:
		vprint("Input to logisticFunc2x is a %s with shape %s" % (str(xType), str(list(x.shape))))
		# determine shape of input
		rtn = 1
		if (list(x.shape)==[dim]):
			vprint("Recognized shape [dim]")
			for n in range(dim):
				rtn = rtn * logisticStepFunc1x(x[n], [row[n] for row in params[1:]])
			vprint("rtn has shape %s" % str(rtn.size()))
			return params[0]*rtn
		elif (list(x.shape)==[dim, domainSize]):	
			vprint("Recognized shape [dim, domainSize]")
			rtn = logisticStepFunc1x(x[0], [row[0] for row in params[1:]])
			for n in range(dim-1):
				rtn = torch.ger(rtn, logisticStepFunc1x(x[n+1], [row[n+1] for row in params[1:]]))	# ger implements outer product
				# returns a matrix, value on a 2d grid
			vprint("rtn has shape %s" % str(rtn.size()))
			return params[0]*rtn
		elif (list(x.shape)==[domainSize]*dim+[dim]):
			vprint("Recognized shape [domainSize,...,domainSize, dim]")
			x_reshape = torch.reshape(x, tuple([dim]+[domainSize]*dim))
			for n in dim:
				rtn = rtn * logisticStepFunc1x(x[n], [row[n] for row in params[1:]])
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
				val = val * logisticStepFunc1x(x_reshape[n], [row[n] for row in params[1:]])
			rtn = val
			if xType == np.ndarray:
				rtn = rtn.numpy()
			vprint("rtn has shape %s" % str(rtn.size()))
			return params[0]*rtn
	elif xType==list:
		if len(x) == dim:
			rtn = 1
			for n in range(dim):
				rtn = rtn * logisticStepFunc1x(x[n], [row[n] for row in params[1:]])
			return params[0]*rtn
	else:
		print("ERROR: logisticFunc2x wasn't passed a list-like structure")
		print("Consider using logisticFunc for 1d domains")
		return -1
	print("Evaluation not implemented for this data structure")
	return -1

class StepFit2x(nn.Module):
	def __init__(self, domainSize, *args, flagTrain=True):
		super(StepFit2x, self).__init__()
		self.dim = 2
		self.domainSize = domainSize
		self.ho = torch.Tensor([1])									# fixed instead of torch.sqrt(torch.abs(torch.randn(1)+1)*10)
		self.devo = (torch.abs(torch.randn(self.dim,1)+1)*2)
		self.meano = torch.randn(self.dim,1)
		self.rateo = torch.abs(torch.randn(self.dim,1))
		print("Step function initiated with...")
		print("\theight: %s" % str(self.ho))
		print("\tstd dev: %s" % str(self.devo))
		print("\tmean: %s" % str(self.meano))
		print("\trate: %s" % str(self.rateo))
		self.flagTrain = flagTrain
		if not flagTrain:												# when train is false, don't keep track of grads
			self.h = self.ho.detach().clone()
			self.dev = self.devo.detach().clone()
			self.mean =self.meano.detach().clone()
			self.rate = self.rateo.detach().clone()
		else:
			self.h = nn.Parameter(self.ho.detach().clone())
			self.dev = nn.Parameter(self.devo.detach().clone())
			self.mean = nn.Parameter(self.meano.detach().clone())
			self.rate = nn.Parameter(self.rateo.detach().clone())
		
	def forward(self, q):											# we expect ae vector of input data to be passed to forward
		return self.getField(q)
	def getField(self, x):
		return logisticStepFunc2x(x, (self.h, self.dev, self.mean, self.rate), self.domainSize)
	def getField_o(self, x):
		return logisticStepFunc2x(x, (self.ho, self.devo, self.meano, self.rateo), self.domainSize)
	def getParams(self):
		return (self.h, self.dev, self.mean, self.rate)
	def setParams(self, q):
		if self.flagTrain:
			print("WARNING: Setting the parameters of the distribution despite training status!")
		qTorch = [None]*len(q)
		for i in range(len(qTorch)):
			qTorch[i] = torch.tensor([q[i]])
		state_dict = {'h': qTorch[0], 'dev': qTorch[1], 'mean': qTorch[2], 'rate': qTorch[3]}
		print(self.state_dict())
		self.load_state_dict(state_dict, strict=True)
		return 0
		
class SumStepFit2x(nn.Module):
	def __init__(self, num, domainSize, *args):
		super(SumStepFit2x, self).__init__()
		self.lstDistr = nn.ModuleList([StepFit2x(domainSize) for _ in range(num)])
		self.Num = num
	def forward(self, q):
		f = 0
		for distr in self.lstDistr:
			f = f + distr.forward(q)
		return f
	def getField(self, x):
		f  = 0
		for distr in self.lstDistr:
			f = f + distr.getField(x)
		return f
	def getField_o(self, x):
		f = 0
		for distr in self.lstDistr:
			f = f + distr.getField_o(x)
		return f
	def getParams(self):
		rtn = [distr.getParams() for distr in self.lstDistr]
		print(len(rtn))
		return rtn
 
