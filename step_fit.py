from pylab import *
import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

def logisticFunc(x, h, rate, mean):
	xType = type(x)
	if xType == torch.Tensor or xType == np.array:
		rtn = h/(1+torch.exp(rate*(x-mean)))
		return rtn
	else:
		rtn = h/(1+math.exp(rate*(x-mean)))
		return rtn
def logisticStepFunc(x, params):
	Lshifted = logisticFunc(x, h=params[0], rate=params[3], mean=2*(params[2]-params[1]))
	Rshifted = logisticFunc(x, h=params[0], rate=params[3], mean=2*(params[2]+params[1]))
	return torch.abs(Lshifted - Rshifted)

class StepFit(nn.Module):
	def __init__(self, *args, flagTrain=True):
		super(StepFit, self).__init__()
		self.ho = torch.abs(torch.randn(1))
		self.devo = torch.abs(torch.randn(1))
		self.meano = torch.randn(1)*5
		self.rateo = torch.abs(torch.randn(1)*5)
		print("Step function initiated with...")
		print("\theight: %f" % float(self.ho))
		print("\tstd dev: %f" % float(self.devo))
		print("\tmean: %f" % float(self.meano))
		print("\trate: %f" % float(self.rateo))
		self.flagTrain = flagTrain
		if not flagTrain:												# when train is false, don't keep track of grads
			self.h = self.ho.detach().clone()
			self.dev = self.devo.detach().clone()
			self.mean = self.meano.detach().clone()
			self.rate = self.rateo.detach().clone()
		else:
			self.h = nn.Parameter(self.ho.detach().clone())
			self.dev = nn.Parameter(self.devo.detach().clone())
			self.mean = nn.Parameter(self.meano.detach().clone())
			self.rate = nn.Parameter(self.rateo.detach().clone())
	def forward(self, q):											# we expect ae vector of input data to be passed to forward
		return self.getField(q)
	def getField(self, x):
		return logisticStepFunc(x, (self.h, self.dev, self.mean, self.rate))
	def getField_o(self, x):
		return logisticStepFunc(x, (self.ho, self.devo, self.meano, self.rateo))
	def getParams(self):
		return (self.h, self.dev, self.mean, self.rate)
	def setParams(self, q):
		if self.flagTrain:
			print("WARNING: Setting the parameters of the Gaussian despite training status!")
		qTorch = [None]*len(q)
		for i in range(len(qTorch)):
			qTorch[i] = torch.tensor([q[i]])
		state_dict = {'h': qTorch[0], 'dev': qTorch[1], 'mean': qTorch[2], 'rate': qTorch[3]}
		print(self.state_dict())
		self.load_state_dict(state_dict, strict=True)
		return 0
		
class SumStepFit(nn.Module):
	def __init__(self, num, *args):
		super(SumStepFit, self).__init__()
		self.lstDistr = nn.ModuleList([StepFit() for _ in range(num)])
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
 
