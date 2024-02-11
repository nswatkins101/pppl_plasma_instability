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
from step_fit import *
from gauss_fit_2x import *
from step_fit_2x import *
from gauss_autoencoder import *

dtype = torch.cuda.DoubleTensor if torch.cuda.is_available() else torch.DoubleTensor
device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.set_default_dtype(torch.float64)

def get_model(lr, option, *args):
	vprint("Instantiating a subclass of nn.Module, option=%s" % option)
	# consider using a dictionary
	if option == "SumGaussians":
		model = SumGaussianFit(*args)
	elif option == "SumSteps":
		model = SumStepFit(*args)
	elif option == "Gaussian":
		model = GaussianFit(*args)
	elif option == "Step":
		model = StepFit(*args)
	elif option == "SumGaussians2x":
		model = SumGaussianFit2x(*args)
	elif option == "SumSteps2x":
		model = SumStepFit2x(*args)
	elif option == "Gaussian2x":
		model = GaussianFit2x(*args)
	elif option == "Step2x":
		model = StepFit2x(*args)
	elif option == "Autoencoder":
		model = Autoencoder(*args)
	else:
		print("Warning: Model is not supported")
		model = 0
	return model
	
def get_opt(lr, model, opt="RMSprop"):
	vprint("Params:", end=' ')
	vprint([param.data for param in model.parameters(recurse=True)])
	if opt == "LBFGS":
		optimizer = optim.LBFGS(model.parameters(), lr=lr, max_iter=20, max_eval=None, tolerance_grad=1e-07, tolerance_change=1e-09, history_size=100, line_search_fn=None)
	else:
		optimizer = optim.RMSprop(model.parameters(), lr=lr, momentum=0.75)
		#print("Warning: Optimizer failed to initialize")
	
	return optimizer
	
def closure():
	optimizer.zero_grad()
	output = model(input)
	loss = loss_fn(output, target)
	loss.backward()
	return loss
		
def fit(epochs, model, opt, loss_func, bs, train_ds, valid_ds):
	"""
	trains the model and computes the losses for the training and validation sets
	always call model.train() BEFORE training since it used by some layers
	always call model.eval() BEFORE inference since it used by some layers (like dropout)
	two element dictionary of epochs-lengthed lists for histogram of accuracy and loss
	"""
	train_dl, valid_dl = get_data(train_ds, valid_ds, bs)
	history = {"train_loss": np.zeros(epochs), "train_accuracy": np.zeros(epochs), "valid_loss": np.zeros(epochs), "valid_accuracy": np.zeros(epochs)}
	for epoch in range(epochs):
		model.train()
	   
		for xb, yb in train_dl:
			loss_batch(model, loss_func, xb, yb, opt)

		model.eval()
		with torch.no_grad():
			train_losses, train_loss_nums = zip(
				*[loss_batch(model, loss_func, xb, yb) for xb, yb in train_dl]
			)
			train_accuracies, train_acc_nums = zip(
				*[accuracy(model, loss_func, xb, yb) for xb, yb in train_dl]
			)
			valid_losses, valid_loss_nums = zip(
				*[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl]
			)
			valid_accuracies, valid_acc_nums = zip(
				*[accuracy(model, loss_func, xb, yb) for xb, yb in valid_dl]
			)
		train_loss = np.sum(np.multiply(train_losses, train_loss_nums)) / np.sum(train_loss_nums)
		train_acc = np.sum(np.multiply(train_accuracies, train_acc_nums)) / np.sum(train_acc_nums)
		val_loss = np.sum(np.multiply(valid_losses, valid_loss_nums)) / np.sum(valid_loss_nums)
		val_acc = np.sum(np.multiply(valid_accuracies, valid_acc_nums)) / np.sum(valid_acc_nums)
		history["train_loss"][epoch] = train_loss
		history["train_accuracy"][epoch] = train_acc
		history["valid_loss"][epoch] = val_loss
		history["valid_accuracy"][epoch] = val_acc
		print("Epoch: %i | Train Loss: %f | Validating Loss: %f | Train Accuracy: %f | Validating Accuracy: %f" 
			% (epoch, train_loss, val_loss, train_acc, val_acc))
	return history
 
def get_data(train_ds, valid_ds, bs):
	return (
		DataLoader(train_ds, batch_size=bs, shuffle=True),
		DataLoader(valid_ds, batch_size=bs * 2),
	)
   
def loss_batch(model, loss_func, xb, yb, opt=None):
	"""
	creat a loss_batch function that will calculate the loss for a given batch
	is an optimizer is supplied use backpropagation to train the net
	if not dont
	We expect model to be the VQC model and it doesn't support batch forwarding
	"""
	def closure():
		opt.zero_grad()
		loss = loss_func(model, xb, yb)
		loss.backward()
		return loss
		
	try:
		if type(model) == Autoencoder:
			loss = loss_func(model(xb, yb), yb)			# model() is shorthand for model.forward()
		else:
			loss = loss_func(model, xb, yb)
	except ValueError as e:
		print("ValueError in loss_batch")
		print(e)
		loss = loss_func(torch.stack([model.forward(item) for item in xb]), yb)
 
	if opt is not None:	
		#print("Params:", end=' ')
		#print([param.data for param in model.parameters()])
				
		#print("Grad Data:", end=' ')
		#print([(param.grad.data if param.grad is not None else None) for param in model.parameters()])
		
		opt.zero_grad()
		loss.backward()
		if type(opt) == torch.optim.LBFGS:
			opt.step(closure)
		else:
			opt.step()
	return loss.item(), len(xb)
	
def accuracy(model, loss_func, xb, yb):
	"""
	defines the accuracy between an output of the net and the labels yb
	In the future I intend for accuracy to be a method of the model
	"""
	fitters1x = np.array([GaussianFit, SumGaussianFit, StepFit, SumStepFit])
	fitters2x = np.array([StepFit2x, GaussianFit2x, SumStepFit2x])
	typeModel1x = np.array([type(model)]*len(fitters1x))
	typeModel2x = np.array([type(model)]*len(fitters2x))
	if np.equal(typeModel1x, fitters1x).any():
		avgAccuracy = __accuracyIntegrate1x(model, xb, yb)
		return avgAccuracy, len(xb)
	elif np.equal(typeModel2x, fitters2x).any():
		# I can use 1x method for accuracy because ds
		# has flattened xb and yb to 1d
		avgAccuracy = __accuracyIntegrate1x(model, xb, yb)
		return avgAccuracy, len(xb)
	else:
		print("Accuracy for model not implemented")
		return 0, len(xb)
		
def __accuracyIntegrate1x(model, xb, yb):
	integral = 0.0
	for i in range(len(xb)):
		y = model.getField(xb[i])
		if yb[i] > 0.01:
			if yb[i] > y:
				pointAccuracy = 1.0 - (yb[i]-y)/yb[i]
			else:
				pointAccuracy = 1.0 - (y-yb[i])/y
			integral = integral + pointAccuracy
		else:
			integral = integral + 1.0
	avgAccuracy = integral/float(len(xb))
	return avgAccuracy
	
def plot_acc_loss(history, strList=""):
	plt.style.use("seaborn")
	fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 9))

	ax1.plot(history["train_accuracy"], "-og", label="Training Accuracy")
	ax1.plot(history["valid_accuracy"], "-ob", label="Validating Accuracy")
	
	ax1.set_ylabel("Accuracy")
	ax1.set_ylim([0, 1])
	ax1.set_xlabel("Epoch")
	ax1.legend()

	ax2.plot(history["train_loss"], "-og", label="Training Loss")
	ax2.plot(history["valid_loss"], "-ob", label="Validating Loss")
	
	ax2.set_ylabel("Loss")
	#ax2.set_ylim(top=2.5)
	ax2.set_xlabel("Epoch")
	ax2.legend()
		
	plt.tight_layout()
	#plt.show()
	filename = "classifier_blobs_net_dump"
	for str in strList:
		filename += "_"
		filename += str
	filename = filename + ".png"
	plt.savefig(filename)
	plt.close()
	return
	
def checkFit(hist, tol):
	print("Checking loss %f against tolerance %f" % (hist["valid_loss"][-1], tol))
	if hist["valid_loss"][-1] < tol:
		return True
	else:
		print("Fit doesn't meet error tolerance %f" % tol)
		return False