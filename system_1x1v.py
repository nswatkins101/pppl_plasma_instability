from pylab import *
import os
import math
import copy
import postgkyl as pg
import numpy as np
import matplotlib.pyplot as plt
style.use("postgkyl.mplstyle") 

# Physical constants (using normalized code units).
K_B = 1.0
EPSILON_0 = 1.0 													# Permittivity of free space.
MU_0 = 1.0																# Permiability of free space.
LIGHT_SPEED = 1/math.sqrt(MU_0*EPSILON_0)	# Speed of light.

ELC_MASS = 1.0 													# Electron mass.
ELC_CHARGE = -1.0												# Electron charge.
N_0 = 1.0																# initial reference density

mkDirSuccess = False
dirNum = 0
filepath = ""

def importDGk(filename):
		d = pg.GData(filename)
		dg = pg.GInterpModal(d,2,"ms")
		XX, fv = dg.interpolate()
		#center the grid values, from interpolate the lengths of XX are not in line with fv, off by one so generate a new array from the average of the two options to shorten XX[i]
		for d in range(len(XX)):
			XX[d] = 0.5*(XX[d][:-1] + XX[d][1:])
		print("Loaded %s from importDGk" % filename)
		return XX, fv													# returns numpy ndarrary
	
def _importDGk(filename):
		d = pg.GData(filename)
		dg = pg.GInterpModal(d,2,"ms")
		XX, fv = dg.interpolate()
		#center the grid values, from interpolate the lengths of XX are not in line with fv, off by one so generate a new array from the average of the two options to shorten XX[i]
		for d in range(len(XX)):
			XX[d] = 0.5*(XX[d][:-1] + XX[d][1:])
		print("Loaded %s from importDGk" % filename)
		return XX[0], fv
		
def calcFFT(args, rtn=""):
	q = args[-1]
	X = args[0]
	Nx = len(X)
	dx = X[1] - X[0]
	print("direct lattice spacing is %f" % (dx))
	#S = np.fft.fftfreq(Nx, dx)
	#print(S)
	#q_fft = np.zeros(q.shape[0])
	#q_fft = np.fft.fft(q, norm="ortho")
	#if rtn=="real":
		#return S, np.real(q_fft)
	#elif rtn=="imaginary":
		#return S, np.imag(q_fft)
	#else:
		#return S, q_fft
	S = np.fft.rfftfreq(Nx,dx)
	q_fft = np.fft.rfft(transpose(q), norm="ortho")
	print(q.shape)
	print(q_fft.shape)
	return S, transpose(np.real(q_fft))
		
class Particle:
	def __init__(self,pMass, pCharge):
		self.mass = pMass
		self.charge = pCharge
	
class System:
	def __init__(self, fileloc="", pMinFrames=0, pMaxFrames=0, pMass=1, pCharge=1, pN0=1, pVth=1):
		self.testParticle = Particle(pMass, pCharge)
		self.FILE_TEMPLATE = fileloc+"_%s_%d.%s"
		self.MAX_FRAMES = pMaxFrames
		self.MIN_FRAMES = pMinFrames
		self.W_PE = math.sqrt(pCharge**2*pN0/(EPSILON_0*pMass))
		self.V_TH = pVth														# thermal speed with root(2), note with current params, vt = c
		self.LAMBDA_D = self.V_TH / self.W_PE / math.sqrt(2)			# debye length
		self.E_TH = 3/2*self.testParticle.mass*self.V_TH**2

		self.__XV = [None]*(self.MAX_FRAMES-self.MIN_FRAMES+1)
		self.__fv = [None]*(self.MAX_FRAMES-self.MIN_FRAMES+1)
		self.__X = [None]*(self.MAX_FRAMES-self.MIN_FRAMES+1)
		self.__phi = [None]*(self.MAX_FRAMES-self.MIN_FRAMES+1)
		self.__hamiltonian = [None]*(self.MAX_FRAMES-self.MIN_FRAMES+1)
		self.__temperature = [None]*(self.MAX_FRAMES-self.MIN_FRAMES+1)
		print("Created System members:")
		print("length for \n\t XV: %d \n\t fv: %d \n\t X: %d \n\t phi: %d \n\t hamiltonian: %d \n\t temperature: %d" % (len(self.__XV), len(self.__fv), len(self.__X), len(self.__phi), len(self.__hamiltonian), len(self.__temperature)))
		for n in range(self.MAX_FRAMES-self.MIN_FRAMES+1):
			fr = n+self.MIN_FRAMES
			self.__XV[n], self.__fv[n] = copy.deepcopy(self.importField("elc", fr))
			self.__X[n], self.__phi[n] = copy.deepcopy(self.calcPhi(fr))
			self.__hamiltonian[n] = copy.deepcopy((self.calcH(fr))[1])
			self.__temperature[n] = self.calcTemp(fr)	# no need for deep copy because calcTemp returns a scalar
		
	def getField(self, field, fr, dimless=True):
		n = fr - self.MIN_FRAMES
		if field == "hamiltonian":
			print("Type of Hamiltonian: %s" % (type(self.__hamiltonian)))
			print("Type of Hamiltonian[%d]: %s" % (n, type(self.__hamiltonian[n])))
			print("Returning dimensionless H with shape %s" % str(self.__hamiltonian[n].shape))
			if dimless:
				return self.__XV[n][0], self.__XV[n][1], self.__hamiltonian[n]
			else:
				return self.__XV[n][0], self.__XV[n][1], self.__hamiltonian[n]
		elif field == "phi":
			print("Type of Phi: %s" % (type(self.__phi)))
			print("Type of Phi[%d]: %s" % (n, type(self.__phi[n])))
			print("Returning dimensionless Phi with shape %s" % str(self.__phi[n].shape))
			if dimless:
				return self.__X[n][0]/self.LAMBDA_D, self.__phi[n]/ (self.E_TH/self.testParticle.charge)
			else:
				return self.__X[n][0], self.__phi[n]
		elif field == "elc":
			print("Type of fv: %s" % (type(self.__fv)))
			print("Type of fv[%d]: %s" % (n, type(self.__fv[n])))
			print("Returning dimensionless fv with shape %s" % str(self.__fv[n].shape))
			if dimless:
				return self.__XV[n][0]/self.LAMBDA_D, self.__XV[n][1]/self.V_TH, self.__fv[n]
			else:
				return self.__XV[n][0], self.__XV[n][1], self.__fv[n]
		elif field == "temperature":
			print("Type of temperature: %s" % (type(self.__temperature)))
			print("Type of temperature[%d]: %s" % (n, type(self.__temperature[n])))
			if dimless:
				return self.__temperature[n]/(self.E_TH/K_B)
			else:
				return self.__temperature[n]
		else:
			print("ERROR: field %s not recognized" % field)
			return None
			
	def importField(self, field, fr):
		return importDGk(self.FILE_TEMPLATE % (field,fr, "bp"))
		
	def calcTemp(self, fr): 
		# loading second moment of distribution
		X, sqV = self.importField("elc_M2", fr)
		# integrating over domain
		dx = X[0][1] - X[0][0]
		integral = 0
		for pos in X[0]:
			integral += sqV*dx
		# dividing by length of domain for avg KE
		avgKE = 0.5*self.testParticle.mass*integral/(X[0][-1]-X[0][0])
		# using equipartition theorem to derive temperature from avg KE
		return 2/3*avgKE/K_B
	
	def calcPhi(self, fr):
		# loading field data and then integrating to determine the potential phi
		X_phi, Ex = self.importField("field", fr)
		# let ref be the voltage reference, the left side of sim box
		ref = X_phi[0][0]
			
		dx = math.fabs(X_phi[0][1] - X_phi[0][0])								# X_phi is of the format [array([...])], hence acess element i with X_phi[0][i]
		phi = np.zeros(Ex.shape[0])							# selecting Ex component with [0]
		for j in range(0, Ex.shape[0]):
				phi[j] = -np.sum(Ex[0:j, 0])*dx				# E = -grad(phi)
		return X_phi, phi			
		
	def calcH(self, fr):
		# load fields and calculate potentials
		X_phi, U = self.calcPhi(fr)
		XV, fv = self.importField("elc", fr)
		XX_h = [X_phi[0], XV[1]]
		H = np.zeros(fv.shape)
		for i in range(fv.shape[0]):
			for j in range(fv.shape[1]):
				H[i][j] = self.testParticle.charge*U[i] + 0.5*self.testParticle.mass*XV[1][j]**2		# single particle Hamiltonian, electron
		return XX_h, H