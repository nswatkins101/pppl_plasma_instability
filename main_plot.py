from pylab import *
import os
import math
import copy
import postgkyl as pg
import numpy as np
import matplotlib.pyplot as plt
style.use("../postgkyl.mplstyle") 

from plot_data import *

# Physical constants (using normalized code units).
K_B = 1.0
EPSILON_0 = 1.0 													# Permittivity of free space.
MU_0 = 1.0																# Permiability of free space.
LIGHT_SPEED = 1/math.sqrt(MU_0*EPSILON_0)	# Speed of light.

ELC_MASS = 1.0 													# Electron mass.
ELC_CHARGE = -1.0												# Electron charge.
N_0 = 1.0																# initial reference density

figure, axes = subplots(1,1)
mySys = System(fileloc="hotWater", pMinFrames=59, pMaxFrames=59, pMass=ELC_MASS, pCharge=ELC_CHARGE, pN0=1, pVth=1)

#plotFigFromData(mySys.getField("phi", 1), figure, axes)
#plotFigFromData(mySys.getField("hamiltonian", 60), figure, axes, contourFlag=True)
#plotFigFromData(mySys.getField("elc", 60), figure, axes)
#Xf, Exf = importDGk("hotWater_field_60.bp")
#plotFigFromData((Xf[0],Exf), figure, axes)
#plotFigFromData(calcFFT(_importDGk("hotWater_field_59.bp"), rtn="real"), figure, axes)
#plotFigFromData(calcFFT(mySys.getField("phi", 59, False), rtn="real"), figure, axes)

#plotFigFromData(calcFFT(importDGk("hotWater_field_60.bp"), rtn="imaginary"), figure, axes)
#show()

#for fr in range(200):
#	plotFigFromData(mySys.getField("hamiltonian", fr), filename=mySys.FILE_TEMPLATE%("h",fr,"png"), saveFlag=True)


for fr in range(200):
	close(figure)
	figure, axes = subplots(1,1)
	plotFigFromData(calcFFT(_importDGk("hotWater_field_%d.bp" % fr), rtn="real"), figure, axes, filename=mySys.FILE_TEMPLATE%("Ex_fft",fr,"png"), saveFlag=True)
#	plotFigFromData(mySys.getField("hamiltonian", fr), figure, axes, contourFlag=True)
#	plotFigFromData(mySys.getField("elc", fr), figure, axes, filename=mySys.FILE_TEMPLATE%("overlay",fr,"png"), saveFlag=True)




