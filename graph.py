from pylab import *
import os
import math
import copy
import postgkyl as pg
import numpy as np
import matplotlib.pyplot as plt
style.use("postgkyl.mplstyle") 


	
class Graph:
	def __init__(self, fileloc=""):
		