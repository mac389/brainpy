import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import randint

#Generate fake data
neuronCount = 2
duration = 10
firingRate = 10
data = rand(neuronCount,duration)
data[data>firingRate*dt]=1
data[data!=1]=0
x_isi = np.zeros(data.shape)
for neuron,trace in enumerate(data):
	spikeTimes = (trace==1).nonzero()[0]
	for time in range(len(trace)):
		if np.nonzero(spikeTimes>time) and np.nonzero(spikeTimes<time):
			x_isi[neuron,time] = min(spikeTimes>time)-max(spikeTimes<time)
print data
print x_isi
		
	
