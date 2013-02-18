from numpy import *
from numpy.random import randint
from scipy import *
import matplotlib.pyplot as plt
import neuroTools as tech

#-------------------------------------------------------------------------------------
######################################################################################
''' Simulation Description

	Hopfield associative network
	sgn is activation function
'''
######################################################################################
#-----------------------------Network Description-------------------------------------
neuron_count = 100
memories = 1-2*randint(2, size=(neuron_count,neuron_count))
weights = (1-eye(neuron_count))*(dot(memories,memories))

iterations = 100
record = zeros((neuron_count,iterations))
record[:,0] = sign(dot(weights,memories[2])) #Study behavior to a random memory
overlap = zeros(iterations,)
for iteration in range(1,iterations):
	record[:,iteration] = sign(dot(weights,record[:,iteration-1]))
	overlap[iteration] = dot(record[:,iteration], memories[2])/float(neuron_count)

'''
bg = plt.figure()
mems = bg.add_subplot(211)
mems.imshow(memories,aspect='auto',interpolation='nearest')
w = bg.add_subplot(212)
w.imshow(weights,aspect='auto',interpolation='nearest')
'''

fig = plt.figure()
evol = fig.add_subplot(211)
cax = evol.imshow(record,aspect='auto',interpolation='nearest', cmap = plt.cm.binary)
similar = fig.add_subplot(212)
similar.plot(overlap)
plt.savefig('hopfield',dpi=200)