from numpy import *
from numpy.random import randint
from scipy import *
import matplotlib.pyplot as plt
import neuroTools as tech
from matplotlib import rcParams, text


#--------------------------------------Plotting Options--------------------------------------------
params = {'backend': 'ps',
          'axes.labelsize': 12,
          'text.fontsize': 20,
          'legend.fontsize': 20,
          'xtick.fontsize': 20,
          'ytick.fontsize': 20,
          'xlabel.fontsize':20,
          'ylabel.fontsize':20,
          'axes.labelweight':'bold',
          'axes.linewidth':3,
          'font.size': 20,
          'text.usetex': True}
rcParams.update(params)
#-------------------------------------------------------------------------------------------------	
neuron_count = 100

def rectify(data): return data*(data>0) #Works for both numbers and vectors

def F(activity, amplitude=150., steepness = 150., offset=20. ): #Activation fuction
	return amplitude*rectify(tanh((activity+offset)/steepness))
	
sparseness = 1/float(neuron_count)
eigenvalue = 1.25

memories = eye(neuron_count)
uniform_inhibition = 1/(sparseness)*ones((neuron_count,neuron_count))
weights = sum(array([outer(memory-sparseness, memory-sparseness) for memory in memories]), axis=0)
weights -= uniform_inhibition
weights *= eigenvalue/(sparseness*neuron_count*(1-sparseness))

iterations = 10000

record = zeros((neuron_count,iterations))
record[:,0] = dot(weights,0.5*memories[:,2] + 0.5*randint(2,size=memories[:,2].shape)) #Study behavior to a random memory
overlap = zeros(iterations,)
for iteration in range(1,iterations):
	record[:,iteration] = F(dot(weights,record[:,iteration-1]))
	overlap[iteration] = dot(record[:,iteration], memories[2])/float(neuron_count)

fig1 = plt.figure()
evol = fig1.add_subplot(211)
cevol = evol.imshow(record,aspect='auto',interpolation='nearest', cmap = plt.cm.binary)
fig1.colorbar(cevol)
similar = fig1.add_subplot(212)
similar.plot(overlap)
 
fig = plt.figure()
plt.subplots_adjust(hspace=0.3)
mems = fig.add_subplot(211)
mem_ax = mems.imshow(memories,aspect='auto',interpolation='nearest', cmap=plt.cm.binary)
fig.colorbar(mem_ax)
mems.set_xlabel(r'Memory $\rightarrow$', fontsize=20)
mems.set_ylabel(r'Neurons $\rightarrow$', fontsize=20)
connections = fig.add_subplot(212)
cax = connections.imshow(weights,aspect='auto',interpolation='nearest', cmap=plt.cm.binary)
fig.colorbar(cax)
connections.set_xlabel(r'Neurons $\rightarrow$', fontsize=20)
connections.set_ylabel(r'Neurons $\rightarrow$', fontsize=20)
plt.show()
