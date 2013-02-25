from numpy import *
from numpy.random import randint, rand
from scipy import *
import matplotlib.pyplot as plt
import neuroTools as tech
from matplotlib import rcParams, text
from numpy.linalg import norm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.signal import detrend

#-------------------------------------------------------------------------------------------------	
''' Module Description
	This Python module implements an autoassociative network to explore, for my SfN 2012, abstract
	how the energy landscape, information content of memories, and discriminability change in 
	acutely intoxicated or chronically addicted states as compared with naive controls. 
'''

#-------------------------------------------------------------------------------------------------	
##################################################################################################
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
##################################################################################################
#--------------------------------------Helper Functions--------------------------------------------
def rectify(data): return data*(data>0)

def F(activity, amplitude=1., steepness = 1., offset=.15 ): #Activation fuction
	return amplitude*rectify(tanh((activity+offset)/steepness))
#-------------------------------------------------------------------------------------------------	
##################################################################################################
#---------------------------Network Parameters (User Defined)--------------------------------------
neuron_count = 30
eigenvalue = 1.25
iterations = 1000
sparseness = 0.2
#-------------------------------------------------------------------------------------------------	
##################################################################################################
#---------------------------Network Parameters (Calculated)---------------------------------------
memories = rand(neuron_count,neuron_count)
memories[memories<sparseness]=1
memories[memories!=1]=0
#should check whether some memories are the same later (or just use a canonical memory file)

uniform_inhibition = 1/(sparseness*neuron_count)*ones((neuron_count,neuron_count))
c = 1
weights = sum(array([outer(memory-c*sparseness, memory-c*sparseness) for memory in memories]),axis=2)
weights -= uniform_inhibition
weights *= eigenvalue/(sparseness*neuron_count*(1-sparseness)*c**2)
#-------------------------------------------------------------------------------------------------	
##################################################################################################
#---------------------------Simulation Control & Storagae-----------------------------------------
record = zeros((neuron_count,iterations))
#-------------------------------------------------------------------------------------------------	
##################################################################################################
#---------------------------Initial Conditions of Simulation--------------------------------------
record[:,0] = dot(weights,memories[2]) #Study behavior to a random memory
#-------------------------------------------------------------------------------------------------	
##################################################################################################
#------------------------------------Simulation---------------------------------------------------
overlap = zeros(iterations,)
for iteration in range(1,iterations):
	record[:,iteration] = F(dot(weights,record[:,iteration-1]))
	overlap[iteration] = dot(record[:,iteration],memories[2])/(norm(record[:,iteration])*norm(memories[2]))
#-------------------------------------------------------------------------------------------------	
##################################################################################################
#------------------------------------Post-processing----------------------------------------------
#Need to vectorize
energy = detrend(record, axis=1)
print record.shape
print energy.shape
energies = array([0.5*outer(erg,erg)*weights for erg in transpose(energy)])
#-------------------------------------------------------------------------------------------------	
##################################################################################################
#------------------------------------Visualization------------------------------------------------
#Network Performance
fig1 = plt.figure()
evol = fig1.add_subplot(211)
cax = evol.imshow(record,aspect='auto',interpolation='nearest', cmap = plt.cm.binary)
plt.colorbar(cax)
similar = fig1.add_subplot(212)
similar.plot(overlap)
box =similar.get_position()
similar.set_position([box.x0, box.y0, box.width*.8, box.height])

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

erg_plot = plt.figure() #Ideally this would eventually be a movie
erg_ax = erg_plot.add_subplot(111)
erg_ax.plot(average(energies,axis=2))
plt.show()
