from brian.library.random_processes import *
from brian import *
from matplotlib import rcParams
from numpy.random import randint, shuffle

rcParams['text.usetex'] = True

tau = 10*ms
C = 1*uF
Vt = 10*mV
eqs=Equations('dV/dt=-V/tau+xi*mV/tau**.5+I/C: volt')
eqs+=OrnsteinUhlenbeck('I',mu=2*nA,sigma=2*nA,tau=10*ms)

brain_size=1000
brain = NeuronGroup(N=brain_size, model=eqs, threshold=Vt,reset=0*mV)
sparse = 0.4 #(Inspired by Reyes Science paper about stronlgy connected networks able to generate reallyl ow correlations)

trace = StateMonitor(brain,'V',record=True)
pop = PopulationRateMonitor(brain, bin=10*ms)
spiker = SpikeMonitor(brain,record=True)
current = StateMonitor(brain,'I',record=True)

synapses = Connection(brain,brain,'I')
synapses.connect_random(brain,brain,weight=10*nA*(tau/ms)/(0.5*sparse*brain_size),sparseness = sparse)

stimulus_indices = array(list(set(randint(brain_size,size=(int(floor(brain_size))/4,)))))
shuffle(stimulus_indices)

duration= 2*second
f=10*Hz
@network_operation
def current_pulse():
	brain.I[stimulus_indices]+=(((defaultclock.t/second)>1)*nA*(5+5*sin(2*pi*defaultclock.t*f)))

run(duration,report='text')
run(duration,report='text')

figure()
subplot(211)
plot(trace.times/ms,trace[stimulus_indices[0]]/mV)
ylabel('Voltage (mV)')
axhline(y=Vt/mV,color='r',linewidth=3)
subplot(212)
plot(pop.times/ms,pop.rate/Hz)
xlabel('Time (ms)')
ylabel('Firing rate (spikes/second)')

figure()
plot(current.times/ms,current[stimulus_indices[0]]/uA)
ylabel('Current $(\mu A)$')
xlabel('Time (ms)')

'''
figure()
title('Connection matrix, weights in nA')
imshow(synapses.W.todense()/nA,interpolation='nearest',aspect='auto');colorbar()
'''
show()

