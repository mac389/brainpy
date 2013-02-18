from XKCD import *
from numpy import *
from matplotlib.pyplot import *

from matplotlib import rcParams
from numpy.random import random_sample

rcParams['font.family']='Comic Sans MS'

from scipy.signal import square


#Measurement schematic
'''
data = linspace(1,8)**2
data += 4*random_sample(size=data.shape)
ax = axes()
ax.plot(data)
ax.set_ylabel('Power')
ax.set_xlabel('Voltage (V)')
ax.set_title('Effect of Current on Power for a Frequency Band')
XKCDify(ax,xaxis_arrow='+-',yaxis_arrow='+-',expand_axes=True)
savefig('changizi_eg.png',dpi=300)
'''

#Parameter space
'''
data = random_sample(size=(2,10))
ax = axes()
ax.scatter(data[0,:],data[1,:],s=20)
ax.set_xlabel('Frequency (Hz)',fontsize=20)
ax.set_ylabel('Volts (V)')
ax.set_title('Stimulation Parameters')
XKCDify(ax,xaxis_arrow='+-',yaxis_arrow='+-',expand_axes=True)
savefig('changizi_paramspace.eps',dpi=300)
'''

noise_gain = 0.1
one = array([0,0,0,1,1,1,0,0,0])
four =2*one+ noise_gain*random_sample(size=one.shape)
eight = 2.5*one+noise_gain*random_sample(size=one.shape)
one += noise_gain*random_sample(size=one.shape)

'''
ax = axes()
ax.plot(r_[one,four,eight])
hold(True)
ax.plot(3.1 + r_[eight,four,one],'b')
ax.plot(6 + r_[four,one,eight],'b')
ax.set_xlabel('Order',fontsize=20)
ax.set_ylabel('Volts (V)')
ax.set_title('Stimulation Parameters')
ax.set_ylim(0,8)
ax.get_yaxis().set_visible(False)
XKCDify(ax,xaxis_arrow='+-',yaxis_arrow='+-',expand_axes=True)
savefig('changizi_vstep.eps',dpi=300)
'''

ax = axes()
ax.set_title('Tenure')
XKCDify(ax)
savefig('tenure.eps',dpi=300)

