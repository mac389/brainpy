import sortUtils as tech
import cPickle
from scipy.io import loadmat

filename = '/Volumes/My Book/Rat/010113/continuous/shank3.waveforms'
import numpy as np


data = loadmat('/Volumes/My Book/shank3_pca_1.mat')
clusters = cPickle.load(open('/Volumes/My Book/shank3_clus_res.pkl','rb'))
eiglist=None
nclus = None
spiketimes = None
#data = np.fromfile(filename,'float64')
#data = np.reshape(data,(600,-1)).transpose() 
#print data.shape
tech.visualize(data['proj'].transpose(),clusters,spiketimes,data['eigvals'].transpose(),nclus,savename='shank32', multi=True)
