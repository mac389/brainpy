import cPickle

import numpy as np
import sortUtils as tech

from sys import argv
from scipy.io import savemat

recording = argv[1]
channel = int(argv[2])
'''
basepath = '/Volumes/My Book/Rat'
localpath = basepath + '/' +recording + '/' + 'continuous'+ '/' + 'ch{0:03}'.format(channel)
'''
localpath = 'ch{0:03}'.format(channel)

WAVEFORMS = localpath+'.waveforms'
data = np.reshape(np.fromfile(WAVEFORMS,'float64'),(200,-1)).transpose()

''' WAVEFORMS now oriented as

		   ------> Time
		  |
		  |
		  |
		  V
		Waveforms
'''  

def reduce_dimensionality(data,numpc=50):
	#Assume waveforms are passed with each waveform being a row
	eigVecs,proj,eigVals = tech.princomp(data,numpc=numpc)
	print 'Doing PCA'       
	
	'''Eigvecs arranged as         
				
					 --------> Eigenvector
					|
	Coefficients	|
					|
					V
	'''
	PCA = localpath+'.pca'
	print eigVals[:numpc].sum()/eigVals.sum()
	mdict={'eigvectors':eigVecs,'projections':proj,'eigenvalues':eigVecs}
	savemat(PCA, mdict=mdict, oned_as='column')
	print 'Saved a copy of analysis to \n\t%s' % localpath
	return proj.transpose()
	
def sort(projections):
	print 'Clustering'
	(models,silhouettes) = tech.cluster(projections, preprocess=False)
	#SK expects the same orientation as PCA
	CLUSTER = localpath+'clusterpkl'
	res = {'models':models,'silhouettes':silhouettes}
	cPickle.dump(res,open(CLUSTER,'wb'))

sort(reduce_dimensionality(data))