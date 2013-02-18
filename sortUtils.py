from numpy import *
from scipy import *

import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from os.path import isfile, splitext

from scipy.stats import scoreatpercentile, percentileofscore
from time import time

from brian import *
from brian.library.electrophysiology import *

import Pycluster as pc
from itertools import product

def partition(filename, filtered=True):
	partitions = ['before','during','after'] #It would be great if these came out naturally
	data = (read_DDT(filename),300,7000,20000) if filtered else read_DDT(filename) #Default is to read the unfiltered trace
	chunkLength = len(data)/len(partitions) #Deliberately dividing by integer
	
	name,_ = splitext(filename)
	for chunk in range(len(partitions)):
		start = i*chunkLength
		stop = (i+1)*chunkLength
		savetxt(name+'.'+partitions[chunk],data[start:stop],delimiter='\t')
	print 'Everything saved'
def get_waveforms(data,spiketimes,lookback=100,lookahead=100):
	answer = zeros((len(spiketimes),(lookback+lookahead)))
	duration = len(data)
	for i in xrange(len(spiketimes)):
		if (spiketimes[i] - lookback) > 0 and (spiketimes[i] + lookahead) < duration:
			answer[i,:] = data[(spiketimes[i]-lookback):(spiketimes[i]+lookback)]
	return answer
	
def detect_spikes(data,threshold):
	print 'Threshold is %0.2f' % float(threshold),'mV'	
	return spike_peaks(data,vc=threshold)

def threshold(data):
	return median(absolute(data-median(data)))

def save_filtered_trace(filename,lowcut=300,highcut=7000,sampling_rate=20000):
	#Assume that filename points to unfiltered data stored in a binary DDT format
	filtered_data = butter_bandpass_filter(read_DDT(filename),lowcut,highcut,sampling_rate)
	name,_ = splitext(filename) #The second tuple argument receives the period.
	savetxt(name+'.filtered_traces',filtered_data,delimiter='\t')
	del filtered_data
	print 'Saved %s to file' % name

def princomp(A,numpc=3):
	# computing eigenvalues and eigenvectors of covariance matrix
	M = (A-mean(A.T,axis=1)).T # subtract the mean (along columns)
	[latent,coeff] = linalg.eig(cov(M))
	p = size(coeff,axis=1)
	idx = argsort(latent) # sorting the eigenvalues
	idx = idx[::-1]       # in ascending order
	# sorting eigenvectors according to the sorted eigenvalues
	coeff = coeff[:,idx]
	latent = latent[idx] # sorting eigenvalues
	if numpc < p or numpc >= 0:
		coeff = coeff[:,range(numpc)] # cutting some PCs
	score = dot(coeff.T,M) # projection of the data in the new space
	return coeff,score,latent

def toxy(data):
	x,y=[],[]
	[(x.append(a[0]), y.append(a[1])) for a in data]
	return x,y

def to_full_matrix(til_data):
	rnk = len(til_data)
	answer = zeros((rnk,rnk))
	for idx in xrange(rnk):
		span = len(til_data[idx])
		answer[idx,:span] = til_data[idx]
	return answer
		

def cluster(data, threshold = 0.6):
	length = len(data)
	nclus = 2
	nclusmax=8
	res = {}
	sil_co_one = 1
	sil_co = [1]
	#Assume 
	while sil_co_one > threshold and nclus < nclusmax:
		clustermap,_,_ = pc.kcluster(data,nclusters=nclus,npass=50)
		centroids,_ = pc.clustercentroids(data,clusterid=clustermap)
		
		clusterx,clustery = toxy(centroids)
		
		'''
		figure()
		scatter(data[:,0],data[:,1],marker='+')
		hold(True)
		scatter(clusterx,clustery,c='r',s=40)
		show()
		'''
		m = to_full_matrix(pc.distancematrix(data))
		
		
		#Find the masses of all clusters
		mass = zeros(nclus)
		for c in clustermap:
			mass[c] += 1
	
		#Create a matrix for individual silhouette coefficients
		sil = zeros((len(data),nclus))

		#Evaluate the distance for all pairs of points		
		for i in xrange(0,length):
			for j in range(i+1,length):
				d = m[j][i]
				
				sil[i, clustermap[j] ] += d
				sil[j, clustermap[i] ] += d
		
		#Average over cluster
		for i in range(0,len(data)):
			sil[i,:] /= mass
			
		#Evaluate the silhouette coefficient
		s = 0
		for i in xrange(0,length):
			c = clustermap[i]
			a = sil[i,c] 
			b = min( sil[i, range(0,c) + range(c+1,nclus)])
			si = (b-a)/max(b,a) #silhouette coefficient of point i
			s+=si
					
		nclus += 1
		sil_co.append( s/length)
		sil_co_one = s/length
		res[str(nclus-2)] = {'clustermap':clustermap,
							'centroids':centroids,
							 'distances':array(m),
							 'mass':mass,
							 'silhouettes':sil_co}
	return res

def extract_waveforms(data,spiketimes,lookahead,lookback,onsets):	
	STA = spike_shape(data,onsets=onsets,before=lookback,after=lookahead)
	slope = slope_threshold(data,onsets=onsets,T=int(5*ms/defaultclock.dt))
	return (STA,slope)

def butter_bandpass(lowcut,highcut,fs,order=2):
	nyq = 0.5*fs
	low = lowcut/nyq
	high = highcut/nyq
	
	b,a = butter(order, [low, high], btype='band')
	return b,a

def butter_bandpass_filter(data,*args, **kwargs):
	b,a = butter_bandpass(*args, **kwargs)
	return filtfilt(b,a,data) 

def read_DDT(filename,OFFSET=432):
	with open(filename,'rb') as stream:
		stream.seek(OFFSET)
		return fromfile(stream,dtype='int16')
