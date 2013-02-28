from numpy import *
from scipy import *

import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from os.path import isfile, splitext
import neuroTools as postdoc

from scipy.stats import scoreatpercentile, percentileofscore
from time import time

from brian import *
from brian.library.electrophysiology import *

from itertools import product

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from time import time
from matplotlib import rcParams

from numpy.random import random

rcParams['text.usetex'] = True
def update_progress(progress):
    print '\r[{0}] {1}%'.format('#'*(progress/10), progress)

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
	
def NEO(data):
	#From Kim and Kim (2000), an IEEE paper
	answer = data*data
	answer -= (roll(data,1)*roll(data,-1))
	return r_[0,answer[1:-1],0]
	
def get_waveforms(data,spiketimes,lookback=100,lookahead=100):
		offsets = arange(-lookback,lookahead)
		indices = spiketimes + offsets[:,None]
		ret = take(data,indices,mode='clip')
		ret[:,spiketimes<lookback]=0
		ret[:,spiketimes+lookahead>=len(data)]=0
		return ret

def get_channel_id(filename):
	name,_ = splitext(filename)
	return str(int(name[-3:]))
	
'''
def get_waveforms(data,spiketimes,lookback=100,lookahead=100,skip=1000, report=True):
	answer = zeros((1+len(spiketimes)/skip,(lookback+lookahead)))
	duration = len(data)
	for z in xrange(0,len(spiketimes),skip):
		i=z/skip
		if i%100 and report:
			update_progress(100*int(z/float(len(spiketimes))))
		if (spiketimes[i] - lookback) > 0 and (spiketimes[i] + lookahead) < duration:
			answer[i,:] = data[(spiketimes[i]-lookback):(spiketimes[i]+lookback)]
	return answer
'''

def add_noise(data,amplitude=10,shape=(200,3000)):
	row,col = data.shape
	res = zeros((row+shape[0],col+shape[1]))
	res[-shape[0]:-shape[1]:] = amplitude*random(size=shape)


def switch_type(inputfile,desired):
	name,_ = splitext(inputfile)
	return name+'.'+desired
	
def detect_spikes(data,threshold, fast=True, refractory=10):
	print 'Threshold is %0.2f' % float(threshold),'mV'	
	if fast:
		crossings = where(data>threshold)[0]
		intervals= (roll(crossings,1)+ roll(crossings,-1))-2*crossings
		return crossings[intervals>refractory]
	else:
		return spike_peaks(data,vc=threshold)
		
def threshold(data):
	return 5*median(absolute(data-median(data)))

def save_filtered_trace(filename,lowcut=300,highcut=7000,sampling_rate=20000,show=False):
	#Assume that filename points to unfiltered data stored in a binary DDT format
	filtered_data = butter_bandpass_filter(read_DDT(filename),lowcut,highcut,sampling_rate)
	extension = '.filtered'
	name,_ = splitext(filename)
	name += extension
	filtered_data.tofile(name)
	print 'Saved %s to file' % name
	if show:
		return filtered_data

def princomp(A,numpc=3):
	print A.shape
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

def find_sing_val_cutoff(data,cutoff=0.95):
	data /= data.sum()
	return where(cumsum(data)>0.95)[0][0]

def scree(eigVals,npc=50):
	#Assume the list is all of the eigenvalues
	rel = cumsum(eigVals)/eigVals.sum()
	x = arange(len(rel))+1
	
	fig = plt.figure()
	ax = plt.add_subplot(111)
	ax.bar(x,rel,width=0.5)
	postdoc.adjust_spines(ax,['bottom','left'])
	ax.set_xlabel('Eigenvector')
	ax.set_ylabel('Fraction of variance')
	plt.show()

def cluster(data, threshold = 0.5,method='sk', preprocess=True):
	length = len(data)
	print data.shape
	nclus = 2
	nclusmax=15
	sil = [-1]
	models=[]
	if preprocess==True:
		print 'Preprocessing by scaling each row by its range'
		data /= (amax(data,axis=0)-amin(data,axis=0))[newaxis,:]
		print 'Now to cluster'	
	if method == 'sk':
		print 'Clustering using Scikits K-means implementation'
		print "This option returns a tuple of"
		print "\t\t (kmeans object, silhouette coefficients)"
		while nclus < nclusmax: #average(sil[-1]) < threshold and
			model = KMeans(init='k-means++',n_clusters=nclus) 
			#Assume data is propery preprocessed
			model.fit(data)
			labels = model.labels_
			#<-- can only sample this in chunks of 100
			print data.shape
			print 'Calculating silhouette_score '
			sil.append(silhouette_score(data,labels,metric='euclidean')) 
			models.append(model)
			print 'For %d clusters, the silhouette coefficient is %.03f'%(nclus,sil[-1])
			nclus += 1
		return (models,sil)
	elif method == 'pyclus':
		import Pycluster as pc
		print 'Clustering using the C Clustering library'
		print 'This option returns a dictionary with the distance matrix, silhouettes, and clusterids for each iteration.'
		res = []
		sil_co_one = 1
		sil_co = [1]
		#Assume 
		while sil_co_one > threshold and nclus < nclusmax:
			print 'No. of clus: %d'%nclus
			print 'Before kcluster'
			clustermap,_,_ = pc.kcluster(data,nclusters=nclus,npass=50)
			print 'After kcluster'
			centroids,_ = pc.clustercentroids(data,clusterid=clustermap)
			print 'After centroids'
	
			m = pc.distancematrix(data)
			
			print 'Finding mass'
			#Find the masses of all clusters
			mass = zeros(nclus)
			for c in clustermap:
				mass[c] += 1
		
			#Create a matrix for individual silhouette coefficients
			sil = zeros((len(data),nclus))
			
			print 'Evaluating pairwise distance'
			#Evaluate the distance for all pairs of points		
			for i in xrange(0,length):
				for j in range(i+1,length):
					d = m[j][i]
					
					sil[i, clustermap[j] ] += d
					sil[j, clustermap[i] ] += d
			
			#Average over cluster
			for i in range(0,len(data)):
				sil[i,:] /= mass
			
			print 'Sil co'	
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
			print 'Sil co %.02f'%sil_co_one
			res.append({'clustermap':clustermap,
						'centroids':centroids,
						 'distances':m,
						 'mass':mass,
						 'silhouettes':sil_co})
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

def low_d(u,s,v,cutoff=2):
	sigma = diag(s)
	sigma[cutoff:][cutoff:] = 0
	print u[:,:len(s)].shape
	print sigma.shape
	a_star = dot(u[:,:len(s)],sigma)
	a_star = dot(a_star,v[:,:cutoff])
	return a_star

def visualize(data,clusters,spiketimes=None,eiglist=None,nclus=None,savename='res',multi=False):
	kmeans,silhouettes = clusters
	sil_cos = map(average,silhouettes)
	#pick out the 
	best = kmeans[argmax(sil_cos)-1]
	nclus = best.n_clusters if not nclus else nclus
	fig = plt.figure()
	plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=.97)
	
	#Clusters of waveforms projected onto the first two principal components
	ax = fig.add_subplot(2,2,1)
	ax.set_axis_bgcolor('white')
	colors = ['#4EACC5', '#FF9C34', '#4E9A06']
	labels_ = best.labels_
	centers = best.cluster_centers_
	unique_labels = unique(labels_)
	for n,col in zip(range(nclus),colors):
		my_members = labels_ == n 
		cluster_center = centers[n]
		ax.plot(data[my_members,0],data[my_members,1],'w',markerfacecolor=col,marker='.', markersize=6)
		hold(True)
		ax.plot(cluster_center[0],cluster_center[1],'o',markerfacecolor=col,markeredgecolor='k',markersize=8)
	postdoc.adjust_spines(ax,['bottom','left'])
	ax.set_ylabel('PC2')
	ax.set_xlabel('PC1')
	ax.tick_params(direction='in')
		
	sils = fig.add_subplot(2,2,2)
	sils.set_axis_bgcolor('none')
	line, =sils.plot(range(len(sil_cos))[2:],sil_cos[2:],'--.')
	line.set_clip_on(False)
	sils.tick_params(direction='in')
	sils.axhline(y=0.5,color='r',linestyle='--')
	postdoc.adjust_spines(sils,['bottom','left'])
	sils.set_xticks(range(len(sil_cos))[2:])
	sils.set_yticks([0,1])
	sils.set_ylabel('Silhouette coefficient')
	sils.set_xlabel('Number of clusters')
	
	if spiketimes:
		isi = fig.add_subplot(2,2,4)
		_,_,patches=isi.hist(diff(spiketimes),bins=200,range=(0,1000), histtype='stepfilled')
		postdoc.adjust_spines(isi,['bottom','left'])
		isi.tick_params(direction='in')
		isi.set_axis_bgcolor('none')
		isi.set_ylabel('Count')
		isi.set_xlabel(r'ISI $(ms)$')
		plt.setp(patches,'facecolor',colors[1])
		
		short_isi = fig.add_axes([0.77, 0.26, 0.15, 0.20])
		short_isi.set_axis_bgcolor('none')
		_,_,spatches=short_isi.hist(diff(spiketimes),bins=200,range=(10,20), histtype='stepfilled')
		postdoc.adjust_spines(short_isi,['bottom','left'])
		short_isi.tick_params(direction='in')
		short_isi.set_ylabel('Count')
		short_isi.set_xlabel(r'ISI $(ms)$')
		short_isi.set_xticklabels(arange(0,10)[::2])
		plt.setp(spatches,'facecolor',colors[1])
	
	if eiglist.size:
		print eiglist.shape
		eigfxns = fig.add_subplot(2,2,3)
		eigfxns.set_axis_bgcolor('none')
		eigfxns.tick_params(direction='in')
		#Assume 6 eigenfunctions
		nfxns =6
		span = len(eiglist[0,:])/2
		x = arange(2*span) if multi else arange(-span,span)
		for i in range(nfxns):
			eigfxns.plot(x,i+eiglist[i,:],'b',linewidth=2)
			hold(True)
		postdoc.adjust_spines(eigfxns,['bottom'])
		if multi:
			eigfxns.set_xlabel(r' $\left(\mu sec\right)$')
		else:
			eigfxns.set_xlabel(r'Time from spike peak $\left(\mu sec\right)$')
		eigfxns.set_ylabel(r'Eigenfunctions')
		#draw_sizebar(eigfxns)

	plt.tight_layout()
	plt.savefig(savename+'png')
	plt.show()

def butter_bandpass_filter(data,*args, **kwargs):
	b,a = butter_bandpass(*args, **kwargs)
	return filtfilt(b,a,data) 

def draw_sizebar(ax):
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
    # draw a horizontal bar with length of 0.1 in Data coordinate
    # (ax.transData) with a label underneath.
    asb =  AnchoredSizeBar(ax.transData,
                          1,
                          r"$1$",
                          loc=8,
                          pad=0.1, borderpad=0.5, sep=5,
                          frameon=False)
    ax.add_artist(asb)

def read_DDT(filename,OFFSET=432):
	with open(filename,'rb') as stream:
		stream.seek(OFFSET)
		return fromfile(stream,dtype='int16')
