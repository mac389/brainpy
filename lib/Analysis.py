from brian import *
from itertools import product
import neuroTools as postdoc
from scipy.signal import fftconvolve
from time import time
import sortUtils as tech

import ISIpy as ISIpy

filenames = ['/Volumes/My Book/Rat/010113_real/continuous/ch045/ch045.spiketimes',
			'/Volumes/My Book/Rat/010113_real/continuous/ch045/ch045.spiketimes']

ld = time()


isCCF=False
isLZC = False

if isLZC  or isCCF:
	print 'Loading Data -> ',
	data = [loadtxt(filename,delimiter='\t')*.1*ms for filename in filenames]
	#Recording must be a list of or generator expression for the lists of spiketimes
	print 'Loaded'
	w = 20

if isCCF:
	print 'Calculating CCFs -> ',
	ccfs = [CCVF(one,two,width=w*ms) for one,two in product(data,data)]
	ccfs = map(lambda ccf: ccf/ccf.max(),ccfs)
	print 'Calculated'
	rowL=len(ccfs)/2
	colL=rowL
	
	acf_panel,ax=subplots(rowL,colL, sharex=True, sharey=True) 
	#Should use absolute not relative normalization
	#Currently use absolute motivation
	for i in range(rowL):
		for j in range(colL):
			print i+j-1
			ax[i,j].plot(arange(-w,w),ccfs[i+j+1])
			ax[i,j].set_ylabel('Covariance')
			ax[i,j].set_xlabel('Time (ms)')
			postdoc.adjust_spines(ax[i,j],['bottom','left'])
	show()

if isLZC:
	print '------'
	print 'Computing LZ:', [postdoc.LZC(datum) for datum in data]
	print '------'


ISID = ISIpy.ISIpy(data_location=filenames)

#Get time series for LZ

	