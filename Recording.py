from numpy import *

import sortUtils as tech 
import XKCD as XKCD 
import matplotlib.pyplot as plt 

from os import mkdir

from os.path import splitext, dirname,basename, isfile, isdir
from scipy.io import savemat,loadmat

from time import time

import cPickle
import json

from matplotlib import rcParams,mlab
rcParams['text.usetex']=True
rcParams['font.size']=16

from brian import *
from brian.library.electrophysiology import *
from brian.library.random_processes import *

from scipy.stats import scoreatpercentile, probplot, percentileofscore
from scipy.stats.mstats import zscore

""" This module takes a DDT file and analyzed the properties of its local field potentials and spike potentials. Each DDT file 
    contains the continuous voltage trace sampled at 20 kHz from one channel of a PLX recording. For further details on converting
    PLX files to DDT or reading DDT files in Python, refer to the documentation in the function read_DDT
    
    One initializes the Recording object by passing to it the filename of the DDT file.
     
    Recording then searches to see if a MAT file with the calculated parameters already exists and so calculates only the 
    quantities it needs to.
    
    If the DDT file is name abc.DDT then the MAT file is name abd.MAT
    
    The parameters are stored in a dictionary called PARAMETERS and the data in a dictionary called DATA.
    The filenames are stores in a dictionary called FILENAMES
    
    
    Load unfiltered voltage trace 
     		|									  First local maximum after the                                   first 100 timepoints
     		|> Filter trace --> Detect spikes  as  First positive crossing of       ---> Collect waveforms as the  on either side of the
     							from energy of     (16 * Median absolute deviations)                                spike time
     							voltage																					|
     	    ------------------------------------------------------------------------------------------------------------|
     		|
     		|> Form a matrix of those waveforms --> Extract the relevant features of that matrix ---> Identify neurons as clusters in the space
     		          --------> Waveforms           by discovering its eigenvectors                       -------------> PC 2
     		         |											 ----------> Time						 |
     		         |											|										 |
     		         |											|	(format to input to 				 | 
     		         V                       					V		Pycluster)						 V
     		       Time  										Waveforms								PC1             
     		       																										|
     	    ------------------------------------------------------------------------------------------------------------|
     	    |						  
     	    |    				  silhouettes		      Parameters and data
     	    |> Verify clusters by ISI intervals ---> Save files in a directory to allow indexing with files from other recordings 
     	    				
     	                                                                    
"""
class Recording(object):
	def __init__(self, filename,verbose=False):
	
		self.verbose = verbose
	
		self.IO = {}
		self.IO['input'] = filename
		self.IO['path'] = dirname(splitext(self.IO['input'])[0])
		self.IO['name'] = basename(splitext(self.IO['input'])[0]) 	
		self.IO['matfile'] = self.IO['name']+'.mat'
		self.IO['pickle'] = self.IO['name']+'.pkl'
		self.IO['savepath'] = self.IO['path']+'/'+self.IO['name']
		
		self.skip = 1000
		'''Let the absolute path of the input file be ../abc.DDT
			 
			 self.IO['input'] is ../abc.DDT
			 self.IO['path'] is ../
			 self.IO['name'] is abc (note the lack of a file extension)
	    ''' 
	    
		#--------------------------File Handling----------------------------------------------------------------
		''' The general approach is to create a directory and save two copies of every piece of analysis that
		    an instantiation of recording does. This instantiation, once finished running, stores all analysis
		    in a dictionary called DATA. That and the accompanying PARAMETERS dictionary are written to a MAT 
		    file. Both dictionaries are then written to JSON files.
		'''
		#-------------------------------------------------------------------------------------------------------

		self.parameters = {}
		self.parameters['dsp'] = {'fs':20000,'lowcut':300,'highcut':7000}
		self.parameters['trace_analysis'] = {'lookahead':100,'lookback':100}
		self.parameters['clustering'] = {'N':1,'centroids': None}
		self.parameters['PCs'] = 3
		 
		'''The following conditional makes sure that a voltage trace is in memory that has been bandpass filtered 
			between 300 Hz and 7000 Hz using a 2nd order Butterworth forwards and backwards.
		'''
		 
		if not isfile(self.IO['matfile']):	
			print 'Loading %s -> ' % self.IO['name'],
			self.data = {}
			self.data['trace'] = self.load_voltage_trace()
			print 'Loaded'
			
			if self.verbose:
				print 'Filtering between {} and {} Hz'.format(self.parameters['dsp']['lowcut'],self.parameters['dsp']['highcut'])
				print 'Assuming a sampling rate of {} Hz'.format(self.parameters['dsp']['fs']) 

			self.data['filtered_trace'] = tech.butter_bandpass_filter(self.data['trace'],**self.parameters['dsp'])
			print 'Filtered'
			self.data['energy'] = self.data['filtered_trace']*self.data['filtered_trace']
			print 'Energy Calculated'
			self.populate_fields()
		else:
			print 'Loading %s from MAT file-> ' % self.IO['name'],
			self.data = loadmat(self.IO['matfile'])
			print 'Loaded '
		
		self.run()
		
	def run(self):
		print 'Saving'
		self.save()
		print 'Saved'
		
	def populate_fields(self):
		self.data['constants']={}
		self.data['constants']['median'] = squeeze(median(self.data['filtered_trace']))
		self.data['constants']['mad'] = squeeze(median(absolute(self.data['energy']-median(self.data['energy']))))
		self.data['constants']['threshold']= squeeze(16*self.data['constants']['mad']) #Artifacts will be their own cluster
		self.data['spiketimes'] = tech.detect_spikes(self.data['energy'],self.data['constants']['threshold'])
		
		self.data['ISI'] = diff(self.data['spiketimes'])
		self.data['wfs'] = tech.get_waveforms(self.data['filtered_trace'],self.data['spiketimes'],**self.parameters['trace_analysis'])	
		print 'Got waveforms'
		self.data['PCA'] = dict(zip(['eigvals','projections','eigvecs'],tech.princomp(self.data['wfs'][::self.skip],numpc=self.parameters['PCs'])))
		print 'Got PCs'
		self.data['clustering'] = tech.cluster(transpose(self.data['PCA']['projections']))
		print 'Got Clusters'
		
		
	def save(self):
	# First, save data
		fmt = '%.4f'
		if not isdir(self.IO['savepath']):
			mkdir(self.IO['savepath'])
		
		#For each, save spiketimes and PCs to their own files
		savetxt(self.IO['savepath']+'/'+self.IO['name']+'.spiketimes',self.data['spiketimes'],delimiter='\t')
		print 'Saved spiketimes to text file'
		
		for key,value in self.data['PCA'].iteritems():
			savetxt(self.IO['savepath']+'/'+self.IO['name']+'.'+key,value,fmt=fmt,delimiter='\t')
			print 'Saved principal components %s to text file' % key
			
		for k in self.data['clustering']: #I know this is nested- still such a hack
			for key,value in self.data['clustering'][k].iteritems():
				if key == 'clustermap':
					fmt = '%u'
				savetxt(self.IO['savepath']+'/'+self.IO['name']+'.'+key,squeeze(value),fmt=fmt,delimiter='\t')
				print 'Saved clustering components %s to text file' % key
		
		with open(self.IO['savepath']+'/constants.json','wb') as f:
			json.dump(self.data['constants'],f)
					
		for key,value in self.data.iteritems():
			if key not in ['PCA','clustering','trace','filtered_trace','energy','constants','wfs']:
				#It's better not to save the waveforms, it takes up so much memory, 
				#easier to dynamically generate them
				savetxt(self.IO['savepath']+'/'+self.IO['name']+'.'+key,value,delimiter='\t')
				print 'Saved %s to text file' % key	
								
		with open(self.IO['savepath']+'/parameters.json','wb') as f:
			json.dump(self.parameters,f)
		print 'Saved parameters to its JSON file'
	# Then, save figures
		print 'Saving Voltage Trace'
		self.save_voltage_trace()
		print 'Saving ISI and Cluster'
		self.save_spike_validation()
		print 'Saving Waveforms'
		self.save_waveforms()
	
	def __repr__(self):
		return 'Recording of %s' % self.filename
	
	def load_voltage_trace(self): #later expand to read PLX files
		return tech.read_DDT(self.IO['input'])
		
	def visualize(self,kind='trace',xkcd=False):
		if kind == 'trace':
			self.visualize_voltage_trace(xkcd=False)
	
	def save_voltage_trace(self, xkcd=False):
		fig = plt.figure()
		trace_panel = fig.add_subplot(211)
		start = 20000
		stop = 40000
		trace_panel.plot(self.data['trace'][(70*start):(80*start):10],'b') #Downsample just for display
		trace_panel.set_xlabel(r'Time $\left(ms\right)$')
		trace_panel.set_ylabel(r'Voltage $ \left(\mu V \right)$')
		
		spike_panel = fig.add_subplot(212)
		spike_panel.plot(self.data['filtered_trace'][(70*start):(80*start):10],'b')
		spike_panel.set_xlabel(r'time $\left(ms\right)$')
		spike_panel.set_ylabel(r'Voltage $\left(\mu V \right)$')
		
		#Draw threshold
		spike_panel.axhline(y=0.25*self.data['constants']['threshold'],linewidth=1,color='r',linestyle='--')
		spike_panel.axhline(y=-0.25*self.data['constants']['threshold'],linewidth=1,color='r',linestyle='--')
		
		'''
		energy_panel = fig.add_subplot(313)
		energy_panel.plot(self.data['energy'][(70*start):(80*start):10],'b')
		energy_panel.set_xlabel(r'time $\left(ms\right)$')
		energy_panel.set_ylabel(r'Energy $\left(mV^{2}\right)$')
		
		#Draw threshold
		energy_panel.axhline(y=self.data['constants']['threshold'],linewidth=1,color='r',linestyle='--')
		'''	
		if xkcd:
			for panel in [trace_panel,spike_panel]:
				XKCD.XKCDify(trace_panel, expand_axes=True)
		plt.savefig(self.IO['savepath']+'/'+self.IO['name']+'_voltage.png',dpi=300)
		plt.close()
		
	def save_waveforms(self):
		fig = plt.figure()
		waveform_panel = fig.add_subplot(211)
		waveform_panel.plot(self.data['wfs'][::self.skip])
		plt.show()
		
	def save_spike_validation(self):
		#find biggest cluster
		final_key = sorted(self.data['clustering'].keys())[-1]
		final_cluster = self.data['clustering'][final_key]
		color = ['0.5','r','k'] #Assume there won't be more than noise and two units
		fig = plt.figure()
		cluster_panel = fig.add_subplot(211)
		clusterx,clustery = tech.toxy(final_cluster['centroids'])
		cluster_panel.scatter(clusterx,clustery,c='r',s=40)
		hold(True)
		nclusters = max(final_cluster['clustermap'])
		hold(True)
		for n in range(nclusters):
			x = self.data['PCA']['projections'][0,:][final_cluster['clustermap'] == n]
			y = self.data['PCA']['projections'][1,:][final_cluster['clustermap'] == n]
			cluster_panel.scatter(x,y,marker='+', c=color[n])
			del x
			del y
		cluster_panel.set_ylabel('PC2')
		cluster_panel.set_xlabel('PC1')
		
		waveform_panel = fig.add_subplot(212)
		for n in range(nclusters):
			waveform_panel.plot(self.data['wfs'][::self.skip][final_cluster['clustermap']==n],color[n])		
		
		waveform_panel.set_xlabel('Time (us)')
		waveform_panel.set_ylabel('Voltage (uV)')
		'''
		isi_panel = fig.add_subplot(212)
		isi_panel.hist(self.data['ISI'],bins=200,range=[0,300], normed=True)
		zoomed_isi = fig.add_axes([.65,.25,.2,.2])
		zoomed_isi.hist(self.data['ISI'],range=[0,20], normed=True)
		isi_panel.set_xlabel('time (ms)')
		isi_panel.set_ylabel('density')
		'''
		plt.savefig(self.IO['savepath']+'/'+self.IO['name']+'_sorting.png',dpi=300)
		plt.close()
		