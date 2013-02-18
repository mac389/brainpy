from interspike_interval import *
from numpy import *
from matplotlib.pyplot import *
from numpy.random import randint
from matplotlib import rcParams
from scipy.io import savemat, loadmat
from switch import *
import UnrecognizedFormatError


class ISIpy(object):
	
	#Constructor begins --------------------------------------------------------------------
	def __init__(self, 
		plot_settings={
			'axes.labelsize':20,
			'axes.titlesize':30,
			'xtick.direction':'out',
			'ytick.direction':'out',
			'axes.linewidth':2.0
		}, 
		control_flow={
			'should_plot':True,
			'should_save':True,
			'save_format':'png',
			'graph_filename':'data',
			'processed_data_filename':'data' #In version 1.0 only save data in MATLAB format
		}, 
		data_location = ''
		):
		
		#Control Flow= Whether to plot, save,...
		self.control_flow = control_flow
		self.should_plot = control_flow['should_plot']
		self.should_save = control_flow['should_save']
		self.save_format = control_flow['save_format']
		self.graph_filename = control_flow['graph_filename']
		self.processed_data_filename=control_flow['processed_data_filename']
		
		#Customize Heatmap Appearance
		self.plot_settings = plot_settings
		rcParams.update(plot_settings)
		
		self.accepted_formats = {
				'MATLAB':'.m',
				'CSV':'.csv',
				'Brian':'.txt,'
			}
		
		self.data_location = data_location
		self.data = self.fetch_data(self.data_location)
		#Assume that data is passed as a MATLAB matrix structured as below
		
		#                   Time ---------------------->
		#		Neurons
		#		   |
		#		   |
		#		   |
		#		   |
		#		   V
		
		#
		
		self.ISIs = self.pairwise_ISI(self.data) 
		
		#ISIs is organized as below. The ends are neglected to avoid confounding edge effects.
	
		#                   ISI ---------------------->
		#		Neurons
		#		   |
		#		   |
		#		   |
		#		   |
		#		   V
		
		self.visualization()
		self.save_everything()
		#Constructor Ends ----------------------------------------------------------------------------------
		
	def fetch_data(self,data_location):	
		if type(data_location)==list:
			print 'Data spread over multiple files'
			return [loadtxt(record,delimiter='\t') for record in data_location]
		else: #assume a single file summarizing the entire recording is passed
			self.suffix = self.data_location[self.data_location.find('.'):]
			if self.suffix =='.mat':
				print 'Loading',data_location
				return loadmat(self.data_location)['data']
			elif self.suffix== '.csv':
				print 'Loading',data_location
				return loadtxt(self.data_location,delimiter=',')
			elif self.suffix=='.txt':
				print 'Loading',data_location
				return loadtxt(self.data_location,delimiter='\t')
			else: 
				print 'Unrecognized Format. Assuming a tab delimited text file'
				return loadtxt(self.data_location,delimiter='\t')	
		
	def pairwise_ISI(self,spikeTimes):
		self.duration = min(amax(spikeTimes,axis=1))
		self.ISI_functions = array([self.process_spiketimes_ISIs(spikeTrain) for spikeTrain in spikeTimes])
		self.ISI_functions = array([self.construct_piecewise_ISI(isi_object, self.duration) for isi_object in self.ISI_functions])
		self.neuron_count = spikeTimes.shape[0]
		self.answer = squeeze(array([[self.make_I(first,second) for first in self.ISI_functions] for second in self.ISI_functions]))
		self.answer = reshape(self.answer, (self.neuron_count**2,-1))
		return self.answer		
		
	def pairs(self,lst):
	    self.i = iter(lst)
	    self.first = self.prev = self.item = self.i.next()
	    for item in self.i:
	        yield self.prev, item
	        self.prev = item
	    yield item, self.first	
		
	def process_spiketimes_ISIs(self,spikeTimes):
		return [interspike_interval(pair[0],pair[1],pair[1]-pair[0]) for pair in self.pairs(spikeTimes)][1:-1]
	
	def construct_piecewise_ISI(self,isi_object, duration):
		self.answer = zeros(duration,)
		self.answer[0] = isi_object[1].interspike_interval
		for timestep in range(1,duration-1):
			self.isi= [interval.interspike_interval for interval in isi_object  if interval.start<=timestep and interval.stop>timestep]
			if self.isi: 
				self.answer[timestep] = self.isi[0]
		return self.answer[2:-1]

	def normalize(self,first,second):
		if first and second:
			return first/float(second)-1 if first<=second else -(second/float(first)-1)
		else: 
			return 0 #Potential Bug
			
	def make_I(self,one,two):
		return array([self.normalize(first,second) for first,second in zip(one,two)])
		
	def make_heatmap(self,ISIs):
		self.average_ISIs = average(ISIs,axis=1)
		return reshape(self.average_ISIs,(sqrt(len(self.average_ISIs)),sqrt(len(self.average_ISIs))))
	
	def visualization(self):
		print 'Plotting ISI heatmap'
		if self.should_plot: 
			figure()
			imshow(self.make_heatmap(self.ISIs), interpolation='nearest',aspect='auto')
			title('ISI Distance')
			xlabel('Neuron')
			ylabel('Neuron')
			colorbar()
			grid(which='major')
			clim(-1,1)
			if self.should_save:
				print 'Saving heatmap and ISI distance timeseries'
				savefig(self.graph_filename+'.'+self.save_format,transparent=True, format=self.save_format)
			show()
	
	def	save_everything(self):
		if self.should_save:
			savemat(self.processed_data_filename, mdict={'ISI_intervals': self.ISIs})
			#The MAT file will be saved to the same directory as this file	
		
			