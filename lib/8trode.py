import sortUtils as tech
import cPickle
from os import listdir, makedirs
from os.path import isfile, splitext, isdir, exists, getsize
from termcolor import cprint
from numpy import *
import cPickle
#All channels < 54
electrode = [[33,35,36,53,38,37,34],[54,43,39,42,41,40],[48,44,47,45],[49,50,46,51],
			[4,14,1,16,2,15,3,30],[32,10,13,28,31,11,12,29],[27,6,9,8,26,23,24,25],[7,17,20,19,22,5,18,21]]
lookback = 100 #microseconds
lookahead = 100
cut_duration = lookahead + lookback
MINIMUM_AMPLITUDE = 4

#Summary: Recording (as DDT) -> 8-trode culled waveforms for PCA
basedir = '/Volumes/My Book/Rat'
skipdirs = ['11302012','11312012','010113_real']
subdirs = [name for name in listdir(basedir) if isdir(basedir+'/'+name) and name not in skipdirs]
concatenate = False
for subdir in subdirs:
	#-- Clear the screen
	print chr(27) + '[2J'
	#-------------------
	cprint('Entering %s'%subdir,'magenta')
	basepath = basedir+'/'+subdir+'/continuous'
	for shank,trode in enumerate(electrode):
		WAVENAME = basepath+'/shank%d.waveforms'% shank
		if exists(WAVENAME):
			cprint('Already analyzed shank %d'% shank,'magenta');
			print 'Moving on\n--'
			continue
		else:
			cfg_name = basepath+'/shank%d.parameters' % shank
			FILTERED = [basepath+'/ch%03d.filtered' % channel for channel in trode]
			UNFILTERED = [basepath+'/ch%03d.ddt' % channel for channel in trode]
			print '---'
			traces = {}
			# Process only good channels to limit RAM consumption
			if exists(cfg_name):
				good_channels,THRESHOLDS = cPickle.load(open(cfg_name,'rb'))
				traces = {tech.get_channel_id(tracename):fromfile(tracename,dtype='float64')
							for tracename,thr in zip(FILTERED,THRESHOLDS) if thr > MINIMUM_AMPLITUDE}
			else:
				cprint('Loading filtered traces','red')
				for tracename in FILTERED:
					if isfile(tracename):
						print '\tLoading %s'%tracename
						traces[tech.get_channel_id(tracename)]=fromfile(tracename,dtype='float64')
					else:
						CONTINUOUS = tech.switch_type(tracename,'ddt')
						print '\tFiltering from voltage trace -> Loading %s'% CONTINUOUS
						traces[tech.get_channel_id(tracename)]=tech.save_filtered_trace(CONTINUOUS,show=True)
				
				cprint('Loaded','green')
				print '\nDetermining live channels.'	
				THRESHOLDS = {str(channel):tech.threshold(traces[str(channel)]) for channel in trode}				
				good_channels = [channel for channel in trode if THRESHOLDS[str(channel)]>MINIMUM_AMPLITUDE]
				cPickle.dump((good_channels,THRESHOLDS),open(cfg_name,'wb'))
			#--
			for channel in trode:
				cprint('%d '%channel,'red' if channel not in good_channels else 'green', end='')
			print '\n---'		
			#--
			USUAL_MAX_SPIKE_NUMBER = 1000000
			wfs = zeros((len(good_channels)*cut_duration,USUAL_MAX_SPIKE_NUMBER))
			for shift,channel in enumerate(good_channels): 
				spikename = basepath+'/ch{0:03}.spiketimes'.format(channel)
				if exists(spikename):
					spiketimes = fromfile(spikename,'int') 
				else: 
					spiketimes = tech.detect_spikes(traces[str(channel)],THRESHOLDS[str(channel)])
					spiketimes.tofile(spikename)
				cprint('Extracting waveforms for spikes on channel %d' % channel,'red')
				#wf = vstack(tuple([tech.get_waveforms(traces[str(channel)],spiketimes) for channel in good_channels]))
				#Can't figure out how to avoid vstack copy
				row,col = wf.shape;print wf.shape
				wfs[shift*row:(shift+1)*row,:col] = tech.get_waveforms(traces[str(other_channel)],spiketimes)
				cprint('\Extracted','green')
			#wfs = hstack(wfs)
			print wfs.shape
			wfs.tofile(basepath+'/shank%d.waveforms'%shank)
			del wfs
		del traces