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
for subdir in subdirs:
	#-- Clear the screen
	print chr(27) + '[2J'
	#-------------------
	cprint('Entering %s'%subdir,'magenta')
	basepath = basedir+'/'+subdir+'/continuous'
	for shank,trode in enumerate(electrode):
		cfg_name = basepath+'/shank%d.parameters' % shank
		FILTERED = [basepath+'/'+'ch%03d.filtered' % channel for channel in trode]
		UNFILTERED = [basepath+'/'+'ch%03d.ddt' % channel for channel in trode]
		print '---'
		# Process only good channels to limit RAM consumption
		if exists(cfg_name):
			good_channel,THRESHOLDS = cPickle.load(open(cfg_name,'rb'))
			traces = [fromfile(tracename,dtype='float64') for tracename,thr in zip(FILTERED,THRESHOLDS) if thr > MINIMUM_AMPLITUDE]
		else:
			cprint('Loading filtered traces','red')
			traces = []
			for tracename in FILTERED:
				if isfile(tracename):
					print '\tLoading %s'%tracename
					traces.append(fromfile(tracename,dtype='float64'))
				else:
					CONTINUOUS = tech.switch_type(tracename,'ddt')
					print '\tFiltering from voltage trace -> Loading %s'% CONTINUOUS
					traces.append(tech.save_filtered_trace(CONTINUOUS,show=True))
			cprint('Loaded','green')
			print '\nDetermining live channels.'	
			THRESHOLDS = [tech.threshold(trace) for trace in traces] 		
		
		good_channels = [channel for i,channel in enumerate(trode) if THRESOLDS[i]>MINIMUM_AMPLITUDE]
		for channel in trode:
			cprint('%d '%channel,'red' if channel not in good_channels else 'green', end='')
		print '---'		
		WAVENAMES = [basepath+'/shank%d_channel%d.waveforms'%(shank,channel) for channel in good_channels]
		for contact,channel in enumerate(good_channels): 
			#Contact is the index. For example the 1st contact on the 1st shank is channel 33
			if exists(WAVENAMES[contact]):
				cprint('Already analyzed channel %02d on shank %d'%(channel,shank),'magenta')
				continue
			else:
				spikename = basepath+'/ch{0:03}.spiketimes'.format(channel)
				if exists(spikename) and getsize(spikename)>1000: #Double check- reall spike times files will be more than 1 MB
					spiketimes = fromfile(spikename,'int') 
				else: 
					spiketimes = tech.detect_spikes(traces[contact],thresh[contact])
					spiketimes.tofile(spikename)
				cprint('Extracting waveforms','red')
				wfs=[tech.get_waveforms(traces[contact],spiketimes) for channel in good_channels]
				wfs = vstack(tuple(wfs))									
				cprint('\tExtracted','green')
				wfs.tofile(wavenames[contact]);print 'Saved %s'%WAVENAMES[contact];print '---'
				cfg['wavefiles'].append(WAVENAMES[contact])
				del wfs
	del traces
	cPickle.dump(cfg,open(cfg_name,'wb'))