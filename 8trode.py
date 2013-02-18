import sortUtils as tech

from os import listdir
from os.path import isfile, splitext

from numpy import *

tetrode_map = [[33,35,36,52,38,37,34],[54,43,39,42,41,40],[48,44,47,45,50],[49,50,46,51],
			[4,14,1,16,2,15,3,30],[32,10,13,28,31,11,12,29],[27,6,9,8,26,23,24,25],[7,17,20,19,22,5,18,21]]

lookback = 100 #microseconds
lookahead = 100




basepath='/Volumes/My Book/Rat/010113_real/continuous'
recording_name = '010113'
files = [file for file in listdir(basepath) if file.endswith('ddt')]
'''
for trode in tetrode_map:
	print 'Loading tetrode with channels',trode
	channel_names = [basepath+'/'+'ch%03d.ddt' % channel for channel in trode]
	channel_names = [channel_name for channel_name in channel_names if isfile(channel_name)]
	for channel_name in channel_names:
		name,_ = splitext(channel_name)
		tracename = name +'.filtered_traces'
		if not isfile (tracename):
			print 'No filtered trace found for %s' % channel_name				
			print 'Filtering -->'
			tech.save_filtered_trace(channel_name)
			print 'Filtered and'
			print '\tSaved as %s' %(tracename)
	print '\t\t-> Saved'
	duration = len(traces[0])
	#The immediately preceding line helps make the matrices sparse so I can include more time samples
	for channel in trode:
		print 'Loading spikes from channel %d' % channel
		filename = 'ch{0:03}/ch{0:03}.spiketimes'.format(channel)
		spiketimes = loadtxt(filename,delimiter='\t')
		for spiketime in spiketimes[::1000]: #have to down sample
			start = spiketime-lookback
			stop = spiketime + lookahead
			if start > 0 and stop < duration: 
				snippets = ravel(array([trace[start:stop] for trace in traces]))
				print snippets
	'''
	
trode = tetrode_map[0]
channel_names = filter(lambda name: isfile(name), [basepath+'/'+'ch%03d.filtered_traces' % channel for channel in trode])
traces = map(memmap,channel_names)
for channel in trode:
	print 'Loading spike from channel %d ' % channel
	filename = 'ch{0:03}/ch{0:03}.spiketimes'.format(channel)
	spiketimes = memmap(basepath+'/'+filename)
	
		