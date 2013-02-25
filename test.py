from Recording import *
from time import time

from os import listdir, chdir, getcwd
from os.path import isdir

'''
dirname = '/Volumes/My Book/Rat/12262012/122602012/continuous/'
files = listdir(dirname)
files = [file for file in files if file.endswith('ddt')]
'''

#filename = '/Volumes/My Book/Rat/12262012/continuous/ch043.ddt'
#filename = '/Volumes/My Book/Rat/010113_real/continuous/ch043.ddt'
filename = '/Volumes/My Book/Rat/010113_real/continuous/ch032.ddt'
#test = '/Volumes/My Book/mydata3.ddt'

r =Recording(filename,verbose=False)
del r
'''
				#33 
#tetrode_map = [[36,34],[48,47],[46],[1,2,3,30,14],[32,13,31,29],[6,23],[18,21,5,19,17]]
for file in files:
	if not isdir(dirname+file[:-4]):
		try:
			start =time()
			print file
			res = Recording(dirname+file, verbose=False)
			del res
			print time()-start
		except ValueError:
			continue 
	else:
		print '%s is a directory, assuming already processed, moving on' % file[:-4]
'''