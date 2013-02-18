import sortUtils as tech

from sys import argv
from numpy import loadtxt,array_split,savetxt


from os.path import splitext

filename = argv[1]
name,_ = splitext(filename)
condition = ['before','during','after']
data = loadtxt(filename,delimiter='\t')
for idx,row in enumerate(array_split(data,3)):
	outname = name+'.filtered_traces.'+condition[idx]
	savetxt(outname,row,delimiter='\t')
	print 'Saved %s' % outname
del data