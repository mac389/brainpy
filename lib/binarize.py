import neuroTools as postdoc

from numpy import loadtxt
from sys import argv
filename = argv[1]

data = loadtxt(filename,delimiter='\t')
print filename
asBinary = postdoc.spiketime2binary(data)
print asBinary