ISIpy Installation & Usage:

Installation
============================

 This package requires:

	1.Python 2.5.4 or greater
	2.Matplotlib 0.98 or greater
	3.Numpy 1.5.0 or greater
	4.Scipy 0.8.0 or greater

 ISIpy may work with earlier version of those programs because it does not use recent features in any of those programs.  This may change in future versions. 

 As with most python distributions, running "python setup.py install" from the command line should add this package's file to your path. 

Usage 
============================

 Refer to the demo file, demo.py, which demonstrates how to call ISIpy on a simulated data set. For your convenience, a simulated data set, in MATLAB format is provided, called 'simulated.mat'. To help generate more simulated data, 'simulate_data.py' is also provided. 
 
Version 1.0.1
============================

	1. Now it can read MATLAB, CSV, and Brian (space-delimited text) output
	2. Verbose console output to tell the user when files are loaded and saved 