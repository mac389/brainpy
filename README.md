brainpy
=======

Tools to analye the electrical activity of single and multiple neurons

    Each RECORDING is an object that takes as its starting point the continuous segments of a PLX file. Using utilities 
    that Patrick Minault developed, that PLX is expanded in an entire directory as the following figure describes.
                                
                                        |-------> Spikes/
                                        |
                               PLX -----|-------> {events.json, metadata.json}
                                        |
                                        |-------> Continuous/
                                                       |
                                                       |                                                       
                                                       V
             import sortUtils.py as tech ------->  8trode.py  <-------- import neuroTools as postdoc
                                                       |
                                                       |
                                                       V
                                                  Analysis.py
                                                  
      The CONTINUOUS directory contains one binary file for each channel used in the recording. The binary file is in 
      the [DDT format] [1] 
      This is the entry point for other file formats.
      
      (More coming soon ...)

      [1]: http://hardcarve.com/wikipic/PlexonDataFileStructureDocumentation.pdf/       "DDT format"
