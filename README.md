brainpy
=======

Tools to analye the electrical activity of single and multiple neurons


Flow of Analysis for one Multi-unit Recording 
=======

    Each RECORDING is an object that takes as its starting point the continuous segments of a PLX file. Using utilities 
    that Patrick Minault developed, that PLX is expanded in an entire directory as the following figure describes.
                                
                                      |---------> Spikes/
                                      |
                             PLX -----|---------> {events.json, metadata.json}
                                      |
                                      |---------> Continuous/
                                      
      The CONTINUOUS directory contains one binary file for each channel used in the recording. The binary file is in 
      the [DDT format](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&sqi=2&ved=0CDIQFjAA&url=http%3A%2F%2Fhardcarve.com%2Fwikipic%2FPlexonDataFileStructureDocumentation.pdf&ei=dOErUYGJApG40gGjqIHQCg&usg=AFQjCNHphGduyd9kn_63YloIMozPHJ3csw&sig2=em4uWMyFlNfnsKt5y1y6rA&bvm=bv.42768644,d.dmQ).
      This is the entry point for other file formats.
      
      (More coming soon ...)
