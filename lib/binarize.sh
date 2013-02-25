find . -name '*.spiketimes' | while read line
do 
	echo $line
	python /Users/michaelchary/Desktop/Now/binarize.py $line > ${line%.*}.binary
done