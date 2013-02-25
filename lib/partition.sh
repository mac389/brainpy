find . -name '*.filtered_traces' | while read line
do 
	echo $line
	python /Users/michaelchary/Desktop/Now/partition.py $line
done