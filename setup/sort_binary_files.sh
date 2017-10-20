#!/bin/bash

for file in `grep -l "M2 " nanograv_data/binary/par_de421/*.par`
do
	echo "copying "$file" to nanograv_data/binary/par_de421/M2/"$file
done
grep -l "M2 " nanograv_data/binary/par_de421/*.par | xargs -d '\n' -I{} mv {} nanograv_data/binary/par_de421/M2/


for file in `grep -l "H3 " nanograv_data/binary/par_de421/*.par`
do
	echo "copying "$file" to nanograv_data/binary/par_de421/H3/"$file
done
grep -l "H3 " nanograv_data/binary/par_de421/*.par | xargs -d '\n' -I{} mv {} nanograv_data/binary/par_de421/H3/


#for file in `ls nanograv_data/binary/par_de421/`
#do 
#	filename=$(basename "$file")
#	filename="${filename%.*}"
#	filename="${filename%.*}"
#	for tim in `ls nanograv_data/tim/`
#	do
#		tim_filename=$(basename "$tim")
#		tim_filename="${tim_filename%.*}"
#		if [ "$filename" == "$tim_filename" ]; 
#		then 
#			echo "copying "${tim}" to nanograv_data/binary/tim/"${tim}
#			cp "nanograv_data/tim/"${tim} nanograv_data/binary/tim/
#		fi
#	done
#done
