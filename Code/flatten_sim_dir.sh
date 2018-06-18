for f in $PWD/*/*.txt.gz
do
	fname=`basename $f` 
	echo $fname
	parentname="$(basename "$(dirname "$f")")"
	echo $parentname
	echo "$PWD"/"$parentname"_"$fname"
	cp "$f" "$PWD"/"$parentname"_"$fname"
done

for f in $PWD/*/*.txt
do
	fname=`basename $f` 
	echo $fname
	parentname="$(basename "$(dirname "$f")")"
	echo $parentname
	echo "$PWD"/"$parentname"_"$fname"
	cp "$f" "$PWD"/"$parentname"_"$fname"
done

