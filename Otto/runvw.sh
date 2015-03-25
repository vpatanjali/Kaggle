for i in `seq 0 0`
do
	echo $i
	vw --quiet --cache_file vw.cache --oaa 9 --hash all -b 28 --passes 10 -f model.vw -p dev_out.csv -d dev.vwds
	vw --quiet -d val.vwds -t -i model.vw -p val_out.csv
	#vw --quiet -d $5 -t -i model.vw -p $6
#	python logloss.py $2 $4
	#python makeoutput.py $5 $6
done
