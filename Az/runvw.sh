for i in `seq 0 0`
do
	echo $i
	export DEV_LIMIT=36210029
	echo $DEV_LIMIT
	vw --quiet --l1 $i -k --hash all -b 28 --loss_function logistic --link logistic -f model.vw -p $2 -d $1
	vw --quiet -d $3 -t -i model.vw -p $4
	#vw --quiet -d $5 -t -i model.vw -p $6
	python logloss.py $DEV_LIMIT $2 $4
	#python makeoutput.py $5 $6
done
