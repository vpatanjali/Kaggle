# first param - name of yaml file, without the .yaml part
# second param - shape of the images

#train.py $1.yaml > $1.out

for sample in train test val
do
    ./predict_csv.py --prediction_type=regression --output_type=float convolutional_network_best.pkl data/$2p_$sample.npy data/$1_$sample.csv
done;

./evaluate.py data/$1 $2
