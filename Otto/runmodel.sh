# first param - name of yaml file, without the .yaml part
# second param - shape of the images

train.py $1.yaml > $1.out

for sample in train test val
do
    ./predict_csv.py --prediction_type=regression --output_type=float $1.pkl $sample.npy $1_$sample.csv
done;

./evaluate.py $1
