python train.py --output 'loss-weight.csv' --alpha 500 --date '13-01-01' --device 'cuda:0' &
python train.py --output 'loss-weight.csv' --alpha 500 --date '13-01-01' --device 'cuda:1' &
python train.py --output 'loss-weight.csv' --alpha 500 --date '13-01-01' --device 'cuda:2' &
python train-rev.py --output 'loss-weight.csv' --alpha 500 --date '13-01-01' --device 'cuda:3' &
python train-rev.py --output 'loss-weight.csv' --alpha 500 --date '13-01-01' --device 'cuda:0' &
python train-rev.py --output 'loss-weight.csv' --alpha 500 --date '13-01-01' --device 'cuda:1' &
wait

python train.py --output 'loss-weight.csv' --alpha 1000 --date '13-01-01' --device 'cuda:2' &
python train.py --output 'loss-weight.csv' --alpha 1000 --date '13-01-01' --device 'cuda:3' &
python train.py --output 'loss-weight.csv' --alpha 1000 --date '13-01-01' --device 'cuda:0' &
python train.py --output 'loss-weight.csv' --alpha 1000 --date '13-01-01' --device 'cuda:1' &
python train.py --output 'loss-weight.csv' --alpha 1000 --date '13-01-01' --device 'cuda:2' &
python train.py --output 'loss-weight.csv' --alpha 1000 --date '13-01-01' --device 'cuda:3' &
wait

python train.py --output 'loss-weight.csv' --alpha 1000 --date '13-01-01' --device 'cuda:0' &
python train.py --output 'loss-weight.csv' --alpha 1000 --date '13-01-01' --device 'cuda:1' &
python train.py --output 'loss-weight.csv' --alpha 1000 --date '13-01-01' --device 'cuda:2' &
python train-rev.py --output 'loss-weight.csv' --alpha 10000 --date '13-01-01' --device 'cuda:3' &
python train-rev.py --output 'loss-weight.csv' --alpha 10000 --date '13-01-01' --device 'cuda:0' &
python train-rev.py --output 'loss-weight.csv' --alpha 10000 --date '13-01-01' --device 'cuda:1' &
wait

python train-rev.py --output 'loss-weight.csv' --alpha 10000 --date '13-01-01' --device 'cuda:2' &
python train-rev.py --output 'loss-weight.csv' --alpha 10000 --date '13-01-01' --device 'cuda:3' &
python train-rev.py --output 'loss-weight.csv' --alpha 10000 --date '13-01-01' --device 'cuda:0' &
python train-rev.py --output 'loss-weight.csv' --alpha 10000 --date '13-01-01' --device 'cuda:1' &
python train-rev.py --output 'loss-weight.csv' --alpha 10000 --date '13-01-01' --device 'cuda:2' &
python train-rev.py --output 'loss-weight.csv' --alpha 10000 --date '13-01-01' --device 'cuda:3' &
python train-rev.py --output 'loss-weight.csv' --alpha 10000 --date '13-01-01' --device 'cuda:0' &
wait

