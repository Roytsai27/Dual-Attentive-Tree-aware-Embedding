for alpha in 10 1 100 0.1 1000 20 50 5 2 200 500 0.5 0.2; do 
    echo "$nhead"
    python train.py --output 'loss-weight.csv' --alpha $alpha --date '13-01-01' --device 'cuda:0' &
    python train-rev.py --output 'loss-weight.csv' --alpha $alpha --date '13-01-01' --device 'cuda:0' &
    python train.py --output 'loss-weight.csv' --alpha $alpha --date '13-01-01' --device 'cuda:1' &
    python train-rev.py --output 'loss-weight.csv' --alpha $alpha --date '13-01-01' --device 'cuda:1' &
    python train.py --output 'loss-weight.csv' --alpha $alpha --date '13-01-01' --device 'cuda:2' &
    python train-rev.py --output 'loss-weight.csv' --alpha $alpha --date '13-01-01' --device 'cuda:2' &
    python train.py --output 'loss-weight.csv' --alpha $alpha --date '13-01-01' --device 'cuda:3' &
    python train-rev.py --output 'loss-weight.csv' --alpha $alpha --date '13-01-01' --device 'cuda:3' &
    python train.py --output 'loss-weight.csv' --alpha $alpha --date '13-01-01' --device 'cuda:4' &
    python train-rev.py --output 'loss-weight.csv' --alpha $alpha --date '13-01-01' --device 'cuda:4' &
    python train.py --output 'loss-weight.csv' --alpha $alpha --date '13-01-01' --device 'cuda:5' &
    python train-rev.py --output 'loss-weight.csv' --alpha $alpha --date '13-01-01' --device 'cuda:5' &
    python train.py --output 'loss-weight.csv' --alpha $alpha --date '13-01-01' --device 'cuda:0' &
    python train-rev.py --output 'loss-weight.csv' --alpha $alpha --date '13-01-01' --device 'cuda:1' &
    python train.py --output 'loss-weight.csv' --alpha $alpha --date '13-01-01' --device 'cuda:2' &
    python train-rev.py --output 'loss-weight.csv' --alpha $alpha --date '13-01-01' --device 'cuda:3' &
    python train.py --output 'loss-weight.csv' --alpha $alpha --date '13-01-01' --device 'cuda:4' &
    python train-rev.py --output 'loss-weight.csv' --alpha $alpha --date '13-01-01' --device 'cuda:5' &
    python train.py --output 'loss-weight.csv' --alpha $alpha --date '13-01-01' --device 'cuda:0' &
    python train-rev.py --output 'loss-weight.csv' --alpha $alpha --date '13-01-01' --device 'cuda:1' &
    wait
done