for alpha in 0.001 0.002 0.005 0.01 0.02 0.05; do 
    echo "$nhead"
    python train.py --output 'loss-weight.csv' --alpha $alpha --date '13-01-01' --device 'cuda:0' &
    python train-rev.py --output 'loss-weight.csv' --alpha $alpha --date '13-01-01' --device 'cuda:0' &
    python train.py --output 'loss-weight.csv' --alpha $alpha --date '13-01-01' --device 'cuda:1' &
    python train-rev.py --output 'loss-weight.csv' --alpha $alpha --date '13-01-01' --device 'cuda:1' &
    python train.py --output 'loss-weight.csv' --alpha $alpha --date '13-01-01' --device 'cuda:2' &
    python train-rev.py --output 'loss-weight.csv' --alpha $alpha --date '13-01-01' --device 'cuda:2' &
    python train.py --output 'loss-weight.csv' --alpha $alpha --date '13-01-01' --device 'cuda:3' &
    python train-rev.py --output 'loss-weight.csv' --alpha $alpha --date '13-01-01' --device 'cuda:3' &
    python train.py --output 'loss-weight.csv' --alpha $alpha --date '13-01-01' --device 'cuda:0' &
    python train-rev.py --output 'loss-weight.csv' --alpha $alpha --date '13-01-01' --device 'cuda:0' &
    python train.py --output 'loss-weight.csv' --alpha $alpha --date '13-01-01' --device 'cuda:1' &
    python train-rev.py --output 'loss-weight.csv' --alpha $alpha --date '13-01-01' --device 'cuda:1' &
    python train.py --output 'loss-weight.csv' --alpha $alpha --date '13-01-01' --device 'cuda:2' &
    python train-rev.py --output 'loss-weight.csv' --alpha $alpha --date '13-01-01' --device 'cuda:2' &
    python train.py --output 'loss-weight.csv' --alpha $alpha --date '13-01-01' --device 'cuda:3' &
    python train-rev.py --output 'loss-weight.csv' --alpha $alpha --date '13-01-01' --device 'cuda:3' &
    python train.py --output 'loss-weight.csv' --alpha $alpha --date '13-01-01' --device 'cuda:0' &
    python train-rev.py --output 'loss-weight.csv' --alpha $alpha --date '13-01-01' --device 'cuda:1' &
    python train.py --output 'loss-weight.csv' --alpha $alpha --date '13-01-01' --device 'cuda:2' &
    python train-rev.py --output 'loss-weight.csv' --alpha $alpha --date '13-01-01' --device 'cuda:3' &
    wait
done