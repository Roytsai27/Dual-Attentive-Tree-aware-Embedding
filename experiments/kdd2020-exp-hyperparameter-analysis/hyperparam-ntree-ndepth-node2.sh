for nt in 200 400; do 
    echo "$nt"
    python train.py --save 1 --output 'ntree-ndepth.csv' --ntree $nt --ndepth 2 --date '13-01-01' --device 'cuda:1' &
    python train.py --save 1 --output 'ntree-ndepth.csv' --ntree $nt --ndepth 3 --date '13-01-01' --device 'cuda:1' &
    python train.py --save 1 --output 'ntree-ndepth.csv' --ntree $nt --ndepth 4 --date '13-01-01' --device 'cuda:1' &
    python train.py --save 1 --output 'ntree-ndepth.csv' --ntree $nt --ndepth 5 --date '13-01-01' --device 'cuda:1' &
    python train.py --save 1 --output 'ntree-ndepth.csv' --ntree $nt --ndepth 6 --date '13-01-01' --device 'cuda:1' &
    python train.py --save 1 --output 'ntree-ndepth.csv' --ntree $nt --ndepth 2 --date '13-01-01' --device 'cuda:1' &
    python train.py --save 1 --output 'ntree-ndepth.csv' --ntree $nt --ndepth 3 --date '13-01-01' --device 'cuda:1' &
    python train.py --save 1 --output 'ntree-ndepth.csv' --ntree $nt --ndepth 4 --date '13-01-01' --device 'cuda:1' &
    python train.py --save 1 --output 'ntree-ndepth.csv' --ntree $nt --ndepth 5 --date '13-01-01' --device 'cuda:1' &
    python train.py --save 1 --output 'ntree-ndepth.csv' --ntree $nt --ndepth 6 --date '13-01-01' --device 'cuda:1' &
    python train.py --save 1 --output 'ntree-ndepth.csv' --ntree $nt --ndepth 2 --date '13-01-01' --device 'cuda:2' &
    python train.py --save 1 --output 'ntree-ndepth.csv' --ntree $nt --ndepth 3 --date '13-01-01' --device 'cuda:2' &
    python train.py --save 1 --output 'ntree-ndepth.csv' --ntree $nt --ndepth 4 --date '13-01-01' --device 'cuda:2' &
    python train.py --save 1 --output 'ntree-ndepth.csv' --ntree $nt --ndepth 5 --date '13-01-01' --device 'cuda:2' &
    python train.py --save 1 --output 'ntree-ndepth.csv' --ntree $nt --ndepth 6 --date '13-01-01' --device 'cuda:2' &
    python train.py --save 1 --output 'ntree-ndepth.csv' --ntree $nt --ndepth 2 --date '13-01-01' --device 'cuda:2' &
    python train.py --save 1 --output 'ntree-ndepth.csv' --ntree $nt --ndepth 3 --date '13-01-01' --device 'cuda:2' &
    python train.py --save 1 --output 'ntree-ndepth.csv' --ntree $nt --ndepth 4 --date '13-01-01' --device 'cuda:2' &
    python train.py --save 1 --output 'ntree-ndepth.csv' --ntree $nt --ndepth 5 --date '13-01-01' --device 'cuda:2' &
    python train.py --save 1 --output 'ntree-ndepth.csv' --ntree $nt --ndepth 6 --date '13-01-01' --device 'cuda:2' &
    python train.py --save 1 --output 'ntree-ndepth.csv' --ntree $nt --ndepth 2 --date '13-01-01' --device 'cuda:3' &
    python train.py --save 1 --output 'ntree-ndepth.csv' --ntree $nt --ndepth 3 --date '13-01-01' --device 'cuda:3' &
    python train.py --save 1 --output 'ntree-ndepth.csv' --ntree $nt --ndepth 4 --date '13-01-01' --device 'cuda:3' &
    python train.py --save 1 --output 'ntree-ndepth.csv' --ntree $nt --ndepth 5 --date '13-01-01' --device 'cuda:3' &
    python train.py --save 1 --output 'ntree-ndepth.csv' --ntree $nt --ndepth 6 --date '13-01-01' --device 'cuda:3' &
    python train.py --save 1 --output 'ntree-ndepth.csv' --ntree $nt --ndepth 2 --date '13-01-01' --device 'cuda:3' &
    python train.py --save 1 --output 'ntree-ndepth.csv' --ntree $nt --ndepth 3 --date '13-01-01' --device 'cuda:3' &
    python train.py --save 1 --output 'ntree-ndepth.csv' --ntree $nt --ndepth 4 --date '13-01-01' --device 'cuda:3' &
    python train.py --save 1 --output 'ntree-ndepth.csv' --ntree $nt --ndepth 5 --date '13-01-01' --device 'cuda:3' &
    python train.py --save 1 --output 'ntree-ndepth.csv' --ntree $nt --ndepth 6 --date '13-01-01' --device 'cuda:3' &
    python train.py --save 1 --output 'ntree-ndepth.csv' --ntree $nt --ndepth 2 --date '13-01-01' --device 'cuda:0' &
    python train.py --save 1 --output 'ntree-ndepth.csv' --ntree $nt --ndepth 3 --date '13-01-01' --device 'cuda:0' &
    python train.py --save 1 --output 'ntree-ndepth.csv' --ntree $nt --ndepth 4 --date '13-01-01' --device 'cuda:0' &
    python train.py --save 1 --output 'ntree-ndepth.csv' --ntree $nt --ndepth 5 --date '13-01-01' --device 'cuda:0' &
    python train.py --save 1 --output 'ntree-ndepth.csv' --ntree $nt --ndepth 6 --date '13-01-01' --device 'cuda:0' &
    python train.py --save 1 --output 'ntree-ndepth.csv' --ntree $nt --ndepth 2 --date '13-01-01' --device 'cuda:0' &
    python train.py --save 1 --output 'ntree-ndepth.csv' --ntree $nt --ndepth 3 --date '13-01-01' --device 'cuda:0' &
    python train.py --save 1 --output 'ntree-ndepth.csv' --ntree $nt --ndepth 4 --date '13-01-01' --device 'cuda:0' &
    python train.py --save 1 --output 'ntree-ndepth.csv' --ntree $nt --ndepth 5 --date '13-01-01' --device 'cuda:0' &
    python train.py --save 1 --output 'ntree-ndepth.csv' --ntree $nt --ndepth 6 --date '13-01-01' --device 'cuda:0' &
    python train.py --save 1 --output 'ntree-ndepth.csv' --ntree $nt --ndepth 2 --date '13-01-01' --device 'cuda:0' &
    python train.py --save 1 --output 'ntree-ndepth.csv' --ntree $nt --ndepth 3 --date '13-01-01' --device 'cuda:0' &
    python train.py --save 1 --output 'ntree-ndepth.csv' --ntree $nt --ndepth 4 --date '13-01-01' --device 'cuda:1' &
    python train.py --save 1 --output 'ntree-ndepth.csv' --ntree $nt --ndepth 5 --date '13-01-01' --device 'cuda:1' &
    python train.py --save 1 --output 'ntree-ndepth.csv' --ntree $nt --ndepth 6 --date '13-01-01' --device 'cuda:2' &
    python train.py --save 1 --output 'ntree-ndepth.csv' --ntree $nt --ndepth 2 --date '13-01-01' --device 'cuda:2' &
    python train.py --save 1 --output 'ntree-ndepth.csv' --ntree $nt --ndepth 3 --date '13-01-01' --device 'cuda:3' &
    python train.py --save 1 --output 'ntree-ndepth.csv' --ntree $nt --ndepth 4 --date '13-01-01' --device 'cuda:3' &
    python train.py --save 1 --output 'ntree-ndepth.csv' --ntree $nt --ndepth 5 --date '13-01-01' --device 'cuda:0' &
    python train.py --save 1 --output 'ntree-ndepth.csv' --ntree $nt --ndepth 6 --date '13-01-01' --device 'cuda:1' &
    wait
done

