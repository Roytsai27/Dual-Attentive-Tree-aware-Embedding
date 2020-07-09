for nd in 4 8 16 32 64; do 
    echo "$nd"
    python train-ndim.py --save 1 --output 'ndim.csv' --dim $nd --device 'cuda:0' &
    python train-ndim.py --save 1 --output 'ndim.csv' --dim $nd --device 'cuda:0' &
    python train-ndim.py --save 1 --output 'ndim.csv' --dim $nd --device 'cuda:1' &
    python train-ndim.py --save 1 --output 'ndim.csv' --dim $nd --device 'cuda:1' &
    python train-ndim.py --save 1 --output 'ndim.csv' --dim $nd --device 'cuda:2' &
    python train-ndim.py --save 1 --output 'ndim.csv' --dim $nd --device 'cuda:2' &
    python train-ndim.py --save 1 --output 'ndim.csv' --dim $nd --device 'cuda:3' &
    python train-ndim.py --save 1 --output 'ndim.csv' --dim $nd --device 'cuda:3' &
    python train-ndim.py --save 1 --output 'ndim.csv' --dim $nd --device 'cuda:4' &
    python train-ndim.py --save 1 --output 'ndim.csv' --dim $nd --device 'cuda:4' &
    python train-ndim.py --save 1 --output 'ndim.csv' --dim $nd --device 'cuda:5' &
    python train-ndim.py --save 1 --output 'ndim.csv' --dim $nd --device 'cuda:5' &
    wait
done