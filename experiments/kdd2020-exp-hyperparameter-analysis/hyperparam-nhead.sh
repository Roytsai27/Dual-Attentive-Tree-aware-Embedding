for nhead in 1 2 4 8 16; do 
    echo "$nhead"
    python train-nhead.py --save 1 --output 'nhead.csv' --head_num $nhead --device 'cuda:0' &
    python train-nhead.py --save 1 --output 'nhead.csv' --head_num $nhead --device 'cuda:1' &
    python train-nhead.py --save 1 --output 'nhead.csv' --head_num $nhead --device 'cuda:2' &
    python train-nhead.py --save 1 --output 'nhead.csv' --head_num $nhead --device 'cuda:3' &
    python train-nhead.py --save 1 --output 'nhead.csv' --head_num $nhead --device 'cuda:4' &
    python train-nhead.py --save 1 --output 'nhead.csv' --head_num $nhead --device 'cuda:5' &
    python train-nhead.py --save 1 --output 'nhead.csv' --head_num $nhead --device 'cuda:0' &
    python train-nhead.py --save 1 --output 'nhead.csv' --head_num $nhead --device 'cuda:1' &
    python train-nhead.py --save 1 --output 'nhead.csv' --head_num $nhead --device 'cuda:2' &
    python train-nhead.py --save 1 --output 'nhead.csv' --head_num $nhead --device 'cuda:3' &
    python train-nhead.py --save 1 --output 'nhead.csv' --head_num $nhead --device 'cuda:4' &
    python train-nhead.py --save 1 --output 'nhead.csv' --head_num $nhead --device 'cuda:5' &
    wait
done