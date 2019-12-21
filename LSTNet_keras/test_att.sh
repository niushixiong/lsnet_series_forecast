
time python3 main.py --data ./data/test.npz --save ./save/test_att.pk --log ./logs/test_att.log --exps 5 --patience 15 \
    --normalize 1 --loss mae --hidCNN 100 --hidRNN 100 --hidSkip 50 --output_fun no \
    --multi 0 --horizon 1 --highway_window 24 --window 30  --skip 6 --ps 3 

