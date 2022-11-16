for num_tasks in 2 3 4 5 6 7
do
    for c_enc in 1 2 4 8 16 32
    do
        for c_dec in 16 32 64 128 256 512
        do
            fastestimator run unet.py --num_tasks $num_tasks --c_enc $c_enc --c_dec $c_dec
            sleep 5
        done
    done
done
