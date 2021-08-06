#/bin/bash
# ./run_pc.sh 2>&1 | tee pc_log.txt & exit
echo untuned cpu:
python tvm_untuned.py --d cpu
sleep 10
echo ===========
echo untuned gpu:
python tvm_untuned.py --d gpu
sleep 10
echo ===========
echo tuned cpu:
python tvm_autotune.py --d cpu --hw pc
sleep 10
echo ===========
echo tuned gpu:
python tvm_autotune.py --d gpu --hw pc
sleep 10
echo ===========
echo autoschedule cpu:
python tvm_autoschedule.py --d cpu --hw pc
sleep 10
echo ===========
echo autoschedule gpu:
python tvm_autoschedule.py --d gpu --hw pc
echo Done!