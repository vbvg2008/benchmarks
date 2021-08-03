#/bin/bash
# ./run.sh 2>&1 | tee p3_log.txt & exit
python tvm_cpu.py
sleep 30
python tvm_gpu.py
echo Done!