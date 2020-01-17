#bin/bash

fastestimator train baseline.py --epochs 50 > baseline_log.txt

fastestimator train dropout.py --epochs 50 > dropout_log.txt

fastestimator train freezeout.py --epochs 50 > freezeout_log.txt