#bin/bash

fastestimator train baseline.py > baseline_log.txt

fastestimator train dropout.py > dropout_log.txt

fastestimator train freezeout.py > freezeout_log.txt