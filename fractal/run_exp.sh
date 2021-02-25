#!/bin/bash


for j in 1 2 3 4 5; do
    fastestimator train fractal.py --num_blocks 3 --block_level $j
done