#!/bin/bash


for i in 1 2 3 4 5; do
    for j in 1 2 3 4 5; do
        python benchmark_fractal.py $i $j
    done
done