#!/bin/sh
#
# Reproduce results of ACM SIGCOMM ICN'13 hash-routing paper
#
echo "ADDING ICARUS TO PYTHON PATH"
export PYTHONPATH=`pwd`:$PYTHONPATH
echo "CREATING DIRECTORIES FOR GRAPHS AND LOGS"
mkdir -p logs
mkdir -p graphs
cd icarus
echo "EXECUTING SIMULATIONS"
python3 run.py
# python3 ml_cache.py
echo "PLOTTING RESULTS"
# python3 plot.py
echo "DONE"
