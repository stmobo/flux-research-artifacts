#!/bin/bash

set -e

RUN_BUILD="true"
if [[ "$1" == "--plot" ]]; then
    RUN_BUILD="false"
    shift
fi

if [[ "$RUN_BUILD" == "true" ]];then
    echo "Building scale up model"
    ../model_scale_study.py "$@" scale_up_model.pkl
fi
echo "Plotting scale up model"
../plot_model_scale_study.py "$@" --paper --plot_max --savefig scale_up_model.pkl
echo "Printing scale up stats"
../print_scale_up_stats.py "$@" scale_up_model.pkl
