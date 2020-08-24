#!/bin/bash

set -e

echo "Plotting All Models"
parallel -n1 --group \
         python3.7 ../plot_with_model.py --paper {2} {3} --model-type {4} --savefig {1}_{4}_model.pdf "$@" {1}_real_data.pkl {1}_models.pkl \
         ::: sleep0 sleep5 firestarter stream :::+ --noop --no-yaxis --no-yaxis --no-yaxis :::+ --noop --plot_max --plot_max --noop ::: analytical analyticalWithContention

# for x in sleep0 sleep5 stream firestarter; do
#     echo "Plotting ${x}"
#     python3.7 ../plot_with_model.py --plot_max --savefig ${x}_real_data.pkl ${x}_empirical_model.pkl
# done
