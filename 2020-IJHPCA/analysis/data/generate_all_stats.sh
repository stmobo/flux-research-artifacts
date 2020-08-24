#!/bin/bash

set -e

parallel -n1 --group \
    ../generate_model_stats.py {}_real_data.pkl {}_models.pkl {}_stats.pkl \
    ::: sleep0 sleep5 stream firestarter

# for x in sleep0 sleep5 stream firestarter; do
#     echo "Stats for ${x}"
#     python3.7 ../generate_model_stats.py ${x}_real_data.pkl ${x}_empirical_model.pkl
# done
