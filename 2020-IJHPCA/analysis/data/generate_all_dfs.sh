#!/bin/bash

set -e

parallel -n1 --group \
    ../generate_df.py model_data /p/lquake/herbein1/{}-model/* \
    ::: sleep0 sleep5 stream firestarter

parallel -n1 --group \
    ../generate_df.py real_data /p/lquake/herbein1/{}-multi-level/* \
    ::: sleep0 sleep5 stream firestarter

# for x in sleep0 sleep5 stream firestarter; do
#     ../generate_df.py model /p/lquake/herbein1/${x}-model/*
#     ../generate_df.py real_data /p/lquake/herbein1/${x}-multi-level/*
# done
