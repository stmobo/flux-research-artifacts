#!/bin/bash

set -e

parallel -n1 --group \
    ../generate_model.py "$@" {}_real_data.pkl {}_model_data.pkl {}_models.pkl \
    ::: sleep0 sleep5 stream firestarter

# for x in sleep0 sleep5 stream firestarter; do
#     ../generate_model.py ${x}_real_data.pkl ${x}_model.pkl empirical_model
# done
