#!/bin/bash

set -e

./generate_all_models.sh
./generate_all_stats.sh
./print_all_stats.sh
./plot_all_models.sh
./scale_up_study.sh
