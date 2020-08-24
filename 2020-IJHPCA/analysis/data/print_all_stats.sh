#!/bin/bash

set -e

../print_model_stats.py {sleep0,sleep5,stream,firestarter}_stats.pkl
../print_real_stats.py {sleep0,sleep5,stream,firestarter}_real_data.pkl
