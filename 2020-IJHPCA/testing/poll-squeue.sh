#!/bin/bash

squeue -u $USER -i 5 -o "%.8i %.9P %.60j %.8u %.2t %.10M %.6D %R"
