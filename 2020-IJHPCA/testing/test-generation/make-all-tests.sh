#!/bin/bash

# Use -gt 1 to consume two arguments per pass in the loop (e.g. each
# argument has a corresponding value to go with it).
# Use -gt 0 to consume one or more arguments per pass in the loop (e.g.
# some arguments don't have a corresponding value to go with it such
# as in the --default example).
# note: if this is set to -gt 0 the /etc/hosts part is not recognized ( may be a bug )
while [[ $# -gt 0 ]]
do
    key="$1"

    case $key in
        -n|--dry-run)
            EXTRA_FLAGS="-n"${EXTRA_FLAGS:+" $EXTRA_FLAGS"}
            ;;
        --repetitions)
            EXTRA_FLAGS="--repetitions=$2"${EXTRA_FLAGS:+" $EXTRA_FLAGS"}
            shift
            ;;
        -h|--help)
            echo ' Options:
    -h|--help
    -n|--dry-run
    --repetitions'
            exit 0
            ;;
        *)
            # unknown option
            ;;
    esac
    shift # past argument or value
done

for app in sleep0 sleep5 stream firestarter; do
    ./variable-num-jobs.py -u $app ${EXTRA_FLAGS}
    ./build-model.py -u $app ${EXTRA_FLAGS}
done
