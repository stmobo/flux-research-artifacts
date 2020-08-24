#!/bin/bash

find ./ -name "results" -exec bash -c \
"[[ -f {}/time.output ]] || echo Didn\'t finish yet: {} && \
[[ -f {}/perf.out ]] || echo Missing perf.out in {}" \
\;
