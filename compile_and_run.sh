#!/usr/bin/env bash

set -e

fullname="$1"
fname=${fullname%.*}

python3 pyast64.py $1 > $fname.s
as $fname.s -march=generic64 -o $fname.o
gcc $fname.o -o $fname.exe -no-pie -z noexecstack
./$fname.exe
