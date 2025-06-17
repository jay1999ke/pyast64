#!/usr/bin/env bash

set -e

fullname="$1"
fname=${fullname%.*}

python3 pyx.py $1 > $fname.s
as $fname.s -march=generic64 -o $fname.o
gcc $fname.o -o $fname.exe -no-pie -z noexecstack
ulimit -s unlimited
perf stat ./$fname.exe
ulimit -s 8196
rm $fname.exe
rm $fname.o