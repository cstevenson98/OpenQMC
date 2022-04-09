#!/usr/bin/env bash

export ELASTIC_N=1
export ELASTIC_KAPPA=1.0
export ELASTIC_DELTA=1.0
export OUT_PATH=/home/conor/dev/OpenQMC/data/elastic.csv

/home/conor/dev/OpenQMC/build/elastic

unset ELASTIC_KAPPA
unset ELASTIC_DELTA
unset OUT_PATH
unset ELASTIC_N
