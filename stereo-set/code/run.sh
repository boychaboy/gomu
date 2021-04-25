#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=$1 python3 evaluation.py --gold-file ../data/dev.json --predictions-dir predictions/.

