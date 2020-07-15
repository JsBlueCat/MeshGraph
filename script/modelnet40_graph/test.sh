#!/usr/bin/env bash

## run the test
python test.py \
--datasets datasets/modelnet40_graph \
--name 40_graph \
--batch_size 64 \
--nclasses 40 \
--last_epoch final