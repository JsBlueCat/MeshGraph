#!/usr/bin/env bash

## run the training
python train.py \
--datasets datasets/modelnet10 \
--name modelnet10 \
--batch_size 1 \
--nclasses 10