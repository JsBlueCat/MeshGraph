#!/usr/bin/env bash

## run the training
python train.py \
--datasets datasets/modelnet40_graph \
--name 40_graph \
--batch_size 64 \
--nclasses 40 \
--epoch 300 \
--init_type orthogonal