#!/bin/bash

for seed in {0..9}; do
    python train.py overwrite=true seed=${seed} loss_weights.base=0.6 tsne_interval=200 render_interval=200 vit_eval_freq=200
done
