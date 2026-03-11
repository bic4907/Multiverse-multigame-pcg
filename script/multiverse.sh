#!/bin/bash

# general
for seed in {0..9}; do
    python train.py overwrite=true seed=${seed} loss_weights.gen=0.4 tsne_interval=200 render_interval=200 vit_eval_freq=200 gen_threshold=0.3
done