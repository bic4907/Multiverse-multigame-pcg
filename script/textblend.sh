#!/bin/bash

for seed in {7..9}; do
    python eval_textblend.py seed=${seed} checkpoint_path="saves/e2e_exp-def_clipdr-0.1_cliptemp-0.14_gen-0.40_s-${seed}/epoch_2000/checkpoints/checkpoint_epoch_2000.pt" checkpoint_epoch=2000
done