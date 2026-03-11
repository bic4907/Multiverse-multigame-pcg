#!/bin/bash

# lode runner
for seed in {0..9}; do
    python train_single.py overwrite=true seed=${seed} trainset_game=lr loss_weights.base=1.0 tsne_interval=200 render_interval=200 vit_eval_freq=200 wandb_project=multigame_single exp_name=single_lr vit_score_blend=false
done

# zelda
for seed in {0..9}; do
    python train_single.py overwrite=true seed=${seed} trainset_game=tloz loss_weights.base=1.0 tsne_interval=200 render_interval=200 vit_eval_freq=200 wandb_project=multigame_single exp_name=single_tloz vit_score_blend=false
done

# super mario bros
for seed in {0..9}; do
    python train_single.py overwrite=true seed=${seed} trainset_game=smb loss_weights.base=1.0 tsne_interval=200 render_interval=200 vit_eval_freq=200 wandb_project=multigame_single exp_name=single_smb vit_score_blend=false
done

# dungeon
for seed in {0..9}; do
    python train_single.py overwrite=true seed=${seed} trainset_game=dg loss_weights.base=1.0 tsne_interval=200 render_interval=200 vit_eval_freq=200 wandb_project=multigame_single exp_name=single_dg vit_score_blend=false
done
