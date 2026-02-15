#!/bin/bash
python run/train.py \
    --grid_width 20 \
    --num_episodes 5000 \
    --lr 1e-3 \
    --gamma 0.99 \
    --hidden_dim 64 \
    --checkpoint_every 500