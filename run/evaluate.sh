#!/bin/bash
python run/evaluate.py \
    --checkpoint artifacts/checkpoints/best.pt \
    --num_episodes 100 \
    --render