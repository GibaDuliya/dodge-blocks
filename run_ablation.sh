#!/bin/bash

# 1. Original (Все новое отключено)
python run/train.py --name 0_original --state absolute --reward basic --episodes 10000 --seed 42

# 2. Baseline 
python run/train.py --name 1_with_baseline --state absolute --reward basic --baseline --episodes 10000 --seed 42

# 3. Только энтропия
python run/train.py --name 2_only_entropy --state absolute --reward basic --entropy 0.01 --episodes 10000 --seed 42

# 4. Только новые координаты (relative)
python run/train.py --name 3_only_relative_state --state relative --reward basic --episodes 10000 --seed 42

# 5. Только новые награды
python run/train.py --name 4_only_enhanced_reward --state absolute --reward enhanced --episodes 10000 --seed 42

echo "All ablation experiments finished! Check artifacts/ablation/ folder."