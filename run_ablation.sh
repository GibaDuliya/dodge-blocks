#!/bin/bash

# 1. Baseline (Все новое отключено)
python run/train.py --name 0_baseline --state absolute --reward basic --episodes 2000

# # 2. Только нормализация
python run/train.py --name 1_only_norm --state absolute --reward basic --norm --episodes 2000

# # # 3. Только энтропия
python run/train.py --name 2_only_entropy --state absolute --reward basic --entropy 0.01 --episodes 2000

# # # 4. Только новые координаты (relative)
python run/train.py --name 3_only_relative_state --state relative --reward basic --episodes 2000

# # # 5. Только новые награды
python run/train.py --name 4_only_enhanced_reward --state absolute --reward enhanced --episodes 2000

# 6. aFULL (Все вместе)
#python run/train.py --name 5_full_optimized --state relative --reward enhanced --norm --episodes 800

echo "All ablation experiments finished! Check artifacts/ablation/ folder."