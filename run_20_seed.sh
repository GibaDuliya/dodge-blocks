#!/bin/bash

# Скрипт для сравнения обучения без baseline и с baseline на 20 разных seeds

RESULTS_FILE="baseline_benchmark_results.txt"
SUMMARY_FILE="baseline_benchmark_summary.txt"

echo "Baseline Benchmark Results" > $RESULTS_FILE
echo "==========================" >> $RESULTS_FILE
echo "" >> $RESULTS_FILE

# Хранилище для расчета среднего
declare -a EPISODES_ORIGINAL
declare -a EPISODES_BASELINE

echo "Running benchmark for 20 different seeds..."
echo "============================================"

for seed in {1..20}; do
    echo ""
    echo "--- Seed $seed / 20 ---"
    
    # Обучение без baseline
    echo "Training 0_original (seed=$seed)..."
    python run/train.py --name "benchmark_0_original_seed${seed}" \
        --state absolute --reward basic \
        --episodes 10000 --seed $seed > /dev/null 2>&1
    
    #読み取り数のepisodesを取得
    csv_path_original="artifacts/ablation/benchmark_0_original_seed${seed}/stats.csv"
    if [ -f "$csv_path_original" ]; then
        episodes_original=$(tail -1 "$csv_path_original" | cut -d',' -f1)
        EPISODES_ORIGINAL[$seed]=$episodes_original
        echo "0_original: stopped at episode $episodes_original"
    else
        echo "0_original: ERROR - stats.csv not found"
        EPISODES_ORIGINAL[$seed]="ERROR"
    fi
    
    # Обучение с baseline
    echo "Training 1_with_baseline (seed=$seed)..."
    python run/train.py --name "benchmark_1_with_baseline_seed${seed}" \
        --state absolute --reward basic --baseline \
        --episodes 10000 --seed $seed > /dev/null 2>&1
    
    #읽取最後的episodeを取得
    csv_path_baseline="artifacts/ablation/benchmark_1_with_baseline_seed${seed}/stats.csv"
    if [ -f "$csv_path_baseline" ]; then
        episodes_baseline=$(tail -1 "$csv_path_baseline" | cut -d',' -f1)
        EPISODES_BASELINE[$seed]=$episodes_baseline
        echo "1_with_baseline: stopped at episode $episodes_baseline"
    else
        echo "1_with_baseline: ERROR - stats.csv not found"
        EPISODES_BASELINE[$seed]="ERROR"
    fi
    
    # Запись результатов
    echo "Seed $seed: Original=$episodes_original, Baseline=$episodes_baseline" >> $RESULTS_FILE
done

echo ""
echo "============================================"
echo "Generating summary..."
echo ""

# Расчет статистики
echo "Baseline Benchmark Summary" > $SUMMARY_FILE
echo "==========================" >> $SUMMARY_FILE
echo "" >> $SUMMARY_FILE
echo "Episodes to convergence (lower is better):" >> $SUMMARY_FILE
echo "" >> $SUMMARY_FILE

sum_original=0
count_original=0
sum_baseline=0
count_baseline=0

for seed in {16..16}; do
    if [[ "${EPISODES_ORIGINAL[$seed]}" != "ERROR" ]]; then
        sum_original=$((sum_original + ${EPISODES_ORIGINAL[$seed]}))
        count_original=$((count_original + 1))
    fi
    if [[ "${EPISODES_BASELINE[$seed]}" != "ERROR" ]]; then
        sum_baseline=$((sum_baseline + ${EPISODES_BASELINE[$seed]}))
        count_baseline=$((count_baseline + 1))
    fi
done

if [ $count_original -gt 0 ]; then
    avg_original=$((sum_original / count_original))
else
    avg_original=0
fi

if [ $count_baseline -gt 0 ]; then
    avg_baseline=$((sum_baseline / count_baseline))
else
    avg_baseline=0
fi

echo "0_original (no baseline):" >> $SUMMARY_FILE
echo "  Average episodes to convergence: $avg_original" >> $SUMMARY_FILE
echo "  Successful runs: $count_original / 20" >> $SUMMARY_FILE
echo "" >> $SUMMARY_FILE

echo "1_with_baseline:" >> $SUMMARY_FILE
echo "  Average episodes to convergence: $avg_baseline" >> $SUMMARY_FILE
echo "  Successful runs: $count_baseline / 20" >> $SUMMARY_FILE
echo "" >> $SUMMARY_FILE

if [ $avg_baseline -gt 0 ] && [ $avg_original -gt 0 ]; then
    improvement=$((100 * (avg_original - avg_baseline) / avg_original))
    echo "Improvement with baseline: $improvement%" >> $SUMMARY_FILE
fi

echo "" >> $SUMMARY_FILE
echo "Detailed results:" >> $SUMMARY_FILE
cat $RESULTS_FILE >> $SUMMARY_FILE

# Вывод в консоль
echo "========== BENCHMARK RESULTS =========="
cat $SUMMARY_FILE
echo "======================================="
echo ""
echo "Results saved to:"
echo "  - Detailed: $RESULTS_FILE"
echo "  - Summary:  $SUMMARY_FILE"