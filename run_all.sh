
set -e

cd "$(dirname "$0")"

python src/data_prep.py

MODELS=("mlp" "mlp_attention")
LOSSES=("bce" "weighted_bce" "focal")
RATIOS=("1to1" "1to10" "1to100" "full")

for model in "${MODELS[@]}"; do
  for loss in "${LOSSES[@]}"; do
    for ratio in "${RATIOS[@]}"; do
      echo "=== $model / $loss / $ratio ==="
      python src/train.py --model "$model" --loss "$loss" --ratio "$ratio" --epochs 20
      python src/evaluate.py --model "$model" --loss "$loss" --ratio "$ratio"
    done
  done
done


python src/explain.py --model mlp_attention --loss focal --ratio full

echo " See results/metrics_table.csv for the full comparison table."
