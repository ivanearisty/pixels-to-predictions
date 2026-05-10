#!/usr/bin/env bash
# Wait for v3 training to finish, then run the full inference + ensemble pipeline.
#
# Outputs:
#   logs/<variant>.{val,test}_logits.log   per-variant logit-prediction logs
#   results/logits/<variant>-{val,test}.npz   per-variant 5-letter logits
#   results/submissions/*.csv              candidate submissions for each ensemble combo
#
# Each variant trained at a specific image_size; inference uses the matching size:
#   v1: 384  |  v2: 384  |  v3: 512

set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate

V3_PID=$(cat logs/v3.train.pid)
echo "[$(date)] waiting for v3 train pid $V3_PID..."
while kill -0 "$V3_PID" 2>/dev/null; do sleep 30; done
echo "[$(date)] v3 finished, starting downstream"

mkdir -p results/logits results/submissions

declare -A IMAGE_SIZE=( [v1]=384 [v2]=384 [v3]=512 )

# 1. Per-variant val + test scoring with logit method.
for variant in v1 v2 v3; do
    sz=${IMAGE_SIZE[$variant]}
    for split in val test; do
        echo "[$(date)] scoring $variant $split @ image_size=$sz"
        python -m pixels_to_predictions.predict \
            --checkpoint "outputs/${variant}/checkpoint-final" \
            --out "/tmp/p2p_${variant}.${split}.csv" \
            --split "$split" \
            --no-metadata \
            --inference-mode logits \
            --save-logits "results/logits/${variant}-${split}.npz" \
            --batch-size 4 \
            --image-size "$sz" \
            > "logs/${variant}.${split}_logits.log" 2>&1
        echo "[$(date)]   single-model $variant on $split:"
        grep -E "VAL accuracy|Predicted" "logs/${variant}.${split}_logits.log" || true
    done
done

# 2. Ensemble combinations evaluated on val.
echo "[$(date)] === ENSEMBLE VAL EVAL ==="
for combo in "v1 v3" "v1 v2 v3" "v1 v2"; do
    args=()
    name=""
    for v in $combo; do
        args+=("results/logits/${v}-val.npz")
        name="${name}+${v}"
    done
    name="${name#+}"
    echo "[$(date)] ensemble combo: $name"
    python scripts/ensemble.py \
        --logits "${args[@]}" \
        --csv-name val.csv \
        --reference val.csv \
        --out "/tmp/ens_${name}.val.csv"
done

# 3. Single-model val recap for comparison
echo "[$(date)] === SINGLE-MODEL VAL RECAP ==="
for v in v1 v2 v3; do
    grep "VAL accuracy" "logs/${v}.val_logits.log" || true
done

# 4. Write test submissions for each ensemble combo (we'll pick the best by hand)
STAMP=$(date -u +%Y%m%d-%H%M%S)
echo "[$(date)] === WRITING TEST SUBMISSIONS ==="
for combo in "v1" "v3" "v1 v3" "v1 v2 v3"; do
    args=()
    name=""
    for v in $combo; do
        args+=("results/logits/${v}-test.npz")
        name="${name}+${v}"
    done
    name="${name#+}"
    OUT="results/submissions/${STAMP}-${name}-logitens.csv"
    python scripts/ensemble.py \
        --logits "${args[@]}" \
        --csv-name test.csv \
        --reference sample_submission.csv \
        --out "$OUT"
done

echo "[$(date)] === DONE ==="
ls -la results/submissions/${STAMP}-*.csv
