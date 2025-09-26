#!/bin/bash

# choose model like:
MODELS=(
  "models/huggillama/llama-7b-wanda_128_0.1"
  "models/huggyllama/llama-7b_wanda_128_0.2"
  "models/huggyllama/llama-7b_wanda_128_0.3"
)

TASKS="wikitext,boolq,piqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa"
#TASKS="boolq"
DEVICE="cuda:0"
RESULTS_DIR="tmp_results"
LOG_FILE="${RESULTS_DIR}/eval_times.log"

mkdir -p "$RESULTS_DIR"
echo "評価開始時刻: $(date)" > "$LOG_FILE"
echo "==============================" >> "$LOG_FILE"

for MODEL_PATH in "${MODELS[@]}"; do
  SAFE_NAME=$(echo "$MODEL_PATH" | sed 's/\//_/g')
  OUTPUT_PATH="${RESULTS_DIR}/results_${SAFE_NAME}.json"

  echo "[$(date)] 開始: $MODEL_PATH" | tee -a "$LOG_FILE"
  START=$(date +%s)

  lm_eval \
    --model hf \
    --model_args pretrained="$MODEL_PATH,trust_remote_code=True" \
    --tasks "$TASKS" \
    --device "$DEVICE" \
    --batch_size 64 \
    --output_path "$OUTPUT_PATH"

  END=$(date +%s)
  DURATION=$((END - START))

  echo "[$(date)] 終了: $MODEL_PATH (所要時間: ${DURATION}秒)" | tee -a "$LOG_FILE"
  echo "--------------------------------------" >> "$LOG_FILE"
done

echo "評価終了時刻: $(date)" >> "$LOG_FILE"
