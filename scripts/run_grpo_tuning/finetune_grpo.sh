#!/bin/bash
# finetune_grpo.sh
# Thin wrapper around ita.grpo_finetuning.main (HuggingFace transformers + PEFT).
# Relative paths (dataset, output dir, BSM checkpoint) are resolved against the repo root.

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
LOG_DIR="$REPO_ROOT/outputs/grpo/logs"

source "$(dirname "$0")/env_setup.sh"

# ---- Defaults ------------------------------------------------------------
MODEL_ID="Qwen/Qwen3-8B"
DATASET_PATH="Data/Training/training_true_false_uncertain.csv"
OUTPUT_DIR="outputs/grpo/Qwen3-8B-GRPO"
LEARNING_RATE=1e-5
EPOCHS=2
BATCH_SIZE=16
NUM_GENERATIONS=2
SEMANTICS="dfq"
CONSERVATIVENESS=1.0
UPPER_THRESHOLD=0.6666
LOWER_THRESHOLD=0.3333
NO_WANDB=false
PUSH_TO_HUB=false
CONTINUOUS_REWARD=true
RESUME_FROM_CHECKPOINT=""
WANDB_RUN_ID=""
BS_DEVICE="cuda:0"
CONFIG_FILE=""

# Base-score strategy: constant_base_score | llm_base_score | learned_base_score
BASE_SCORE_STRATEGY="constant_base_score"
BASE_SCORE=0.5

# learned_base_score options
BS_MODEL_PATH=""

# ---- Usage ---------------------------------------------------------------
usage() {
    echo "Usage: $0 [options] [base_score_strategy_options]"
    echo ""
    echo "General options:"
    echo "  --config PATH             Load all settings from a YAML file"
    echo "  -i, --model-id            Model ID (default: Qwen/Qwen3-8B)"
    echo "  -d, --dataset             Dataset path"
    echo "  -o, --output-dir          Output dir, relative to the repo root (default: outputs/grpo/Qwen3-8B-GRPO)"
    echo "  -l, --lr                  Learning rate (default: 1e-5)"
    echo "  -e, --epochs              Training epochs (default: 2)"
    echo "  -b, --batch-size          Batch size (default: 16)"
    echo "  -n, --num-generations     GRPO generations (default: 2)"
    echo "  --semantics               Semantics: dfq|qe (default: dfq)"
    echo "  --conservativeness        Conservativeness param (default: 1.0)"
    echo "  --upper-threshold         TRUE threshold (default: 0.6666)"
    echo "  --lower-threshold         FALSE threshold (default: 0.3333)"
    echo "  --no-wandb                Disable W&B logging"
    echo "  --wandb-run-id            Existing W&B run ID to resume logging into"
    echo "  --resume-from-checkpoint  Trainer checkpoint to resume training from"
    echo "  --push-to-hub             Upload checkpoints and artefacts to the HF Hub"
    echo "  --continuous-reward       Use smooth continuous reward (default: binary)"
    echo "  --bs-device               Device for base-score model (default: cuda:0)"
    echo ""
    echo "Base-score strategy (required unless --config is used):"
    echo "  --base-score-strategy     constant_base_score | llm_base_score | learned_base_score"
    echo ""
    echo "  constant_base_score options:"
    echo "    --base-score            Constant value (default: 0.5)"
    echo ""
    echo "  learned_base_score options:"
    echo "    --bs-model-path         Path to the trained base score model checkpoint (required)"
}

# ---- Argument parsing ----------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)               CONFIG_FILE="$2";           shift 2 ;;
        -i|--model-id)          MODEL_ID="$2";              shift 2 ;;
        -d|--dataset)           DATASET_PATH="$2";          shift 2 ;;
        -o|--output-dir)        OUTPUT_DIR="$2";            shift 2 ;;
        -l|--lr)                LEARNING_RATE="$2";         shift 2 ;;
        -e|--epochs)            EPOCHS="$2";                shift 2 ;;
        -b|--batch-size)        BATCH_SIZE="$2";            shift 2 ;;
        -n|--num-generations)   NUM_GENERATIONS="$2";       shift 2 ;;
        --semantics)            SEMANTICS="$2";             shift 2 ;;
        --conservativeness)     CONSERVATIVENESS="$2";      shift 2 ;;
        --upper-threshold)      UPPER_THRESHOLD="$2";       shift 2 ;;
        --lower-threshold)      LOWER_THRESHOLD="$2";       shift 2 ;;
        --no-wandb)             NO_WANDB=true;              shift   ;;
        --push-to-hub)          PUSH_TO_HUB=true;           shift   ;;
        --continuous-reward)    CONTINUOUS_REWARD=true;     shift   ;;
        --resume-from-checkpoint) RESUME_FROM_CHECKPOINT="$2"; shift 2 ;;
        --wandb-run-id)         WANDB_RUN_ID="$2";          shift 2 ;;
        --bs-device)            BS_DEVICE="$2";             shift 2 ;;
        --base-score-strategy)  BASE_SCORE_STRATEGY="$2";  shift 2 ;;
        --base-score)           BASE_SCORE="$2";            shift 2 ;;
        --bs-model-path)        BS_MODEL_PATH="$2";         shift 2 ;;
        -h|--help)              usage; exit 0 ;;
        *) echo "Unknown option: $1"; usage; exit 1 ;;
    esac
done

mkdir -p "$LOG_DIR"

# ---- Config-file shortcut ------------------------------------------------
if [ -n "$CONFIG_FILE" ]; then
    CONFIG_STEM=$(basename "$CONFIG_FILE" .yaml)
    LOG_FILE="$LOG_DIR/${CONFIG_STEM}.log"

    EXTRA_ARGS=""

    # Auto-detect latest checkpoint from the config's output_dir
    OUTPUT_DIR=$(python -c "import yaml,sys; cfg=yaml.safe_load(open(sys.argv[1])); print(cfg.get('output_dir',''))" "$CONFIG_FILE" 2>/dev/null)
    [ -n "$OUTPUT_DIR" ] && [[ "$OUTPUT_DIR" != /* ]] && OUTPUT_DIR="$REPO_ROOT/$OUTPUT_DIR"
    if [ -n "$OUTPUT_DIR" ] && [ -d "$OUTPUT_DIR" ]; then
        LATEST_CKPT=$(ls -dt "$OUTPUT_DIR"/checkpoint-* 2>/dev/null | head -1)
        if [ -n "$LATEST_CKPT" ]; then
            echo "Resuming from checkpoint: $LATEST_CKPT"
            EXTRA_ARGS="$EXTRA_ARGS --resume_from_checkpoint $LATEST_CKPT"
        fi
    fi

    # Extract W&B run ID from existing log to resume the same run
    if [ -f "$LOG_FILE" ]; then
        WANDB_ID=$(grep -oP "run-\d+_\d+-\K[a-z0-9]+" "$LOG_FILE" | tail -1)
        if [ -n "$WANDB_ID" ]; then
            echo "Resuming W&B run: $WANDB_ID"
            EXTRA_ARGS="$EXTRA_ARGS --wandb_run_id $WANDB_ID"
        fi
        echo "" >> "$LOG_FILE"
        echo "$(printf '=%.0s' {1..60})" >> "$LOG_FILE"
        echo "=== RESUMED $(date '+%Y-%m-%d %H:%M:%S') ===" >> "$LOG_FILE"
        echo "$(printf '=%.0s' {1..60})" >> "$LOG_FILE"
    fi

    echo "Using config: $CONFIG_FILE  →  log: $LOG_FILE"
    python -m ita.grpo_finetuning.main --config "$CONFIG_FILE" $EXTRA_ARGS >> "$LOG_FILE" 2>&1
    exit $?
fi

# ---- Validate base-score strategy ----------------------------------------
case "$BASE_SCORE_STRATEGY" in
    constant_base_score|llm_base_score|learned_base_score) ;;
    *) echo "Error: --base-score-strategy must be one of: constant_base_score, llm_base_score, learned_base_score"; exit 1 ;;
esac

# ---- Build log filename --------------------------------------------------
LOG_FILE="$LOG_DIR/$(basename "$OUTPUT_DIR")_lr_${LEARNING_RATE}_ep_${EPOCHS}_bs_${BATCH_SIZE}_sem_${SEMANTICS}_bss_${BASE_SCORE_STRATEGY}.log"

# ---- Build optional flags ------------------------------------------------
WANDB_FLAG=""
[ "$NO_WANDB" = "true" ] && WANDB_FLAG="--no-wandb"

HUB_FLAG=""
[ "$PUSH_TO_HUB" = "true" ] && HUB_FLAG="--push-to-hub"

CONTINUOUS_FLAG=""
[ "$CONTINUOUS_REWARD" = "true" ] && CONTINUOUS_FLAG="--continuous-reward"

RESUME_FLAG=""
[ -n "$RESUME_FROM_CHECKPOINT" ] && RESUME_FLAG="--resume-from-checkpoint $RESUME_FROM_CHECKPOINT"

WANDB_ID_FLAG=""
[ -n "$WANDB_RUN_ID" ] && WANDB_ID_FLAG="--wandb-run-id $WANDB_RUN_ID"

# ---- Build base-score subcommand and its options -------------------------
BS_SUBCOMMAND="$BASE_SCORE_STRATEGY"
BS_EXTRA_FLAGS=""

case "$BASE_SCORE_STRATEGY" in
    constant_base_score)
        BS_EXTRA_FLAGS="--base-score $BASE_SCORE"
        ;;
    llm_base_score)
        ;;
    learned_base_score)
        if [ -z "$BS_MODEL_PATH" ]; then
            echo "Error: learned_base_score requires --bs-model-path"
            exit 1
        fi
        BS_EXTRA_FLAGS="--bs-model-path $BS_MODEL_PATH"
        ;;
esac

# ---- Launch --------------------------------------------------------------
python -m ita.grpo_finetuning.main \
    --model-id         "$MODEL_ID" \
    --dataset          "$DATASET_PATH" \
    --output-dir       "$OUTPUT_DIR" \
    --learning-rate    "$LEARNING_RATE" \
    --epochs           "$EPOCHS" \
    --batch-size       "$BATCH_SIZE" \
    --num_generatons   "$NUM_GENERATIONS" \
    --semantics        "$SEMANTICS" \
    --conservativeness "$CONSERVATIVENESS" \
    --upper-threshold  "$UPPER_THRESHOLD" \
    --lower-threshold  "$LOWER_THRESHOLD" \
    --bs-device        "$BS_DEVICE" \
    $WANDB_FLAG \
    $HUB_FLAG \
    $CONTINUOUS_FLAG \
    $RESUME_FLAG \
    $WANDB_ID_FLAG \
    $BS_SUBCOMMAND $BS_EXTRA_FLAGS \
    > "$LOG_FILE" 2>&1
