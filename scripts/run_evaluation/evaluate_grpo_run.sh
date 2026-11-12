#!/bin/bash
# Evaluate a GRPO-trained argument generator on the TFU and AVeriTeC evaluation sets.
#
# Arguments are generated with the trained generator and then evaluated with the
# same base-score source, semantics and thresholds the generator was trained with,
# all read from its GRPO config. For runs with threshold optimisation, the final
# optimised thresholds are read from <model>/logs/threshold_log.json.
# Generation is skipped for any dataset whose generated arguments already exist.
#
# Usage: evaluate_grpo_run.sh CONFIG [options]
#   CONFIG                GRPO config the generator was trained with (configs/grpo/*.yaml)
#   --model PATH|HF_ID    Generator to evaluate (default: the config's output_dir)
#   --out DIR             Output directory (default: outputs/eval/<config name>)
#   --lower X --upper Y   Override the evaluation thresholds
#   --device DEV          Device for argument generation (default: cuda:0)
#   --bs-device DEV       Device for the base score model (default: cuda:0)
set -eo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"

usage() { sed -n '2,/^set -eo/p' "$0" | sed '$d; s/^# \{0,1\}//'; }

[ $# -ge 1 ] || { usage; exit 1; }
case "$1" in -h|--help) usage; exit 0 ;; esac
CONFIG="$(realpath "$1")"; shift
MODEL="" OUT="" LOWER="" UPPER="" DEVICE="cuda:0" BS_DEVICE="cuda:0"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)     MODEL="$2";     shift 2 ;;
        --out)       OUT="$2";       shift 2 ;;
        --lower)     LOWER="$2";     shift 2 ;;
        --upper)     UPPER="$2";     shift 2 ;;
        --device)    DEVICE="$2";    shift 2 ;;
        --bs-device) BS_DEVICE="$2"; shift 2 ;;
        -h|--help)   usage; exit 0 ;;
        *) echo "Unknown option: $1"; usage; exit 1 ;;
    esac
done

cd "$REPO"

# Read the settings needed for evaluation from the GRPO config
eval "$(python - "$CONFIG" <<'EOF'
import shlex, sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
values = {
    "CFG_OUTPUT_DIR": cfg["output_dir"],
    "CFG_SEMANTICS": cfg["semantics"],
    "CFG_CONSERVATIVENESS": cfg.get("conservativeness", 1.0),
    "CFG_LOWER": cfg["lower_threshold"],
    "CFG_UPPER": cfg["upper_threshold"],
    "CFG_THRESHOLD_OPT": str(cfg.get("enable_threshold_optimization", False)).lower(),
    "CFG_STRATEGY": cfg["base_score_strategy"],
    "CFG_BASE_SCORE": cfg.get("base_score", 0.5),
    "CFG_BS_MODEL_PATH": cfg.get("bs_model_path", ""),
}
for key, value in values.items():
    print(f"{key}={shlex.quote(str(value))}")
EOF
)"

NAME="$(basename "$CONFIG" .yaml)"
MODEL="${MODEL:-$CFG_OUTPUT_DIR}"
OUT="${OUT:-outputs/eval/$NAME}"

if [ -z "$LOWER$UPPER" ] && [ "$CFG_THRESHOLD_OPT" = "true" ]; then
    THRESHOLD_LOG="$MODEL/logs/threshold_log.json"
    if [ ! -f "$THRESHOLD_LOG" ]; then
        echo "Error: $NAME uses threshold optimisation, but $THRESHOLD_LOG was not found."
        echo "Pass the final thresholds with --lower and --upper."
        exit 1
    fi
    read -r LOWER UPPER < <(python -c "import json,sys; r=json.load(open(sys.argv[1]))[-1]; print(r['lower'], r['upper'])" "$THRESHOLD_LOG")
fi
LOWER="${LOWER:-$CFG_LOWER}"
UPPER="${UPPER:-$CFG_UPPER}"

case "$CFG_STRATEGY" in
    constant_base_score) BS_ARGS=(constant_base_score --base-score "$CFG_BASE_SCORE") ;;
    llm_base_score)      BS_ARGS=(llm_base_score) ;;
    learned_base_score)  BS_ARGS=(learned_base_score --bs-model-path "$CFG_BS_MODEL_PATH") ;;
    *) echo "Error: unknown base_score_strategy '$CFG_STRATEGY'"; exit 1 ;;
esac

export PYTHONUNBUFFERED=1
mkdir -p "$OUT"
LOG="$OUT/run.log"

{
    echo ""
    echo "============================================================"
    echo "=== STARTED $(date '+%Y-%m-%d %H:%M:%S') ==="
    echo "Config     : $CONFIG"
    echo "Generator  : $MODEL"
    echo "Base scores: ${BS_ARGS[*]}"
    echo "Thresholds : lower=$LOWER upper=$UPPER"
    echo "============================================================"
} | tee -a "$LOG"

# name | evaluation data | label column
DATASETS=(
    "true_false|Data/Evaluation/evaluation_true_false_uncertain.csv|valid"
    "averitec|Data/Evaluation/evaluation_averitec.csv|auto"
)

for entry in "${DATASETS[@]}"; do
    IFS="|" read -r name data label <<< "$entry"
    generated="$OUT/${name}_generated.json"
    if [ -f "$generated" ]; then
        echo "=== Skipping generation for $name: $generated exists ===" | tee -a "$LOG"
    else
        echo "=== Generating arguments: $name ===" | tee -a "$LOG"
        python -m ita.evaluation.generate_arguments \
            --model-id "$MODEL" \
            --eval_data_paths "$data" \
            --label_column "$label" \
            --output_path "$generated" \
            --device "$DEVICE" \
            --no_wandb \
            2>&1 | tee -a "$LOG"
    fi
done

for entry in "${DATASETS[@]}"; do
    IFS="|" read -r name _ _ <<< "$entry"
    echo "=== Evaluating: $name ===" | tee -a "$LOG"
    python -m ita.evaluation.evaluate_arguments \
        --arguments_path "$OUT/${name}_generated.json" \
        --output_path "$OUT/${name}_eval.csv" \
        --semantics "$CFG_SEMANTICS" \
        --conservativeness "$CFG_CONSERVATIVENESS" \
        --upper_threshold "$UPPER" \
        --lower_threshold "$LOWER" \
        --bs-device "$BS_DEVICE" \
        --no_wandb \
        "${BS_ARGS[@]}" \
        2>&1 | tee -a "$LOG"
done

echo "=== Done. Results in $OUT ===" | tee -a "$LOG"
