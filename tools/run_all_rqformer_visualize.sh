#!/usr/bin/env bash
set -euo pipefail

# Compact single-GPU workflow for RQFormer-2.
# Usage examples:
#   bash ~/RQFormer/tools/run_all_rqformer_visualize.sh
#   DATASETS=dotav1_0 RUN_TRAIN=1 RUN_TEST=0 RUN_BBOX=0 RUN_HEATMAP=0 RUN_CHARTS=0 RUN_CONFUSION=0 bash ~/RQFormer/tools/run_all_rqformer_visualize.sh
#   DATASETS=dotav1_0 RUN_TRAIN=0 RUN_TEST=1 RUN_BBOX=0 RUN_HEATMAP=0 RUN_CHARTS=0 RUN_CONFUSION=0 bash ~/RQFormer/tools/run_all_rqformer_visualize.sh
#   DATASETS=dotav1_0 RUN_TRAIN=0 RUN_TEST=0 RUN_BBOX=1 RUN_HEATMAP=1 bash ~/RQFormer/tools/run_all_rqformer_visualize.sh

REPO_DIR="${REPO_DIR:-$HOME/RQFormer}"
CKPT_DIR="${CKPT_DIR:-$HOME/pth}"
WORK_ROOT="${WORK_ROOT:-$REPO_DIR/work_dirs}"

DEVICE="${CUDA_VISIBLE_DEVICES:-0}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-2}"
SAMPLE_IMAGES="${SAMPLE_IMAGES:-10}"
SCORE_THR="${SCORE_THR:-0.3}"
DATASETS="${DATASETS:-all}"
FORCE="${FORCE:-0}"

RUN_SETUP="${RUN_SETUP:-1}"
INSTALL_TORCH_IF_MISSING="${INSTALL_TORCH_IF_MISSING:-1}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_TEST="${RUN_TEST:-1}"
RUN_CHARTS="${RUN_CHARTS:-1}"
RUN_CONFUSION="${RUN_CONFUSION:-1}"
RUN_BBOX="${RUN_BBOX:-1}"
RUN_HEATMAP="${RUN_HEATMAP:-1}"
RUN_SUMMARY="${RUN_SUMMARY:-1}"

cd "$REPO_DIR"
export CUDA_VISIBLE_DEVICES="$DEVICE"

now() { date '+%Y-%m-%d %H:%M:%S'; }
count_files() { [[ -d "$1" ]] && find "$1" -type f | wc -l || echo 0; }
selected_dataset() { [[ "$DATASETS" == "all" || ",$DATASETS," == *",$1,"* ]]; }
need_file() { [[ -f "$2" ]] || { echo "[SKIP $1 MISSING] $2"; return 1; }; }
need_dir() { [[ -d "$2" ]] || { echo "[SKIP $1 MISSING] $2"; return 1; }; }

has_module() {
  python - "$1" <<'PY'
import importlib.util, sys
sys.exit(0 if importlib.util.find_spec(sys.argv[1]) else 1)
PY
}

env_ready() {
  python - <<'PY'
import torch, mmcv, mmdet, mmengine, mmrotate
print('Environment ready:', torch.__version__, mmcv.__version__, mmdet.__version__, mmrotate.__version__)
PY
}

setup_env() {
  [[ "$RUN_SETUP" == "1" ]] || { echo "[SKIP SETUP] RUN_SETUP=0"; return 0; }
  echo "[SETUP] Python: $(python -c 'import sys; print(sys.executable)')"
  env_ready && return 0

  python -m pip install -U pip setuptools wheel
  if ! has_module torch; then
    [[ "$INSTALL_TORCH_IF_MISSING" == "1" ]] || { echo "[ERROR] torch missing"; exit 1; }
    python -m pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
  fi
  python -m pip install -r requirements/build.txt
  python -m pip install packaging matplotlib pycocotools six terminaltables scipy scikit-learn imagecorruptions
  command -v mim >/dev/null 2>&1 || python -m pip install -U openmim
  has_module mmcv || mim install "mmcv>=2.0.0rc2,<2.1.0"
  has_module mmengine || python -m pip install "mmengine>=0.1.0"
  has_module mmdet || python -m pip install "mmdet>=3.0.0rc5,<3.1.0"
  python -m pip install -v -e .
  env_ready
}

latest_ckpt() {
  local dir="$1"
  [[ -d "$dir" ]] || return 0
  [[ -f "$dir/latest.pth" ]] && { echo "$dir/latest.pth"; return 0; }
  find "$dir" -maxdepth 1 -type f -name 'epoch_*.pth' 2>/dev/null | sort -V | tail -n 1 || true
}

resolve_ckpt() {
  local train_dir="$1" external="$2" trained
  trained="$(latest_ckpt "$train_dir")"
  [[ -n "$trained" && -f "$trained" ]] && echo "$trained" || echo "$external"
}

run_train() {
  local title="$1" config="$2" out_dir="$3" done="$4" ckpt
  [[ "$RUN_TRAIN" == "1" ]] || { echo "[SKIP TRAIN] $title"; return 0; }
  ckpt="$(latest_ckpt "$out_dir")"
  [[ -f "$done" && "$FORCE" != "1" ]] && { echo "[SKIP TRAIN DONE] $title"; return 0; }
  [[ -n "$ckpt" && -f "$ckpt" && "$FORCE" != "1" ]] && { echo "[SKIP TRAIN CKPT] $ckpt"; touch "$done"; return 0; }

  mkdir -p "$out_dir"
  echo "[RUN TRAIN] $title"
  python tools/train.py "$config" --work-dir "$out_dir"
  touch "$done"
}

run_test() {
  local title="$1" config="$2" ckpt="$3" out_dir="$4" done="$5" pred="$6"
  [[ "$RUN_TEST" == "1" ]] || { echo "[SKIP TEST] $title"; return 0; }
  [[ -f "$done" && -f "$pred" && "$FORCE" != "1" ]] && { echo "[SKIP TEST DONE] $title"; return 0; }
  need_file "TEST CKPT" "$ckpt" || return 0

  [[ "$FORCE" == "1" ]] && rm -rf "$out_dir"
  mkdir -p "$out_dir"
  echo "[RUN TEST] $title"
  python tools/test.py "$config" "$ckpt" --work-dir "$out_dir" --out "$pred" \
    --cfg-options test_dataloader.batch_size="$BATCH_SIZE" test_dataloader.num_workers="$NUM_WORKERS"
  touch "$done"
}

run_charts() {
  local title="$1" method_dir="$2" out_dir="$3" done="$4"
  [[ "$RUN_CHARTS" == "1" ]] || { echo "[SKIP CHARTS] $title"; return 0; }
  [[ -f "$done" && "$FORCE" != "1" ]] && { echo "[SKIP CHARTS DONE] $title"; return 0; }

  mapfile -t logs < <(find "$method_dir/train" "$method_dir/test" -type f -name '*.json' 2>/dev/null | sort)
  [[ "${#logs[@]}" -gt 0 ]] || { echo "[SKIP CHARTS NO JSON] $title"; return 0; }

  rm -rf "$out_dir"; mkdir -p "$out_dir"
  python tools/analysis_tools/plot_rqformer_charts.py "${logs[@]}" --out-dir "$out_dir" --title "$title"
  touch "$done"
}

run_confusion() {
  local title="$1" config="$2" pred="$3" out_dir="$4" done="$5"
  [[ "$RUN_CONFUSION" == "1" ]] || { echo "[SKIP CONFUSION] $title"; return 0; }
  [[ -f "$done" && "$FORCE" != "1" ]] && { echo "[SKIP CONFUSION DONE] $title"; return 0; }
  need_file "PREDICTIONS" "$pred" || return 0

  mkdir -p "$out_dir"
  python tools/analysis_tools/confusion_matrix.py "$config" "$pred" "$out_dir" --score-thr "$SCORE_THR" \
    --cfg-options test_dataloader.batch_size="$BATCH_SIZE" test_dataloader.num_workers="$NUM_WORKERS"
  touch "$done"
}

run_visual() {
  local kind="$1" title="$2" config="$3" ckpt="$4" source="$5" out_dir="$6" done="$7" enabled script
  enabled="RUN_${kind}"
  [[ "${!enabled}" == "1" ]] || { echo "[SKIP $kind] $title"; return 0; }
  [[ -f "$done" && "$FORCE" != "1" ]] && { echo "[SKIP $kind DONE] $title"; return 0; }
  need_file "$kind CKPT" "$ckpt" || return 0
  [[ -d "$source" || -f "$source" ]] || { echo "[SKIP $kind SOURCE] $source"; return 0; }

  [[ "$kind" == "BBOX" ]] && script="visualize_rqformer_bboxes.py" || script="visualize_rroi_attention.py"
  rm -rf "$out_dir"; mkdir -p "$out_dir"
  python "tools/analysis_tools/$script" "$config" "$ckpt" "$source" \
    --out-dir "$out_dir" --device cuda:0 --score-thr "$SCORE_THR" --max-images "$SAMPLE_IMAGES"
  touch "$done"
}

run_job() {
  local name="$1" title="$2" config="$3" external_ckpt="$4" dataset_dir="$5" image_source="$6" group="$7"
  selected_dataset "$name" || { echo "[SKIP DATASET] $name"; return 0; }
  need_dir "DATASET" "$dataset_dir" || return 0
  need_file "CONFIG" "$config" || return 0

  local root="$WORK_ROOT/$group/$name"
  local train_dir="$root/train" test_dir="$root/test" chart_dir="$root/charts"
  local bbox_dir="$root/images/bboxes" heatmap_dir="$root/images/heatmaps"
  local pred="$test_dir/predictions.pkl" log="$root/logs/workflow_${name}.txt" ckpt
  mkdir -p "$(dirname "$log")" "$root/images"

  {
    echo "============================================================"
    echo "[JOB] $title"
    echo "Started: $(now)"
    echo "Output:  $root"
    echo "============================================================"

    run_train "$title" "$config" "$train_dir" "$root/.train.done"
    ckpt="$(resolve_ckpt "$train_dir" "$external_ckpt")"
    echo "[USING CKPT] $ckpt"
    run_test "$title" "$config" "$ckpt" "$test_dir" "$root/.test.done" "$pred"
    run_charts "$title" "$root" "$chart_dir" "$root/.charts.done"
    run_confusion "$title" "$config" "$pred" "$chart_dir/confusion_matrix" "$root/.confusion.done"
    run_visual BBOX "$title" "$config" "$ckpt" "$image_source" "$bbox_dir" "$root/.bbox.done"
    run_visual HEATMAP "$title" "$config" "$ckpt" "$image_source" "$heatmap_dir" "$root/.heatmap.done"

    echo "Finished: $(now)"
    echo "Charts:   $(count_files "$chart_dir")"
    echo "BBoxes:   $(count_files "$bbox_dir")"
    echo "Heatmaps: $(count_files "$heatmap_dir")"
  } 2>&1 | tee "$log"
}

setup_env

echo "============================================================"
echo "[RQFORMER-2 WORKFLOW]"
echo "Repo: $REPO_DIR"
echo "Datasets: $DATASETS | Force: $FORCE"
echo "Stages: train=$RUN_TRAIN test=$RUN_TEST charts=$RUN_CHARTS confusion=$RUN_CONFUSION bbox=$RUN_BBOX heatmap=$RUN_HEATMAP summary=$RUN_SUMMARY"
echo "============================================================"

run_job "dior" \
  "RQFormer | DIOR-R | R50 | 3x | Query 500 | t0.85" \
  "projects/RQFormer/configs/rroiformer_le90_r50_q500_layer2_sq1_dq1_t0.85_3x_dior.py" \
  "$CKPT_DIR/rroiformer_le90_r50_q500_layer2_sq1_dq1_t0.85_3x_dior.pth" \
  "data/DIOR" \
  "data/DIOR/JPEGImages-test" \
  "rroiformer"

run_job "dotav1_0" \
  "RQFormer | DOTA-v1.0 | R50 | 2x | Query 500 | t0.9" \
  "projects/RQFormer/configs/rroiformer_le90_r50_q500_layer2_sq1_dq1_t0.9_2x_dotav1.0.py" \
  "$CKPT_DIR/rroiformer_le90_r50_q500_layer2_sq1_dq1_t0.9_2x_dotav1.0.pth" \
  "data/split_ss_dota" \
  "data/split_ss_dota/trainval/images" \
  "rroiformer"

run_job "dotav1_5" \
  "RQFormer | DOTA-v1.5 | R50 | 2x | Query 500 | t0.9" \
  "projects/RQFormer/configs/rroiformer_le90_r50_q500_layer2_sq1_dq1_t0.9_2x_dotav1.5.py" \
  "$CKPT_DIR/rroiformer_le90_r50_q500_layer2_sq1_dq1_t0.9_2x_dotav1.5.pth" \
  "data/split_ss_dota1_5" \
  "data/split_ss_dota1_5/trainval/images" \
  "rroiformer"

if [[ "$RUN_SUMMARY" == "1" ]]; then
  python tools/analysis_tools/write_rqformer_summary.py \
    --repo-dir "$REPO_DIR" \
    --work-root "$WORK_ROOT" \
    --out "$WORK_ROOT/rroiformer/summary_results.md"
fi
echo "============================================================"
echo "[DONE OR SKIPPED] Output: $WORK_ROOT"
echo "find \"$WORK_ROOT\" -path '*/charts/*' -type f | head"
echo "find \"$WORK_ROOT\" -path '*/images/bboxes/*' -type f | head"
echo "find \"$WORK_ROOT\" -path '*/images/heatmaps/*' -type f | head"
echo "============================================================"
