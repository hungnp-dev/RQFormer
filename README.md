# RQFormer-2

Clean research source for improving RQFormer on three oriented object detection datasets:

- DIOR-R: ablation and module analysis.
- DOTA-v1.0: main benchmark.
- DOTA-v1.5: harder benchmark with smaller and denser objects.

## Main Configs

```bash
projects/RQFormer/configs/rroiformer_le90_r50_q500_layer2_sq1_dq1_t0.85_3x_dior.py
projects/RQFormer/configs/rroiformer_le90_r50_q500_layer2_sq1_dq1_t0.9_2x_dotav1.0.py
projects/RQFormer/configs/rroiformer_le90_r50_q500_layer2_sq1_dq1_t0.9_2x_dotav1.5.py
```

## Core Method Files

```bash
projects/RQFormer/rroiformer/oriented_ddq_rcnn.py
projects/RQFormer/rroiformer/oriented_ddq_fcn_rpn.py
projects/RQFormer/rroiformer/rroiformer_decoder.py
projects/RQFormer/rroiformer/rroiformer_decoder_layer.py
projects/RQFormer/rroiformer/rroiattention.py
projects/RQFormer/rroiformer/match_cost.py
projects/RQFormer/rroiformer/TopkHungarianAssigner.py
```

## Train

```bash
python tools/train.py projects/RQFormer/configs/rroiformer_le90_r50_q500_layer2_sq1_dq1_t0.85_3x_dior.py
python tools/train.py projects/RQFormer/configs/rroiformer_le90_r50_q500_layer2_sq1_dq1_t0.9_2x_dotav1.0.py
python tools/train.py projects/RQFormer/configs/rroiformer_le90_r50_q500_layer2_sq1_dq1_t0.9_2x_dotav1.5.py
```

## Test

```bash
python tools/test.py projects/RQFormer/configs/rroiformer_le90_r50_q500_layer2_sq1_dq1_t0.85_3x_dior.py /path/to/dior.pth
python tools/test.py projects/RQFormer/configs/rroiformer_le90_r50_q500_layer2_sq1_dq1_t0.9_2x_dotav1.0.py /path/to/dotav1.0.pth
python tools/test.py projects/RQFormer/configs/rroiformer_le90_r50_q500_layer2_sq1_dq1_t0.9_2x_dotav1.5.py /path/to/dotav1.5.pth
```

## One-For-All Evaluation And Visualization

Default rented-machine layout:

```bash
repo:        ~/RQFormer
checkpoints: ~/pth
output:      ~/RQFormer/work_dirs
```

Run the full workflow:

```bash
bash ~/RQFormer/tools/run_all_rqformer_visualize.sh
```

By default the workflow installs missing environment packages in the current active Python/conda environment, trains missing datasets, tests them, then exports only a small sample of visualizations. If a dataset already has a checkpoint in its train folder, training is skipped.

```bash
10 bounding-box images
10 RRoI attention heatmaps
training/testing charts from JSON logs
confusion matrix chart from dumped predictions
```

Useful options:

```bash
RUN_SETUP=0 bash ~/RQFormer/tools/run_all_rqformer_visualize.sh
FORCE=1 bash ~/RQFormer/tools/run_all_rqformer_visualize.sh
DATASETS=dior,dotav1_0 bash ~/RQFormer/tools/run_all_rqformer_visualize.sh
SAMPLE_IMAGES=20 bash ~/RQFormer/tools/run_all_rqformer_visualize.sh
RUN_TEST=0 RUN_BBOX=1 RUN_HEATMAP=1 bash ~/RQFormer/tools/run_all_rqformer_visualize.sh
RUN_TRAIN=0 RUN_TEST=0 RUN_BBOX=0 RUN_HEATMAP=0 RUN_CHARTS=1 RUN_CONFUSION=1 bash ~/RQFormer/tools/run_all_rqformer_visualize.sh
RUN_TRAIN=0 bash ~/RQFormer/tools/run_all_rqformer_visualize.sh
```

`RUN_SETUP=0` skips environment setup when the machine is already ready. When `RUN_TRAIN=0`, the workflow skips training and uses checkpoints from the train folder if available, otherwise from `~/pth`. Existing train/test/chart/confusion/bbox/heatmap stages are skipped unless `FORCE=1`.

Output structure:

```bash
work_dirs/rroiformer/dior/
work_dirs/rroiformer/dotav1_0/
work_dirs/rroiformer/dotav1_5/
```

Each dataset folder contains:

```bash
logs/
train/
test/
charts/
images/bboxes/
images/heatmaps/
```

The chart stage reads MMEngine `.json` logs and saves available curves such as loss, learning rate, evaluation metrics, and runtime statistics.
The confusion stage saves `charts/confusion_matrix/confusion_matrix.png` when `test/predictions.pkl` is available.
