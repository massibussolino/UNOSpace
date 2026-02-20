#!/usr/bin/env bash
# UNOPose inference on the Panoptes dataset.
#
# Usage – single sample:
#   ./core/unopose/panoptes_infer.sh <CFG> <GPU_IDS> <CKPT> <IDX> [DATASET_PATH]
#
# Usage – full trajectory:
#   ./core/unopose/panoptes_infer.sh <CFG> <GPU_IDS> <CKPT> 0 [DATASET_PATH] \
#       --trajectory [--start-idx 1] [--end-idx N] [--plot-path traj.png]
#
# Example (single):
#   ./core/unopose/panoptes_infer.sh configs/main_cfg.py 0 ./weights/model_final.pth 10
#
# Example (trajectory):
#   ./core/unopose/panoptes_infer.sh configs/main_cfg.py 0 ./weights/model_final.pth 0 \
#       panoptes-datasets/integral --trajectory --plot-path trajectory.png

set -x
this_dir=$(dirname "$0")

CFG=$1
CUDA_VISIBLE_DEVICES=$2
CKPT=$3
IDX=${4:-0}
DATASET_PATH=${5:-"$this_dir/../../panoptes-datasets/integral"}

if [ ! -e "$CKPT" ]; then
    echo "Checkpoint $CKPT does not exist."
    exit 1
fi

if [ ! -d "$DATASET_PATH" ]; then
    echo "Dataset path $DATASET_PATH does not exist."
    exit 1
fi

OMP_NUM_THREADS=1
MKL_NUM_THREADS=1

PYTHONPATH="$this_dir/../..":$PYTHONPATH \
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
python "$this_dir/../../panoptes_infer.py" \
    --config-file "$CFG" \
    --ckpt "$CKPT" \
    --idx "$IDX" \
    --dataset-path "$DATASET_PATH" \
    ${@:6}
