#!/usr/bin/env bash

PYTHON=${PYTHON:-"python3"}

# CONFIG=$1
# CHECKPOINT=$2
GPUS=$1
PORT=${PORT:-29500}

# $PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
#     $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4}

$PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test.py --launcher pytorch ${@:2}


# ./tools/dist_test.sh configs/mask_rcnn_r50_fpn_1x.py \
#     checkpoints/mask_rcnn_r50_fpn_1x_20181010-069fa190.pth \
#     8 --out results.pkl --eval bbox segm