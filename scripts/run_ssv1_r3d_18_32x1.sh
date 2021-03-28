#!/bin/bash

DATA_PATH=path_to_frame_list
FRAME_PATH=path_to_frames
COTERE_TYPE=CTSR

python tools/run_net.py \
--cfg configs/SSv1/R3D_18_COTERE_32x1.yaml \
DATA.PATH_TO_DATA_DIR ${DATA_PATH} \
DATA.PATH_PREFIX ${FRAME_PATH} \
COTERE.TYPE ${COTERE_TYPE}
