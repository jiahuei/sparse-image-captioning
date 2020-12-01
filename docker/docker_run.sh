#!/usr/bin/env bash
SCRIPT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

#    -p 6006:6006 \
DOC_DIR="/home/jiahuei/Documents"


#docker run -it \
#    --gpus all \
#    -v ${DOC_DIR}/object_relation_transformer:/master/src \
#    -v ${DOC_DIR}/3_Datasets/object_relation_data:/master/data \
#    -v ${DOC_DIR}/3_Datasets/mscoco:/master/datasets/mscoco \
#    -v ${DOC_DIR}/3_Datasets/InstaPIC1M:/master/datasets/insta \
#    -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY="$DISPLAY" \
#    -u "$(id -u)":"$(id -g)" \
#    --rm jiahuei/pytorch:0.4.1-py27-java8

# increase shared memory size either with --ipc=host or --shm-size
docker run -it \
    --gpus all \
    --ipc=host \
    -v "${SCRIPT_ROOT}/..:/master/src" \
    -v ${DOC_DIR}/1_TF_files:${DOC_DIR}/1_TF_files \
    -v ${DOC_DIR}/3_Datasets/object_relation_data:/master/data \
    -v ${DOC_DIR}/3_Datasets/mscoco:/master/datasets/mscoco \
    -v ${DOC_DIR}/3_Datasets/InstaPIC1M:/master/datasets/insta \
    -v ${DOC_DIR}/3_Datasets/stanza_resources:/master/datasets/stanza_resources \
    -v ${DOC_DIR}/3_Datasets/ImageNet:/master/datasets/imagenet \
    -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY="$DISPLAY" \
    -u "$(id -u)":"$(id -g)" \
    --rm jiahuei/pytorch:1.6.0-tf2.3-java8
