#!/usr/bin/env bash
SCRIPT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

#    -p 6006:6006 \
LOG_DIR="${HOME}/logs"


# increase shared memory size either with --ipc=host or --shm-size
docker run -it \
    --gpus all \
    --ipc=host \
    -v "${SCRIPT_ROOT}/..:/master/src" \
    -v "${SCRIPT_ROOT}/logs:${LOG_DIR}" \
    -v "${SCRIPT_ROOT}/datasets:/master/datasets" \
    -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY="$DISPLAY" \
    -u "$(id -u)":"$(id -g)" \
    --rm jiahuei/pytorch:1.6.0-tf2.3-java8
