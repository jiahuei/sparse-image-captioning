#!/usr/bin/env bash
SCRIPT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

cd ${SCRIPT_ROOT}
docker build -t jiahuei/pytorch:1.6.0-java8 -f ${SCRIPT_ROOT}/pytorch16 .
