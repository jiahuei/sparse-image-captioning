#!/usr/bin/env bash
SCRIPT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"


# Change this
DATASET_DIR="/master/datasets"
###

MSCOCO_DIR="${DATASET_DIR}/mscoco"
BU_DIR="${DATASET_DIR}/mscoco/bu"

export MPLCONFIGDIR="/tmp/matplotlib"
export PYTHONPATH=${PYTHONPATH}:"${SCRIPT_ROOT}/.."


if [ ! -d "${MSCOCO_DIR}" ]; then
    echo "Creating MS-COCO directory at ${MSCOCO_DIR}."
    mkdir ${MSCOCO_DIR}
fi

#############################################
### Karpathy Caption File

FILE="caption_datasets.zip"
if [ -f "${MSCOCO_DIR}/${FILE}" ]; then
    echo "Found ${FILE}."
else
    echo "Downloading ${FILE}"
    wget "http://cs.stanford.edu/people/karpathy/deepimagesent/${FILE}" -O "${MSCOCO_DIR}/${FILE}"
fi
if [ -f "${MSCOCO_DIR}/dataset_coco.json" ]; then
    echo "Found dataset_coco.json."
else
    echo "Extracting ${FILE}"
    unzip "${MSCOCO_DIR}/${FILE}" -d "${MSCOCO_DIR}"
fi

#FILE="dataset_coco_pos.json"
#if [ -f "${MSCOCO_DIR}/${FILE}" ]; then
#    echo "Found ${FILE}."
#else
#    python ${SCRIPT_ROOT}/generate_pos.py \
#        --json_path ${MSCOCO_DIR}/dataset_coco.json
#fi


#############################################
### MS-COCO Images

FILE="train2014.zip"
if [ -f "${MSCOCO_DIR}/${FILE}" ]; then
    echo "Found ${FILE}."
else
    echo "Downloading MS-COCO ${FILE}"
    wget "http://images.cocodataset.org/zips/${FILE}"
    unzip "${MSCOCO_DIR}/${FILE}" -d "${MSCOCO_DIR}"

    wget https://msvocds.blob.core.windows.net/images/262993_z.jpg
    mv 262993_z.jpg ${MSCOCO_DIR}/train2014/COCO_train2014_000000167126.jpg
fi
FILE="val2014.zip"
if [ -f "${MSCOCO_DIR}/${FILE}" ]; then
    echo "Found ${FILE}."
else
    echo "Downloading MS-COCO ${FILE}"
    wget "http://images.cocodataset.org/zips/${FILE}"
    unzip "${MSCOCO_DIR}/${FILE}" -d "${MSCOCO_DIR}"
fi
FILE="test2014.zip"
if [ -f "${MSCOCO_DIR}/${FILE}" ]; then
    echo "Found ${FILE}."
else
    echo "Downloading MS-COCO ${FILE}"
    wget "http://images.cocodataset.org/zips/${FILE}"
    unzip "${MSCOCO_DIR}/${FILE}" -d "${MSCOCO_DIR}"
fi


mkdir -p ${BU_DIR}
# shellcheck disable=SC2164
pushd ${BU_DIR}

#############################################
### SCST

FILE="cocotalk_label.h5"
if [ -f "${BU_DIR}/${FILE}" ]; then
    echo "Found ${FILE}."
else
    echo "Running pre-processing"
    python ${SCRIPT_ROOT}/prepro_labels.py \
        --input_json ${MSCOCO_DIR}/dataset_coco.json \
        --output_json ${BU_DIR}/cocotalk.json \
        --output_h5 ${BU_DIR}/cocotalk
fi

FILE="coco-train-idxs.p"
if [ -f "${BU_DIR}/${FILE}" ]; then
    echo "Found ${FILE}."
else
    echo "Running pre-processing for SCST"
    python ${SCRIPT_ROOT}/prepro_ngrams.py \
        --input_json ${MSCOCO_DIR}/dataset_coco.json \
        --dict_json ${BU_DIR}/cocotalk.json \
        --output_pkl ${BU_DIR}/coco-train \
        --split train
fi

popd

#############################################
### ResNet-101 Features

#mkdir -p "${DATA_DIR}/imagenet_weights"
#
#FILE="resnet101.pth"
#if [ -f "${DATA_DIR}/imagenet_weights/${FILE}" ]; then
#    echo "Found ${FILE}."
#else
#    echo "Downloading ${FILE}"
#    # https://medium.com/@acpanjan/download-google-drive-files-using-wget-3c2c025a8b99
#    wget --load-cookies /tmp/cookies.txt \
#        "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt \
#        --keep-session-cookies --no-check-certificate \
#        'https://docs.google.com/uc?export=download&id=0B7fNdx_jAqhtSmdCNDVOVVdINWs' \
#        -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=0B7fNdx_jAqhtSmdCNDVOVVdINWs" \
#        -O "${DATA_DIR}/imagenet_weights/${FILE}" && rm -rf /tmp/cookies.txt
#fi
#
#mkdir -p ${MSCOCO_DIR}/resnet101
## shellcheck disable=SC2164
#pushd ${MSCOCO_DIR}/resnet101
#
#python ${SCRIPT_ROOT}/prepro_feats.py \
#    --input_json ${MSCOCO_DIR}/dataset_coco.json \
#    --output_dir ${MSCOCO_DIR}/resnet101/cocotalk \
#    --images_root ${MSCOCO_DIR}
#popd

#############################################
### Bottom-Up Features

mkdir -p ${BU_DIR}/bu_data
# shellcheck disable=SC2164
pushd ${BU_DIR}/bu_data

FILE="trainval.zip"
if [ -f "${BU_DIR}/bu_data/${FILE}" ]; then
    echo "Found ${FILE}."
else
    echo "Downloading adaptive bottom-up features: ${FILE}"
    wget "https://storage.googleapis.com/up-down-attention/${FILE}"
fi

FILE="test2014.zip"
if [ -f "${BU_DIR}/bu_data/${FILE}" ]; then
    echo "Found ${FILE}."
else
    echo "Downloading adaptive bottom-up features: ${FILE}"
    wget "https://storage.googleapis.com/up-down-attention/${FILE}"
fi

FILE="karpathy_train_resnet101_faster_rcnn_genome.tsv.0"
if [ -f "${BU_DIR}/bu_data/trainval/${FILE}" ]; then
    echo "Found ${FILE}."
else
    echo "Unzipping adaptive bottom-up features"
    unzip "${BU_DIR}/bu_data/trainval.zip" -d "${BU_DIR}/bu_data"
    unzip "${BU_DIR}/bu_data/test2014.zip" -d "${BU_DIR}/bu_data"
fi

if [ -f "${BU_DIR}/cocobu_box/1.npy" ]; then
    echo "Found Bottom-Up box features, skipping 'make_bu_data.py'."
else
    python ${SCRIPT_ROOT}/make_bu_data.py \
        --downloaded_feats ${BU_DIR}/bu_data \
        --output_dir ${BU_DIR}/cocobu
fi

if [ -f "${BU_DIR}/cocobu_box_relative/1.npy" ]; then
    echo "Found relative Bottom-Up box features, skipping 'prepro_bbox_relative_coords.py'."
else
    python ${SCRIPT_ROOT}/prepro_bbox_relative_coords.py \
        --input_json ${MSCOCO_DIR}/dataset_coco.json \
        --input_box_dir ${BU_DIR}/cocobu_box \
        --output_dir ${BU_DIR}/cocobu_box_relative \
        --image_root ${MSCOCO_DIR}
fi
popd
