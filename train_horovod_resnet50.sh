export OverrideDefaultFP64Settings=1
export IGC_EnableDPEmulation=1
NUMBER_OF_PROCESS=2
PROCESS_PER_NODE=2
export WORKSPACE=/home/your/path/to/AI_training_by_iTEX/
MODEL_DIR=${WORKSPACE}/output
export PYTHONPATH=${WORKSPACE}/tensorflow-models
DATA_DIR=/home/your/path/to/datasets/ImageNet100/tf_records/

CONFIG_FILE=${WORKSPACE}/intel-extension-for-tensorflow/examples/train_horovod/resnet50/itex_dummy.yaml
if [ ! -d "$MODEL_DIR" ]; then
    mkdir -p $MODEL_DIR
else
    rm -rf $MODEL_DIR && mkdir -p $MODEL_DIR
fi

horovodrun -np 2 -H localhost:2 \
python ${PYTHONPATH}/official/vision/image_classification/classifier_trainer.py \
--mode=train_and_eval \
--model_type=resnet \
--dataset=imagenet \
--model_dir=$MODEL_DIR \
--data_dir=$DATA_DIR \
--config_file=$CONFIG_FILE

