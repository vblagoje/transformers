#!/bin/bash

# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

CODEDIR=${24:-${PWD}}
job_name=${1:-"bert_lamb_pretraining_`date +%s`"}
train_batch_size=${2:-64}
learning_rate=${3:-"6e-3"}
precision=${4:-"fp16"}
num_gpus=${5:-8}
warmup_proportion=${6:-"0.2843"}
train_steps=${7:-7038}
save_checkpoint_steps=${8:-200}
resume_training=${9:-"false"}
create_logfile=${10:-"true"}
accumulate_gradients=${11:-"true"}
gradient_accumulation_steps=${12:-16}
seed=${13:-12439}
train_batch_size_phase2=${16:-8}
learning_rate_phase2=${17:-"4e-3"}
warmup_proportion_phase2=${18:-"0.128"}
train_steps_phase2=${19:-1563}
gradient_accumulation_steps_phase2=${20:-64}
num_workers=0
DATA_DIR_PHASE1=${21:-${CODEDIR}/data_phase1}
BERT_CONFIG=${22:-"${CODEDIR}/bert_configs/bert-mini.json"}
DATA_DIR_PHASE2=${23:-${CODEDIR}/data_phase2}
init_checkpoint=${25:-"None"}
RESULTS_DIR=$CODEDIR/results/$job_name
CHECKPOINTS_DIR=$RESULTS_DIR/checkpoints
MASTER_PORT=${27:-"1234"}

mkdir -p $CHECKPOINTS_DIR

if [ ! -d "$DATA_DIR_PHASE1" ]; then
  echo "Warning! $DATA_DIR_PHASE1 directory missing. Training cannot start"
  exit -1
fi
if [ ! -d "$DATA_DIR_PHASE2" ]; then
  echo "Warning! $DATA_DIR_PHASE2 directory missing. Training cannot start"
  exit -1
fi
if [ ! -d "$RESULTS_DIR" ]; then
  echo "Error! $RESULTS_DIR directory missing."
  exit -1
fi
if [ ! -f "$BERT_CONFIG" ]; then
  echo "Error! BERT large configuration file not found at $BERT_CONFIG"
  exit -1
fi

PREC=""
if [ "$precision" = "fp16" ]; then
  PREC="--fp16"
elif [ "$precision" = "fp32" ]; then
  PREC=""
elif [ "$precision" = "tf32" ]; then
  PREC=""
else
  echo "Unknown <precision> argument"
  exit -2
fi

ACCUMULATE_GRADIENTS=""
if [ "$accumulate_gradients" == "true" ]; then
  ACCUMULATE_GRADIENTS="--gradient_accumulation_steps=$gradient_accumulation_steps"
fi

CHECKPOINT=""
if [ "$resume_training" == "true" ]; then
  CHECKPOINT="--resume_from_checkpoint"
fi

INPUT_DIR=$DATA_DIR_PHASE1
CMD=" $CODEDIR/run_pretraining.py"
CMD+=" --input_dir=$DATA_DIR_PHASE1"
CMD+=" --output_dir=$RESULTS_DIR/phase1"
CMD+=" --config_file=$BERT_CONFIG"
CMD+=" --per_device_train_batch_size=$train_batch_size"
CMD+=" --max_steps=$train_steps"
CMD+=" --warmup_proportion=$warmup_proportion"
CMD+=" --save_steps=$save_checkpoint_steps"
CMD+=" --learning_rate=$learning_rate"
CMD+=" --seed=$seed"
CMD+=" --logging_steps=2"
CMD+=" --save_total_limit=1"
CMD+=" $PREC"
CMD+=" $ACCUMULATE_GRADIENTS"
CMD+=" $CHECKPOINT"
CMD+=" --do_train"
CMD+=" --phase1"
CMD+=" --num_workers=$num_workers"

CMD="python3 -m torch.distributed.launch --nproc_per_node=$num_gpus $CMD"

$CMD

echo "finished phase1"

#Start Phase2

PREC=""
if [ "$precision" = "fp16" ]; then
  PREC="--fp16"
elif [ "$precision" = "fp32" ]; then
  PREC=""
elif [ "$precision" = "tf32" ]; then
  PREC=""
else
  echo "Unknown <precision> argument"
  exit -2
fi

ACCUMULATE_GRADIENTS=""
if [ "$accumulate_gradients" == "true" ]; then
  ACCUMULATE_GRADIENTS="--gradient_accumulation_steps=$gradient_accumulation_steps_phase2"
fi

INPUT_DIR=$DATA_DIR_PHASE2
CMD=" $CODEDIR/run_pretraining.py"
CMD+=" --input_dir=$DATA_DIR_PHASE2"
CMD+=" --output_dir=$RESULTS_DIR/phase2"
CMD+=" --phase1_output_dir=$RESULTS_DIR/phase1"
CMD+=" --config_file=$BERT_CONFIG"
CMD+=" --per_device_train_batch_size=$train_batch_size_phase2"
CMD+=" --max_steps=$train_steps_phase2"
CMD+=" --warmup_proportion=$warmup_proportion_phase2"
CMD+=" --save_steps=$save_checkpoint_steps"
CMD+=" --learning_rate=$learning_rate_phase2"
CMD+=" --seed=$seed"
CMD+=" --logging_steps=2"
CMD+=" --save_total_limit=1"
CMD+=" $PREC"
CMD+=" $ACCUMULATE_GRADIENTS"
CMD+=" $CHECKPOINT"
CMD+=" --do_train --phase2"
CMD+=" --num_workers=$num_workers"
CMD="python3 -m torch.distributed.launch --nproc_per_node=$num_gpus $CMD"

$CMD

echo "finished phase2"
