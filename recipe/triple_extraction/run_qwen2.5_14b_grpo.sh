#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# GRPO fine-tuning launcher for triple extraction (Qwen2.5-14B-Instruct base).
# Run from the root of a verl checkout (see README.md in this directory for
# the exact commit + patches to apply first).
set -xeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_PATH=${MODEL_PATH:?"set MODEL_PATH to the base model, e.g. a local Qwen2.5-14B-Instruct checkout"}
TRAIN_FILE=${TRAIN_FILE:?"set TRAIN_FILE to the train.parquet built by examples/data_preprocess/triple_extraction_group_generalization.py"}
VAL_FILE=${VAL_FILE:?"set VAL_FILE to the matching test.parquet"}
CKPT_DIR=${CKPT_DIR:?"set CKPT_DIR to a directory to save checkpoints and validation generations"}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-"qwen2.5_14b_triple_extraction"}

python3 -m verl.trainer.main_ppo \
    --config-path="${SCRIPT_DIR}/config" \
    --config-name=triple_extraction_trainer \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    custom_reward_function.path="${SCRIPT_DIR}/reward_score_group_generalization_avg.py" \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.default_local_dir="${CKPT_DIR}" \
    trainer.validation_data_dir="${CKPT_DIR}/val_generations" \
    "$@"
