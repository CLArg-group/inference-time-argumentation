#!/bin/bash
# Optionally point the Hugging Face cache at a disk with enough space, e.g.:
# export HF_HOME="/path/to/large/disk/.cache/huggingface"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
# GRPO training is run on a single GPU; override CUDA_VISIBLE_DEVICES to pick a different one.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTORCH_ALLOC_CONF=expandable_segments:True
echo "Environment variables set for CUDA devices (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)."
