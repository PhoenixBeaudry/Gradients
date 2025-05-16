# Contributor Guide

## Project Information
This project implements a mining system for the Bittensor subnet Gradients.
A miner in this subnet receives requests (tasks) to finetune large language models.
These tasks can be one of SFT, DPO, or GRPO finetuning.

A request is received in miner/tuning.py If the task is accepted it is sent to a RunPod serverless worker running 2xH100 GPUs.

The RunPod serverless worker runs the docker image defined in serverless/serverless.dockerfile.

This docker image is populated by the files it needs to finetune the models, these files are serverless/hpo_optuna.py, serverless/train.py, serverless/train_dpo.py, and serverless/train_grpo.py.

The miner runs hpo_optuna.py which does a hyperparameter search to find optimal parameters for the full training run, then it launches the relevant full training script with the found parameters.

The task includes the number of hours we have to complete the finetuning.

All the miners compete to get the lowest eval_loss in the given time, or highest eval_loss if it is a GRPO task.

## Developer Tips
When working on this repo the only files we edit are: miner/endpoints/tuning and all the files in serverless/ except serverless/runpod_tests/

