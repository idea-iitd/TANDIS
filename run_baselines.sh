#!/bin/bash

baseline=grand
basemod=GNNGuard-gcn
dataset=cora
down_task=node_classification
budget=20
nprocs=10
seed=123

saved_model=attack_models/${down_task}/${dataset}/model-${basemod}

python eval_attack.py \
    -method ${baseline} \
    -base_model ${saved_model} \
    -dataset ${dataset} \
    -down_task ${down_task} \
    -budget ${budget} \
    -nprocs ${nprocs} \
    -seed ${seed} \
    -device cpu \
    -lcc \
    -saved_name sols_${budget} \
    # -sols_type txt \
    # -sampled \
    # -device cuda \