#!/bin/sh
env_tasks="mujoco"
scenario="HalfCheetah-v4" # Humanoid-v4 HalfCheetah-v4
# agent_conf="2x3"
# agent_obsk=1
algo="cmorl"
# exp="mlp"
seed_max=1

echo "env_tasks is ${env_tasks}, scenario is ${scenario}, algo is ${algo}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=1 python runner/main_cmorl.py --env-name ${scenario}
done
