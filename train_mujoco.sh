#!/bin/sh
env_tasks="mujoco"
scenario="HalfCheetah-v4"
ifdm="dm"
# Humanoid-v4 HalfCheetah-v4 Walker-v4 same_Ant-v4 Hopper-v4 HumanoidStandup-v4
# Pusher-v4 Reacher-v4 Swimmer-v4 HalfCheetah-v4-Soft-v1
# Humanoid-dm Walker-dm
algo="cmorl"
seed_max=1
seed_set=1
safety_b=0.03
start_safety_epoch=500
momentum="True"

echo "env_tasks is ${env_tasks}, scenario is ${scenario}, algo is ${algo}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    if [ ${scenario} = "Humanoid-dm-O3" ] || [ ${scenario} = "HalfCheetah-v4-Soft-v1" ]; then
      CUDA_VISIBLE_DEVICES=1 python cmorl/runner/main_cmorl_O3.py --env-name ${scenario} --seed ${seed_set} --safety-bound ${safety_b} --start-safety ${start_safety_epoch} --with-momentum ${momentum}
    else
      if [ ${ifdm} = "dm-humanoid" ]; then
        CUDA_VISIBLE_DEVICES=1 python cmorl/runner/main_cmorl_humanoid.py --env-name ${scenario} --seed ${seed_set} --safety-bound ${safety_b} --start-safety ${start_safety_epoch} --with-momentum ${momentum}
      else
        CUDA_VISIBLE_DEVICES=1 python cmorl/runner/main_cmorl_O2.py --env-name ${scenario} --seed ${seed_set} --safety-bound ${safety_b} --start-safety ${start_safety_epoch} --with-momentum ${momentum}
      fi
    fi
done
