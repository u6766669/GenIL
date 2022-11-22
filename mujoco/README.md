# Genetic Imitation Learning for reward extrapolation

## Preparing

Make sure mujoco is installed in advance.

## Training

- Example Script to infer reward and run reinforcement learning

####HalfCheetah
```
python genil_mujoco.py --env_id HalfCheetah-v2 --env_type mujoco --learners_path ./learner/demo_models/halfcheetah/checkpoints --log_dir ./log/halfcheetah/test/GenIL_halfcheetah --custom_reward preference_normalized --ppo_log_path preference_norm_ctrl_coeff_011 --ctrl_coeff 0.1 --stochastic --rl_runs 5 --infer --train_rl```
```
####Hopper
```
python genil_mujoco.py --env_id Hopper-v2 --env_type mujoco --learners_path ./learner/demo_models/hopper/checkpoints --log_dir ./log/hopper/test/GenIL_hopper/ --custom_reward preference_normalized --ppo_log_path preference_norm_ctrl_coeff_011 --ctrl_coeff 0.0001 --stochastic --rl_runs 5 --max_chkpt 380 --model_iter 500 --step_size 300 --snippets_size 2000 --infer --train_rl```
```

### Eval

- To evaluate extrapolation, put `--eval` option at the end.
- To evaluate policy performance, put `--eval_rl` option at the end.
```
python genil_mujoco.py --env_id HalfCheetah-v2 --env_type mujoco --learners_path ./learner/demo_models/halfcheetah/checkpoints --log_dir ./log/halfcheetah/test/GenIL_halfcheetah --custom_reward preference_normalized --ppo_log_path preference_norm_ctrl_coeff_011 --ctrl_coeff 0.1 --stochastic --rl_runs 5 --eval_rl```
```