# Atari Experiments #


## GenIL for Atari ##


GenIL.py is the main file to run.

First unzip the model file in directory /models.

Here's an example of how to run it. 

```python GenIL.py --env_name breakout --reward_model_path ./learned_models/breakout_test.params --models_dir .```



## Visualizing learned reward functions ##
The plot visualization file is VisualizeAtariLearnedReward.py

Here's an example of how to run it. 

```python3 VisualizeAtariLearnedReward.py --env_name breakout --models_dir . --reward_net_path ./learned_models/breakout_test.params --save_fig_dir ./viz```

## RL on learned reward function ##

For a trained reward you can run RL to get policy.

Note: baselines must be installed

```
cd baselines
pip install -e .
```

You can run RL as follows:

```
OPENAI_LOG_FORMAT='stdout,log,csv,tensorboard' OPENAI_LOGDIR=[your_log_dir_here] python -m baselines.run --alg=ppo2 --env=[Atari env here] --custom_reward pytorch --custom_reward_path [path_to_learned_reward_model] --seed 0 --num_timesteps=5e7  --save_interval=500 --num_env 9
```

## Evaluation of learned policy ##

Use evaluateLearnedPolicy.py to evaluate the performance

For example:

```python evaluateLearnedPolicy.py --env_name breakout --checkpointpath [path_to_rl_checkpoint]```
