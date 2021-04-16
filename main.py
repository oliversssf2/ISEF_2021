import matplotlib.pyplot as plt
import numpy as np
import dplm_base


# dplm_instance.set_slot([-6, 18, 0])
# moment_weight, moment_spring_list, moment_total = dplm_instance.calculate_current_moment()
# lower_limit, upper_limit, step_size, total_angle_num = dplm_instance.get_allowed_angle_range().values()
# 26 32 38
#13 37 38

import gym
import math
import os
import time

from stable_baselines3 import DQN, A2C, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.cmd_util import make_vec_env

def setup1():

    cwd = os.getcwd()
    angle_range = {
        'lower_limit' : -40,
        'upper_limit' : 60,
        'step_size' : 1
    }

    env = gym.make('gym_dplm:dplm-v0', 
                    dplm_config_file = cwd+"/para1.csv",
                    spring_num = 3,
                    slot_num = 20,
                    spring_constants = [600,300,250],
                    spring_init_lengths = [0.2,0.2,0.13],
                    rmse_limit = 2,
                    **angle_range)

    # env = make_vec_env(lambda: env, n_envs=40)

    env = make_vec_env(lambda: env, n_envs=10)
    model = PPO('MlpPolicy', env, verbose=1, tensorboard_log='./dplm_tensorboards/ppo_tensorboard/')
    return env, model

def setup2():
    cwd = os.getcwd()
    angle_range = {
        'lower_limit' : -40,
        'upper_limit' : 60,
        'step_size' : 1
    }

    env = gym.make('gym_dplm:dplm-v0', 
                    dplm_config_file = cwd+"/para1.csv",
                    spring_num = 3,
                    slot_num = 20,
                    spring_constants = [600,300,250],
                    spring_init_lengths = [0.2,0.2,0.13],
                    rmse_limit = 2,
                    **angle_range)

    # env = make_vec_env(lambda: env, n_envs=40)

    # env = make_vec_env(lambda: env, n_envs=10)
    model_name = './rl_model/model_dqn'
    model = DQN('MlpPolicy', env, verbose=1, tensorboard_log='./dplm_tensorboards/dqn_2_tensorboard/')
    return env, model, model_name

def setup3():
    cwd = os.getcwd()
    angle_range = {
        'lower_limit' : -40,
        'upper_limit' : 60,
        'step_size' : 1
    }

    env = gym.make('gym_dplm:dplm-v0', 
                    dplm_config_file = cwd+"/para1.csv",
                    spring_num = 3,
                    slot_num = 20,
                    spring_constants = [600,300,250],
                    spring_init_lengths = [0.2,0.2,0.13],
                    rmse_limit = 2,
                    **angle_range)

    env = make_vec_env(lambda :env, n_envs=5)

    model = A2C('MlpPolicy', env, verbose=1, tensorboard_log='./dplm_tensorboards/a2c_tensorboard/')
    model_name = './rl_model/model_a2c'
    return env, model, model_name


   
if __name__ == '__main__':
    env,model, model_name = setup2()
    model.learn(50000)
    model.save(model_name)


    # model = A2C('MlpPolicy', env, verbose=1).learn(50000)
    # model.save('dplm')

    # Test the trained agent
    for i in range(10):
        print('Test: NO.{}'.format(i+1))
        obs = env.reset()
        n_steps = 50
        for step in range(n_steps):
            action, _ = model.predict(obs, deterministic=True)
            print("Step {}".format(step + 1))
            print("Action: ", action)
            obs, reward, done, info = env.step(action)
            print('obs=', obs, 'reward=', reward, 'done=', done)
            # env.render()
            if done.any():
                # Note that the VecEnv resets automatically
                # when a done signal is encountered
                print("Goal reached!", "reward=", reward)
                break
# cwd = os.getcwd()
# angle_range = {
#     'lower_limit' : -20,
#     'upper_limit' : 60,
#     'step_size' : 1
# }

# env = gym.make('gym_dplm:dplm-v0', 
#                 dplm_config_file = cwd+"/para1.csv",
#                 spring_num = 3,
#                 slot_num = 20,
#                 spring_constants = [300,300,300],
#                 spring_init_lengths = [0.16,0.16,0.16],
#                 rmse_limit = 3,
#                 **angle_range)

# # env = make_vec_env(lambda: env, n_envs=40)

# env = make_vec_env(lambda: env, n_envs=10)
# model = PPO('MlpPolicy', env, verbose=1)

