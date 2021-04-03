from gym.envs.registration import register

register(
    id='dplm-v0',
    entry_point='gym_dplm.envs:DplmEnv',
)
