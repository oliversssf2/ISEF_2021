import gym
from gym import error, spaces, utils
from gym.utils import seeding

class DplmEnv(gym.Env):
    """
    Description: 
        Multiple spings (elastic material) are installed on a DPLM to achieve
        moment balance, cancelling out the effect of the gravitational force 
        acting on the mechanism. The goal is to find a installation of spring
        that 
    
    DPLM configuration:
        Each linkage: 20 slots for spring installation
        Number of springs: 3
        DPLM angle range(degree): [-30, 75] theta within Z
        Angle step: 1 degreee
    
    Ovservation:
        Type: Tuple(Discrete(39), Discrete(39), Discrete(39))
        
        Note: 39 = 2*20-1 because there are 39 results when we subtract the chosen 
        slot index of the upper linkage from the lower linkage (-20,-19,...,
        19,20)

    Action:
        Type: Tuple(Discrete(5), Discrete(5), Discrete(5))
        
        Note: Each of the three springs can either remain in the original slot (+0), 
        move to the left r right adjacent slot (+1/-1), move to the third slot 
        to its either side (+3/-3). So there are in total 5^n possible actions.
        (Tuple(5,5,...,5)(n in total))

    Reward:
        Reward is the RMSE of the difference between moment_total and 0 across 
        the angle range times negative 1. The reasion to multiply the RMSE by
        -1 is to ensure that the reward value is bigger (less negative) when 
        the RMSE is small, which indicates that the moment is roughly balance 
        across all angles.

    Starting state:
        All observations are assigned three discrete random value in [-21,21] 
        with replacement.

    Episode Termination:
        Episode length is geater than 100
        RMSE < 0.1??
        Solved Requirements:
        Considered solved when the average RMSE is smaller than 0.15 for 100
        consecutive trials.??
    """
    metadata = {'render.modes': ['human']}

    def __init__(self):
        pass
    def step(self, action):
        pass
    def reset(self):
        pass
    def render(self, mode='human'):
        pass
    def close(self):
        pass