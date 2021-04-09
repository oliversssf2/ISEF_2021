import gym
from gym import error, spaces, utils, logger
from gym.utils import seeding
import math
import numpy as np
import dplm_base
import matplotlib.pyplot as plt


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
        Type: MultiDiscrete[spring_num* (2*slot_num-1)]
        
        Note: 2*slot_num -1 because there are 2*slot_num-1 results when we 
        subtract the chosen slot index of the upper linkage from the lower 
        linkage (slot_num, slot_num-1, ..., -slot_num+1, -slot_num)

    Action:
        Type: MultiDiscrete[slot_num * 4] 
        
        Note: Each of the three springs can either move to the left or right
        adjacent slot (+1/-1), or move to the third slot to its either side
         (+3/-3). So there are in total slot_num*4 possible actions.
        Discrete(slot_num * 4)

        Order of actions:
        4k+0: -1
        4k+1: +1
        4k+2: -3
        4K+3: +3
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
        RMSE <= 2
        Solved Requirements:
        Considered solved when the average RMSE is smaller than 0.15 for 100
        consecutive trials.??

    State:
        The state of the dplm agent contains the slots the springs are installed on.
        For example: [-14, 6, 15]
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, dplm_config_file, spring_num = 3, slot_num = 20, \
                 spring_constants=[300, 300, 300], spring_init_lengths=[0.16,0.16,0.16],\
                 rmse_limit = 2, **allowed_angle_range):
        
        #Initialize the dplm agent with given parameters
        self.dplm_agent = dplm_base.dplm(dplm_config_file)
        self.dplm_agent.set_dplm_slot_num(slot_num)
        self.dplm_agent.set_dplm_spring_num(spring_num)
        self.dplm_agent.set_dplm_spring_constants(spring_constants)
        self.dplm_agent.set_dplm_spring_lengths(spring_init_lengths)
        self.dplm_agent.set_dplm_allowed_angle_range(allowed_angle_range['lower_limit'],\
                                                     allowed_angle_range['upper_limit'],\
                                                     allowed_angle_range['step_size'])

        #Value at which the episode is considererd to be successful
        self.rmse = None
        self.rmse_threshold = rmse_limit 
        self.action_per_spring_num = 8
        self.action_per_spring = [-1,1, -3, 3, -5, 5, -7, 7]

        self.possible_installation_num = 2*slot_num-1
        self.action_space = spaces.Discrete(spring_num * self.action_per_spring_num)
        self.observation_space = spaces.MultiDiscrete(np.array([self.possible_installation_num,
                                                       self.possible_installation_num,
                                                       self.possible_installation_num]))

        #Construct an action array as following (assuming 3 springs are installed):
        #[-1, 0, 0]
        #[1, 0, 0]
        #[-3, 0, 0]
        #[3, 0, 0]
        #[0, -1, 0]
        #[0, 1, 0]
        #...
        self.action_list = []
        for spring in range(spring_num):
            for action in range(self.action_per_spring_num):
                lst = [0]*spring_num
                lst[spring] = self.action_per_spring[action]
                self.action_list.append(lst)

        self.steps_beyond_done = None



        self.seed()

        #plotting to jupyter
        fig = plt.figure()
        plt.show(block=False)
        pass
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
   
    def step(self, action):
        #First check if the action is feasible (infeasible if the installation
        # position exceed the number of slots on the dplm (the observation space)).
        #
        #If the action is feasible, add it to the original state and call the
        #dplm agent to update its installaion positions of its springs, then 
        #calculate the rmse to be the reward.
        #
        #if the action is infeasible, set the reward to -10000 and not update the
        #state of the dplm agent.

        # print('Step: action is {}'.format(action))
        action_to_take = self.action_list[action]
        new_state = [sum(x) for x in zip(action_to_take, self.state)]
        
        if (((max(new_state))>(2*self.dplm_agent.get_slot_num()-2)) 
            or ((min(new_state))<0)):
            reward = -1
            done = True

        else:
            self.state = new_state
            self.dplm_agent.set_slot([x-self.dplm_agent.get_slot_num()+1 for x in new_state])
            self.rmse = self.dplm_agent.current_rmse()
            reward = 1/self.rmse
            # print('Step: new state is {}'.format(new_state))
            # print('Step: new positions are {}.'.format([x-self.dplm_agent.get_slot_num()+1 for x in new_state]))
            done = bool(
                not self.rmse
                and self.rmse < self.rmse_threshold
            ) 
  

        if not done:
            pass
        elif self.steps_beyond_done is None:
            # print("DONE!!")
            self.steps_beyond_done = 0
            # reward = 1 #??? should i do this or should i just use the RMSE
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_done += 1
            # reward = 0 #????
        
        return np.array(self.state), reward, done, {}
        
    
        pass
    def reset(self):
        slot_num = self.dplm_agent.get_slot_num()
        spring_num = self.dplm_agent.get_spring_num()
        self.state = [self.np_random.randint(0, 2*slot_num-1) for i in range(spring_num)] 
        self.dplm_agent.set_slot([x-slot_num+1 for x in self.state])
        self.steps_beyond_done = None

        # print('Reset: spring_num is {}. slot_num is{}.'.format(self.dplm_agent.get_spring_num(), self.dplm_agent.get_slot_num))
        # print('Reset: self state is {}. New positions are {}'.format(self.state,[x-slot_num+1 for x in self.state]))
        return np.array(self.state)

    def render(self, mode='human'):
        if mode == 'human':
            moment_weight, moment_spring_list, moment_total = self.dplm_agent.calculate_current_moment()
            lower_limit, upper_limit, step_size, total_angle_num = self.dplm_agent.get_allowed_angle_range().values()
            a,b,c = self.dplm_agent.calculate_current_moment()
            # %matplotlib widget
            plt.cla()    
            plt.plot(range(lower_limit, upper_limit+1), a, label = 'moment_weight', ls = '--', lw = 3, color = 'mediumaquamarine')

            ax = plt.gca()

            for i in range(len(moment_spring_list)):
                plt.plot(range(lower_limit,upper_limit+1), b[i], label = 'moment_spring_{}'.format(i+1), ls = '--', lw = 3, color = 'cornflowerblue')

            plt.plot(range(lower_limit, upper_limit+1), c, label = 'moment_total', ls = '--', lw = 3, color = 'mediumslateblue')
            plt.axhline(y = 0, ls = '-', lw = 3, color = 'darkgrey')

            plt.axis(ymin=-20, ymax=50)
            plt.legend()
            plt.xlabel('angle [degree]')
            plt.ylabel('moment [Nm]')
            ax.xaxis.set_major_formatter('{x}°')


            plt.text(-10,-10, r'$RMSE={:.2f}$'.format(self.dplm_agent.current_rmse()))

            plt.pause(0.001)
        elif mode=='text':
            pass
            # print('Render: state: {}, rmse: {}'.format(self.state, self.rmse))

    def close(self):
        pass
    
if __name__ == '__main__':
    print()