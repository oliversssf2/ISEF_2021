# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
import numpy as np
from geneticalgorithm import geneticalgorithm as ga
import matplotlib.pyplot as plt
import numpy as np
import dplm_base

import time
dplm_instance = dplm_base.dplm('para1.csv')
dplm_instance.show_dplm_config()
dplm_instance.set_dplm_slot_num(10)
dplm_instance.set_dplm_spring_num(3)
dplm_instance.set_dplm_allowed_angle_range(-20, 60, 1)

dplm_instance.set_dplm_spring_constants([400,300,200])
dplm_instance.set_dplm_spring_lengths([0.2, 0.15, 0.1])

# dplm_instance.calculate_current_moment
# dplm_instance.set_slot([-6, 18, 0])
# moment_weight, moment_spring_list, moment_total = dplm_instance.calculate_current_moment()
# lower_limit, upper_limit, step_size, total_angle_num = dplm_instance.get_allowed_angle_range().values()
# 26 32 38
#13 37 38

install_position_step = 1e-2
spring_constant_step = 1e1
spring_length_step = 1e-2


def f(X):
    dplm_instance.set_springs_positions(np.array([X[0],X[1],X[2]])*install_position_step)
    dplm_instance.set_dplm_spring_lengths(np.array([X[3],X[4],X[5]])*spring_length_step)
    dplm_instance.set_dplm_spring_constants(np.array([X[6],X[7],X[8]])*spring_constant_step)
    return dplm_instance.current_rmse(False)

varbound=np.array([[-0.4/install_position_step,0.4/install_position_step]]*3+ [[0.2/spring_length_step,0.4/spring_length_step]]*3+ [[200/spring_constant_step, 400/spring_constant_step]]*3)
varbound = varbound.astype(int)

aps={'max_num_iteration': None,\
                'population_size':100,\
                'mutation_probability':0.1,\
                'elit_ratio': 0.01,\
                'crossover_probability': 0.5,\
                'parents_portion': 0.3,\
                'crossover_type':'uniform',\
                'max_iteration_without_improv':30}

model=ga(function=f,dimension=9,variable_type='int',variable_boundaries=varbound, algorithm_parameters=aps)


start = time.time()
print("Timer starts at {}".format(start))
model.run()
end = time.time()
print('timer ends at {}'.format(end))
print('time elapsed: {}'.format(end - start))

print("printing output dict")
for i in model.output_dict.items():
    print(i)

print("printing report")
for j in model.report:
    print(j)
