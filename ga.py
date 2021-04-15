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



def setup1(dplm_instance): 
    aps={'max_num_iteration': None,\
                'population_size':100,\
                'mutation_probability':0.2,\
                'elit_ratio': 0.01,\
                'crossover_probability': 0.6,\
                'parents_portion': 0.3,\
                'crossover_type':'uniform',\
                'max_iteration_without_improv':30}

    dplm_instance.show_dplm_config()
    dplm_instance.set_dplm_slot_num(10)
    dplm_instance.set_dplm_spring_num(3)
    dplm_instance.set_dplm_allowed_angle_range(-40, 60, 1)


    install_position_step = 1e-2
    spring_constant_step = 1e1
    spring_length_step = 1e-2

    # dplm_instance.set_dplm_spring_constants([400,300,200])
    # dplm_instance.set_dplm_spring_lengths([0.2, 0.15, 0.1])
    def f(X):
        dplm_instance.set_springs_positions(np.array([X[0],X[1],X[2]])*install_position_step)
        dplm_instance.set_dplm_spring_lengths(np.array([X[3],X[4],X[5]])*spring_length_step)
        dplm_instance.set_dplm_spring_constants(np.array([X[6],X[7],X[8]])*spring_constant_step)
        return dplm_instance.current_rmse()

    varbound=np.array([[-0.4/install_position_step,0.4/install_position_step]]*3+ [[0.2/spring_length_step,0.4/spring_length_step]]*3+ [[200/spring_constant_step, 400/spring_constant_step]]*3)
    varbound = varbound.astype(int)
    model=ga(function=f,dimension=9,variable_type='int',variable_boundaries=varbound, algorithm_parameters=aps)
    return model

def setup2(dplm_instance): 
    aps={'max_num_iteration': None,\
                'population_size':300,\
                'mutation_probability':0.2,\
                'elit_ratio': 0.01,\
                'crossover_probability': 0.5,\
                'parents_portion': 0.3,\
                'crossover_type':'uniform',\
                'max_iteration_without_improv':30}

    dplm_instance.show_dplm_config()
    dplm_instance.set_dplm_allowed_angle_range(-40, 60, 1)


    install_position_step = 1e-2
    spring_constant_step = 1e1
    spring_length_step = 1e-2

    # dplm_instance.set_dplm_spring_constants([400,300,200])
    # dplm_instance.set_dplm_spring_lengths([0.2, 0.15, 0.1])
    def f(X):
        dplm_instance.add_triangle(62.5*X[0], 0.6)
        dplm_instance.set_springs_positions([X[1], X[2]])
        val = dplm_instance.current_rmse()
        dplm_instance.rm_triangle()
        return val

    varbound=np.array([[1, 20]]+ [[-.40, .40]]*2)
    vartype=np.array([['int'],['real'],['real']])
    model=ga(function=f,dimension=3,
             variable_boundaries=varbound, algorithm_parameters=aps,
             variable_type_mixed=vartype)
    return model

def setup3(file): 
    dplm_instance = dplm_base.dplm(file)
    aps={'max_num_iteration': None,\
                'population_size':300,\
                'mutation_probability':0.2,\
                'elit_ratio': 0.01,\
                'crossover_probability': 0.5,\
                'parents_portion': 0.3,\
                'crossover_type':'uniform',\
                'max_iteration_without_improv':30}

    dplm_instance.show_dplm_config()
    dplm_instance.set_dplm_allowed_angle_range(-40, 60, 1)

    spring_constant = 52/2
    spring_initial_length = 0.184
    slot_num = 39

    install_position_step = 1e-2
    spring_constant_step = 1e1
    spring_length_step = 1e-2

    # dplm_instance.set_dplm_spring_constants([400,300,200])
    # dplm_instance.set_dplm_spring_lengths([0.2, 0.15, 0.1])
    dplm_instance.set_dplm_slot_num(slot_num)
    def f(X):
        dplm_instance.add_triangle(spring_constant*X[0], spring_initial_length)
        dplm_instance.set_slot([X[1], X[2]])
        val = dplm_instance.current_rmse()
        dplm_instance.rm_triangle()
        return val

    varbound=np.array([[1, 20]]+ [[-15, 16]]*2)
    vartype=np.array([['int'],['int'],['int']])
    model=ga(function=f,dimension=3,
             variable_boundaries=varbound, algorithm_parameters=aps,
             variable_type_mixed=vartype)
    return model

# dplm_instance = dplm_base.dplm('para1.csv')
model = setup3('para2.csv')


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

# %%
