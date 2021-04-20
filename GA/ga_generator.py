# %% [markdown]
# #GA_generator
# This script generates a set of optimum installation positions of spring using
#  genetic algorithm over a random variable space.
#
# ## Independent varible
# 1. Number of springs: range from 1 to 10. 
# 
# ## Random variables
# 1. Spring constants: randomized within a given range
# 2. Spring initial lengths: randomized within a given range
# 3. Extremity load mass: randomized within a given range
# 
# ## Outputs:
# ### in .csv
# Header: the number of spring, the name of the entries in the correct order
# Data: s_const[double[]], s_init_len[double[]], extremity_load_kg[double], s_inst_pos[double[]], rmse[double]
# 
# ### other
# The GA learning curve graph PNG (probably not because it will slow done the generation progress)
#
# ## Program control flow
# 1. import
# 2. dplm instance setup
# 3. genetic algorithm setup
# 4. pass the dplm instance, the GA object,a sample count, and a backup step into a loop function 
#
# ### Inside the loop function
# 1. set a counter
# 2. run the AG once (disable its plotting and progress bar function) and
# 3. if the backup step divides current count, save  
# f
#


# %%
#imports
from IPython import get_ipython

import sys
import os
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import csv
import shutil


import numpy as np
from numpy.random import default_rng
from geneticalgorithm import geneticalgorithm as ga
import matplotlib.pyplot as plt
import numpy as np
import dplm_base
import time
import random


# # %%
# # define ranges of random parameters
# # spring constants [150, 500] (N/m)
# s_c_range = np.array([150, 500])
# # spring initial length [0.2, 0.4] (m)
# s_l_range = np.array([0.15, 0.4])
# # extremity load [0, 5] (kg)
# e_l_range = np.array([0, 10])
# # number of springs (2 to six)
# s_num_range = np.array([2, 6])

# # optimization constraint
# # installation position: [-0.4, 0.4]
# i_p_range = np.array([-0.4, 0.4])
# i_p_step_size = 1e-2

# #angle range and step size (lower, upper, step)
# ang_ran = [-40, 60, 4]





def file_set_up():
    # file name prefix
    currentdir = os.getcwd()
    name = sys.argv[2]
    save_dir = os.path.join(currentdir, name)

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_file = os.path.join(save_dir, name)+'.csv'

    with open(save_file, mode = 'w+', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['s_num[int]', 's_const[N/m]', 's_len[m]', 'e_load','s_pos', 'rmse'])
    csvfile.close()

    return save_file

def setup_paras1():
    # define ranges of random parameters
    # spring constants [150, 500] (N/m)
    s_c_range = np.array([150, 500])
    # spring initial length [0.2, 0.4] (m)
    s_l_range = np.array([0.15, 0.4])
    # extremity load [0, 5] (kg)
    e_l_range = np.array([0, 10])
    # number of springs (2 to six)
    s_num_range = np.array([2, 6])

    # optimization constraint
    # installation position: [-0.4, 0.4]
    i_p_range = np.array([-0.4, 0.4])
    i_p_step_size = 1e-2

    #angle range and step size (lower, upper, step)
    ang_ran = [-40, 60, 4]

    # GA parameters
    aps={'max_num_iteration': None,\
                'population_size':100,\
                'mutation_probability':0.1,\
                'elit_ratio': 0.01,\
                'crossover_probability': 0.5,\
                'parents_portion': 0.3,\
                'crossover_type':'uniform',\
                'max_iteration_without_improv':30}

    # Objective sample size
    sample_size = 100000
    # Backup size
    backup_size = 50 

    dplm_instance = dplm_base.dplm('para1.csv')
    dplm_instance.set_dplm_allowed_angle_range(*ang_ran)
    
    # dplm_instance.set_dplm_spring_num(s_num)

    def fitness_func(X):
       dplm_instance.set_springs_positions(X*i_p_step_size)
       return dplm_instance.current_rmse() 
    rng = default_rng()

    return s_c_range, s_l_range, e_l_range, s_num_range, i_p_range, i_p_step_size, ang_ran, aps, sample_size, backup_size, dplm_instance, fitness_func, rng



def setup_paras2():
    # define ranges of random parameters
    # spring constants [150, 500] (N/m)
    s_c_range = np.array([150, 500])
    # spring initial length [0.15, 0.4] (m)
    s_l_range = np.array([0.15, 0.4])
    # extremity load [0, 5] (kg)
    e_l_range = np.array([0, 10])
    # number of springs (2 to six)
    s_num_range = np.array([5, 5])

    # optimization constraint
    # installation position: [-0.4, 0.4]
    i_p_range = np.array([-0.4, 0.35])
    i_p_step_size = 1e-2

    #angle range and step size (lower, upper, step)
    ang_ran = [-40, 60, 4]

    # GA parameters
    aps={'max_num_iteration': None,\
                'population_size':3000,\
            'mutation_probability':0.1,
                'elit_ratio': 0.05,\
                'crossover_probability': 0.5,\
                'parents_portion': 0.3,\
                'crossover_type':'uniform',\
                'max_iteration_without_improv':200} 
    # aps={'max_num_iteration': None,\
    #             'population_size':1500,\
    #             'mutation_probability':0.01,\
    #             'elit_ratio': 0.02,\
    #             'crossover_probability': 0.5,\
    #             'parents_portion': 0.3,\
    #             'crossover_type':'uniform',\
    #             'max_iteration_without_improv':200}
    # aps={'max_num_iteration': None,\
    #             'population_size':300,\
    #             'mutation_probability':0.01,\
    #             'elit_ratio': 0.01,\
    #             'crossover_probability': 0.3,\
    #             'parents_portion': 0.3,\
    #             'crossover_type':'uniform',\
    #             'max_iteration_without_improv':50}
    # Objective sample size
    sample_size = 100000
    # Backup size
    backup_size = 50 

    dplm_instance = dplm_base.dplm('para1.csv')
    dplm_instance.set_dplm_allowed_angle_range(*ang_ran)
    
    # dplm_instance.set_dplm_spring_num(s_num)

    def fitness_func(X):
    #    dplm_instance.set_springs_positions(X*i_p_step_size)
       dplm_instance.set_springs_positions(X)
       return dplm_instance.current_rmse() 
    rng = default_rng()

    return s_c_range, s_l_range, e_l_range, s_num_range, i_p_range, i_p_step_size, ang_ran, aps, sample_size, backup_size, dplm_instance, fitness_func, rng


def setup_paras3():
    # define ranges of random parameters
    # spring constants [150, 500] (N/m)
    s_c_range = np.array([75, 250])
    # spring initial length [0.2, 0.4] (m)
    s_l_range = np.array([0.3, 0.8])
    # extremity load [0, 5] (kg)
    e_l_range = np.array([0, 10])
    # number of springs (2 to six)
    s_num_range = np.array([2, 2])

    # optimization constraint
    # installation position: [-0.4, 0.4]
    i_p_range = np.array([-0.4, 0.4])
    i_p_step_size = 1e-2

    #angle range and step size (lower, upper, step)
    ang_ran = [-40, 60, 4]

    # GA parameters
    aps={'max_num_iteration': None,\
                'population_size':3000,\
            'mutation_probability':0.01,
                'elit_ratio': 0.05,\
                'crossover_probability': 0.5,\
                'parents_portion': 0.3,\
                'crossover_type':'uniform',\
                'max_iteration_without_improv':200} 

     # Objective sample size
    sample_size = 100000
    # Backup size
    backup_size = 50

    dplm_instance = dplm_base.dplm('para1.csv')
    dplm_instance.set_dplm_allowed_angle_range(*ang_ran)
    
    # dplm_instance.set_dplm_spring_num(s_num)
    # def f(X):
    #     dplm_instance.add_triangle(26*X[0], 0.185)
    #     dplm_instance.set_slot([X[1], X[2]])
    #     val = dplm_instance.current_rmse()
    #     dplm_instance.rm_triangle()
    #     return val
    # varbound = np.array()
    def fitness_func(X):
        init_len = dplm_instance.get_spring_init_lengths()
        # print('lenght: {}'.format(init_len))
        spring_con = dplm_instance.get_spring_constatnts()
        # print('spring_con: {}'.format(spring_con))
        dplm_instance.set_dplm_spring_lengths([])
        dplm_instance.set_dplm_spring_constants([])
        # dplm_instance.set_dplm_spring_num(0)
        dplm_instance.add_triangle(float(spring_con[0])*int(X[0]), float(init_len[0]))
        dplm_instance.set_springs_positions(list(X[1:]))
        val = dplm_instance.current_rmse()
        dplm_instance.rm_triangle()

        dplm_instance.set_dplm_spring_num(2)
        dplm_instance.set_dplm_spring_lengths(init_len)
        dplm_instance.set_dplm_spring_constants(spring_con)
        dplm_instance.set_dplm_spring_num(0)
        return val
    rng = default_rng()

    return s_c_range, s_l_range, e_l_range, s_num_range, i_p_range, i_p_step_size, ang_ran, aps, sample_size, backup_size, dplm_instance, fitness_func, rng
# %%
if __name__ == "__main__":
    save_file = file_set_up()
    s_c_range, s_l_range, e_l_range, s_num_range, i_p_range,\
         i_p_step_size, ang_ran, aps, sample_size, backup_size,\
              dplm_instance, fitness_func, rng = setup_paras3()

    
    sample_count = 0
    #write the header
    

    buffer = []
    for i in range(sample_size):
        print('sample count is {}'.format(sample_count))
        start = time.time() #start timer
        # generate random parameters and change the state of the dplm with them
        s_num = int(rng.integers(*s_num_range, endpoint = True))
        s_c = rng.uniform(*s_c_range, s_num)
        s_l = rng.uniform(*s_l_range, s_num)
        e_l = rng.uniform(*e_l_range)
        
        dplm_instance.set_dplm_spring_num(s_num) #can't do this when using triangle
        dplm_instance.set_dplm_spring_constants(s_c)
        dplm_instance.set_dplm_spring_lengths(s_l)
        dplm_instance.set_extremity_load(e_l)
        
        #for triangle
        dplm_instance.set_dplm_spring_num(0)
        
        # varbound=np.array([[i_p_range[0]/i_p_step_size,i_p_range[1]/i_p_step_size]]*s_num)
        
        # This one is for optimizing separate springs
        # varbound=np.array([[i_p_range[0],i_p_range[1]]]*s_num)

        #This one is for optimizain rubber bands
        # dplm_instance.add_triangle
        varbound = np.array([[0,6]]+[[i_p_range[0], i_p_range[1]]]*2)
        vartype = np.array([['int']]+[['real']]*s_num)

        model=ga(function=fitness_func,
                dimension=s_num+1,
                variable_type_mixed=vartype,
                # variable_type=vartype,
                variable_boundaries=varbound,
                algorithm_parameters=aps,
                convergence_curve=False,
                progress_bar = True)   
        
        model.run()
        export = [s_num,
                  *s_c, 
                  *s_l, 
                  e_l,
                  *(model.output_dict['variable']),
                  model.output_dict['function']]
        # print('exporting: {}'.format(export))
        buffer.append(export)

        del model 
        sample_count+=1
        
        end = time.time() #end timer and print 
        print('time elapsed for one sample: {}'.format(end-start))

        if (sample_count%backup_size==0):
            with open(save_file, mode = 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                for entry in buffer:
                    writer.writerow(entry)
            csvfile.close()
            buf_file = '{}_buf_{}'.format(save_file, int(sample_count/backup_size))
            shutil.copy(save_file, buf_file)
            print('creating buffer: {}'.format(buf_file))
            buffer = []
            
            
