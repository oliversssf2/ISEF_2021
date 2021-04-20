import dplm_base
import numpy as np 
import matplotlib.pyplot as plt
import csv
import pandas as pd
import os
from pathlib import Path
import time



def ensure_path(path_str):
    cwd = Path.cwd()
    path = Path.joinpath(cwd, path_str)
    try:
        Path.mkdir(path, parents=True)
    except FileExistsError:
        path_str = input('This path [{}] already exists. Enter a new folder name'.format(path_str))
        path = Path.joinpath(cwd, path_str)

    return path
def greedy1(dplm_instance):
    start = time.time()
    init_guess = np.random.randint(0, high=dplm_instance.get_slot_num()*2-1, size=dplm_instance.get_spring_num())-dplm_instance.get_slot_num()+1
    guess = np.array(init_guess, copy=True)
    for greedy_iter_num in range(5): #three iterations for each greedy
        for ind in range(dplm_instance.get_spring_num()):
            rmse = np.zeros(dplm_instance.get_slot_num())
            for slot in range(-dplm_instance.get_slot_num()+1, dplm_instance.get_slot_num()):
                guess[ind] = slot
                dplm_instance.set_slot(guess)
                rmse[slot] = dplm_instance.current_rmse()
                guess[ind] = np.argmin(rmse)
        dplm_instance.set_slot(guess)
    final_rmse = dplm_instance.current_rmse()
    final_guess = guess
    end = time.time()
    time_elapsed = end-start
    return init_guess, final_guess, final_rmse, time_elapsed
def greedy2(dplm_instance, s_c_range, s_c_step, s_l_range, s_l_step):
    dplm_instance.set_dplm_spring_num(6)
    dplm_instance.set_dplm_slot_num(20)
    spring_num = dplm_instance.get_spring_num()
    slot_num = dplm_instance.get_slot_num()
    init_guess = [np.random.randint(0, high=slot_num*2-1, size=spring_num)-slot_num+1,
                  np.random.randint(0, high = int((s_c_range[1]-s_c_range[0])/s_c_step)+1, size=spring_num)*s_c_step+s_c_range[0],
                  np.random.randint(0, high = int((s_l_range[1]-s_l_range[0])/s_l_step)+1, size = spring_num)*s_l_step+s_l_range[0]]
    
    
    guess = np.array(init_guess, copy=True)
    start = time.time()
    for greedy_iter_num in range(1): #three iterations for each greedy
        for ind in range(dplm_instance.get_spring_num()):
            rmse = np.zeros(dplm_instance.get_slot_num())
            for slot in range(-dplm_instance.get_slot_num()+1, dplm_instance.get_slot_num()):
                guess[0][ind] = slot
                dplm_instance.set_slot(list(guess[0]))
                rmse[slot] = dplm_instance.current_rmse()
            guess[0][ind] = np.argmin(rmse)
            rmse_2 = np.zeros(int((s_c_range[1]-s_c_range[0])/s_c_step)+1)
            for s_c in range(int((s_c_range[1]-s_c_range[0])/s_c_step)+1):
                guess[1][ind] = s_c_range[0]+s_c_step*s_c
                dplm_instance.set_dplm_spring_constants(list(guess[1]))
                rmse_2[s_c] = dplm_instance.current_rmse()
            guess[1][ind] = s_c_range[0] + s_c_step*np.argmin(rmse_2)
            rmse_3 = np.zeros(int((s_l_range[1]-s_l_range[0])/s_l_step)+1)
            for s_l in range(int((s_l_range[1]-s_l_range[0])/s_l_step)+1):
                guess[2][ind] = s_l_range[0]+s_l_step*s_l
                dplm_instance.set_dplm_spring_lengths(list(guess[2]))
                rmse_3[s_l] = dplm_instance.current_rmse()
            guess[2][ind] = s_l_range[0]+s_l_step*np.argmin(rmse_3)

        dplm_instance.set_slot(guess[0])
        dplm_instance.set_dplm_spring_constants(guess[1])
        dplm_instance.set_dplm_spring_lengths(guess[2])
    final_rmse = dplm_instance.current_rmse()
    for i in range(len(guess)):
        guess[i] = list(guess[i])
    final_guess = guess
    end = time.time()
    time_elapsed = end-start
    return init_guess, final_guess, final_rmse, round(time_elapsed,2)
def plot_gragh(path, sample_count):
    lower_limit, upper_limit, step_size, total_angle_num = dplm_instance.get_allowed_angle_range().values()

    a,b,c, rmse = dplm_instance.calculate_current_moment()
    ax = plt.gca()
    ax.cla()    
    ax.set_title('Optimizing positions for 4 RBs',fontweight="semibold")
    # plt.figure()
    ax.plot(range(lower_limit, upper_limit+1), a, label = 'M_W', ls = '--', lw = 3, color = 'grey')

    for i in range(len(b)):
        ax.plot(range(lower_limit,upper_limit+1), b[i], label = 'M_RB{}'.format(i+1), ls = '--', lw = 3, color = 'cornflowerblue')

    ax.plot(range(lower_limit, upper_limit+1), c, label = 'M_NET', ls = '--', lw = 3, color = 'red')
    ax.axhline(y = 0, ls = '-', lw = 3, color = 'darkgrey')

    ax.axis(ymin=-20, ymax=50)
    ax.axis(xmin = -20, xmax = 60)
    ax.legend()
    plt.xlabel('Angle [degree]')
    plt.ylabel('Moment [Nm]')
    ax.xaxis.set_major_formatter("{x}°")


    ax.text(-17,-15, 'RMSE={:.2f} \nInitial random state: {} \nFinal install positions: {}'.format(rmse,init_guess, guess))
    plt.savefig(Path.joinpath(path,'test_{}.png'.format(sample_count+1)))
    # plt.show()
    plt.pause(0.001)
    # del fig
    
    print(rmse) 
def plot_gragh2(path, sample_count):
    lower_limit, upper_limit, step_size, total_angle_num = dplm_instance.get_allowed_angle_range().values()

    a,b,c, rmse = dplm_instance.calculate_current_moment()
    ax = plt.gca()
    ax.cla()    
    # plt.figure()
    ax.plot(range(lower_limit, upper_limit+1), a, label = 'moment_weight', ls = '--', lw = 1, color = 'grey')

    for i in range(len(b)):
        ax.plot(range(lower_limit,upper_limit+1), b[i], label = 'moment_spring_{}'.format(i+1), ls = '-', lw = 1, color = 'cornflowerblue')

    ax.plot(range(lower_limit, upper_limit+1), c, label = 'moment_total', ls = '-', lw =4, color = 'gold')
    ax.axhline(y = 0, ls = '-', lw = 3, color = 'darkgrey')

    ax.axis(ymin=-20, ymax=50)
    ax.legend()
    plt.xlabel('angle [degree]')
    plt.ylabel('moment [Nm]')
    ax.xaxis.set_major_formatter("{x}°")


    ax.text(.1,.85, 'RMSE={:.2f} \nInitial random state: {} \
                      \nFinal install positions: {}\
                      \nFinal spring constant: {}\
                      \nFinal spring length: {}'.format(rmse,init_guess, guess[0], guess[1], guess[2]),
                      horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
    plt.savefig(Path.joinpath(path,'test_{}.png'.format(sample_count+1)))
    # plt.show()
    plt.pause(0.001)
    # del fig
    
    print(rmse) 

#settings: f is the greedy function to use
plotting = True
save_path_str = 'greedy_graphs/4_springs_5_loops_for_ppt'
sample_size = 100
f = greedy1
plot_func = plot_gragh


if __name__ == '__main__':
    dplm_instance = dplm_base.dplm('para1.csv')
    dplm_instance.show_dplm_config()
    dplm_instance.set_dplm_slot_num(20)
    dplm_instance.set_dplm_spring_num(4)
    # dplm_instance.set_slot([-4, 13, 8])
    dplm_instance.set_dplm_spring_constants([600,300,250, 230])
    dplm_instance.set_dplm_spring_lengths([0.1, 0.2, 0.17, .13])
    dplm_instance.set_dplm_allowed_angle_range(-20, 60, 1)

    spring_constant_range = [200, 800]
    spring_constant_step = 10
    spring_length_range = [0.15, 0.3]
    spring_length_step = 0.1
    
    greedy_timer = []

    path = ensure_path(save_path_str)

    with open(Path.joinpath(path, 'greedy.csv'), mode='w+', newline='') as csvfile:
        #header
        csvfile.writelines('number of springs: {}\n'.format(dplm_instance.get_spring_num()))
        csvfile.writelines('Number of slots: {}\n'.format(dplm_instance.get_slot_num()))
        csvfile.writelines('spring_constants: {}\n'.format(dplm_instance.get_spring_constatnts()))
        csvfile.writelines('spring_lengths: {}\n'.format(dplm_instance.get_spring_init_lengths()))
        
        #csv settings
        writer = csv.writer(csvfile, delimiter = ',', quotechar = '"')
        writer.writerow(['rmse', 'initial guess', 'final install positions', 'time elapsed'])

        #Start interactive plotting
        # plt.ion()
        fig = plt.figure(figsize=[5, 4])

        for sample_count in range(sample_size):
            # init_guess, guess, rmse, time_elapsed= f(dplm_instance,spring_constant_range, spring_constant_step, spring_length_range, spring_length_step)
            init_guess, guess, rmse, time_elapsed= f(dplm_instance)
            greedy_timer.append(time_elapsed)
            
            print('time elapsed for this iteration is {}s'.format(time_elapsed))
            writer.writerow([f'{rmse:.2f}', list(init_guess), list(guess), time_elapsed])

            if plotting == True:
                plot_func(path, sample_count)

    print(np.array(time_elapsed).mean())
    k = pd.read_csv(Path.joinpath(path,'greedy.csv'), header = 4)

    k.index = np.arange(1, len(k)+1)
    k.index.name='sample'
    k.to_excel(Path.joinpath(path,'greedy.xlsx'))
    k.to_csv(Path.joinpath(path,'greedy_processed.csv'))
    k.to_clipboard(sep = ',')

