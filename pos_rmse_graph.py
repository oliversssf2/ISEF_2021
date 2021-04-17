import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import dplm_base
import math
from scipy.stats import gaussian_kde

if __name__ == '__main__':
    dplm_instance = dplm_base.dplm('para1.csv')
    dplm_instance.set_dplm_spring_num(1)
    # dplm_instance.set_slot([-4, 13, 8])

    dplm_instance.set_dplm_allowed_angle_range(-40, 60, 1)

    ga_data = pd.read_csv('./GA/GA data/combined_copy_proccessed.csv')
    ga_data.columns = ['s_num', 's_c', 's_l', 'e_l','s_pos', 'rmse']
    good_ga_data = ga_data[ga_data['rmse']<=2]
    good_ga_data.columns = ['s_num', 's_c', 's_l', 'e_l','s_pos', 'rmse']

    fig, axes = plt.subplots(2,3)
    for spring_num in range(2,7):
        axis = plt.subplot(2,3,spring_num-1)
        ga = good_ga_data[good_ga_data['s_num']==spring_num]
        s_pos = [[float(x) for x in ga['s_pos'].iloc[i][1:-1].split(',')] for i in range(ga.shape[0])]
        s_c = [[float(x) for x in ga['s_c'].iloc[i][1:-1].split(',')] for i in range(ga.shape[0])]
        s_l = [[float(x) for x in ga['s_l'].iloc[i][1:-1].split(',')] for i in range(ga.shape[0])]
        

        s_pos_flattend = []
        for i in s_pos:
            for j in i:
                # print('j is {}'.format(j))
                if not math.isnan(j):
                    s_pos_flattend.append(j)
        # print(s_pos_flattend)
        
        s_c_flattend = []
        for i in s_c:
            for j in i:
                # print('j is {}'.format(j))
                if not math.isnan(j):
                    s_c_flattend.append(j)

        s_l_flattend = []
        for i in s_l:
            for j in i:
                # print('j is {}'.format(j))
                if not math.isnan(j):
                    s_l_flattend.append(j)

        rmse = []
        for ind in range(len(s_l_flattend)):
            dplm_instance.set_dplm_spring_constants([s_c_flattend[ind]])
            dplm_instance.set_dplm_spring_lengths([s_l_flattend[ind]])
            dplm_instance.set_springs_positions([s_pos_flattend[ind]])
            rmse.append(dplm_instance.current_rmse())

        xy = np.vstack([s_pos_flattend,rmse])
        z = gaussian_kde(xy)(xy)    
        axis.scatter(s_pos_flattend, rmse, c=z, s=1)
    plt.show()