import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.sputils import upcast
import dplm_base
import math
from scipy.stats import gaussian_kde
import matplotlib.cm as cm

from multiprocessing import Process

def func_1():
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
            rmse.append(dplm_instance.current_rmse_only_springs())

        xy = np.vstack([s_pos_flattend,rmse])
        z = gaussian_kde(xy)(xy)     
        axis.scatter(s_pos_flattend, rmse, c=z, s=1)
    plt.show()

def rmse_dist_tri():
    #RMSE Distribution
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    # pos = [[0,0], [0,1],[0,2], [1,0], [1,1],[1,2]]


    # %gui osx
    # %matplotlib


    ga_data = pd.read_csv('./GA/GA data/ga_triangles_3_processed.csv')
    ga_data.columns = ['s_num', 's_c', 's_l', 'e_l', 'tri_num', 'i_pos', 'rmse']
    good_ga_data = ga_data[ga_data['rmse']<=2]
    # print(good_ga_data)
    # good_ga_data.to_csv('./GA/GA data/good_ombined_copy_proccessed.csv', index=False)

    fig = plt.gcf()
    ax = plt.gca()
    fig.suptitle('RMSE distribution RB triangles')
    # plt.tight_layout(pad = 2)

    # ga_for_diff_s_num = []
    count = 0

    # ax = plt.subplot(1,1)
    # ax = axes[pos[i-2][0]][pos[i-2][1]]
    # ax.set_title('{} springs'.format(i))
    k = good_ga_data
    # print("the mean rmse for the triangle is {}".format(k['rmse'].mean()))
    dis_str = 'Mean RMSE = {:.2f}\n Median RMSE = {:.2f}\n sample size: {}'.format(k['rmse'].mean(),k['rmse'].median(), (k.count()['rmse']+1)*5)
    ax.text(0.5, 0.5, horizontalalignment='center',
                                        verticalalignment='center', transform =                                               ax.transAxes,
                                        s = dis_str)
    ax.hist(k['rmse'], bins = 50)
    plt.show()
def dens_dist_tri():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import dplm_base
    import math
    from scipy.stats import gaussian_kde

    axis = plt.gca()
    fig = plt.gcf()


    axis.set_title('Triangle')
    axis.set_xlabel('Installation position [m]')
    axis.set_ylabel('Net rubber band RMSE')
    axis.set_xlim(-0.4, 0.4)
    # axis.set_ylabel('Net rubber band RMSE')
    # axis.set_xlim(-0.4, 0.4)

    dplm_instance = dplm_base.dplm('para1.csv')
    # dplm_instance.set_dplm_spring_num(1)
    # dplm_instance.set_slot([-4, 13, 8])

    dplm_instance.set_dplm_allowed_angle_range(-40, 60, 1)

    ga_data = pd.read_csv('./GA/GA data/ga_triangles_3_processed.csv')
    ga_data.columns = ['s_num', 's_c', 's_l', 'e_l', 'tri_num', 'i_pos', 'rmse']
    good_ga_data = ga_data[ga_data['rmse']<=2]
    good_ga_data.columns = ['s_num', 's_c', 's_l', 'e_l', 'tri_num', 'i_pos', 'rmse']

    # print(good_ga_data)

    _s_c_flattened = [float(x) for x in good_ga_data['s_c']]
    s_c_flattened = []
    for x in _s_c_flattened:
        s_c_flattened.extend([x,x])

    _s_l_flattened = [float(x) for x in good_ga_data['s_l']]
    s_l_flattened = []
    for x in _s_l_flattened:
        s_l_flattened.extend([x,x])

    _e_l_flattened = [float(x) for x in good_ga_data['e_l']] 
    e_l_flattened = []
    for x in _e_l_flattened: 
        e_l_flattened.extend([x,x])

    _tri_num_flattened = [float(x) for x in good_ga_data['tri_num']]
    tri_num_flattened = []
    for x in _tri_num_flattened:
        tri_num_flattened.extend([x,x])



    # for i in 

    # def find_similar_values(s_pos, rmse):

    s_pos = [[float(x) for x in good_ga_data['i_pos'].iloc[i][1:-1].split(',')] for i in range(good_ga_data.shape[0])]
    # print(len(s_pos))
    # print(s_pos)

    l_half = 0
    u_half = 0
    s_pos_flattened = []
    for s in s_pos:
        for pos in s:
            s_pos_flattened.append(pos)
            if(pos>=0.2):
                u_half+=1
            else: 
                l_half+=1
    print('Number of rubber bands with installation position >=0.2: {}'.format(u_half))
    print('Number of rubber bands with installation position <0.2: {}'.format(l_half))

    # print(s_pos_flattened[0:100])

    # print(len(s_c_flattened))
    # print(len(s_l_flattened))
    # print(len(s_pos_flattened))
    rmse = []
    for ind in range(int(len(s_c_flattened)/2)):
        dplm_instance.add_triangle(tri_num_flattened[2*ind]*s_c_flattened[2*ind], s_l_flattened[2*ind])
        # dplm_instance.set_dplm_spring_constants([s_c_flattened[ind]*tri_num_flattened[ind]])
        # dplm_instance.set_dplm_spring_lengths([s_l_flattened[ind]])
        dplm_instance.set_springs_positions([s_pos_flattened[2*ind], s_pos_flattened[2*ind+1]])
        rmse.append(dplm_instance.current_rmse_only_springs_triangle()[0])
        rmse.append(dplm_instance.current_rmse_only_springs_triangle()[1])
        dplm_instance.rm_triangle()
    # print()

    # print(rmse)

    xy = np.vstack([s_pos_flattened,rmse])
    z = gaussian_kde(xy)(xy)    
    plt.scatter(s_pos_flattened, rmse, c=z, s=1)
    plt.axvline(0.2, color = 'red')
    plt.text(-0.1, 135, 'RBs with D>=0.2:\n{}'.format(l_half), horizontalalignment='center')
    plt.text(0.3, 135, 'RBs with D<0.2:\n{}'.format(u_half), horizontalalignment='center')
    plt.colorbar()
    plt.show()
def dens_dist_mul():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import dplm_base
    import math
    from scipy.stats import gaussian_kde

    dplm_instance = dplm_base.dplm('para1.csv')
    dplm_instance.set_dplm_spring_num(1)
    # dplm_instance.set_slot([-4, 13, 8])

    dplm_instance.set_dplm_allowed_angle_range(-40, 60, 1)

    ga_data = pd.read_csv('./GA/GA data/combined_copy_proccessed.csv')
    ga_data.columns = ['s_num', 's_c', 's_l', 'e_l','s_pos', 'rmse']
    good_ga_data = ga_data[ga_data['rmse']<=2]
    good_ga_data.columns = ['s_num', 's_c', 's_l', 'e_l','s_pos', 'rmse']


    # def find_similar_values(s_pos, rmse):
        


    fig, axes = plt.subplots(1,3)
    fig.suptitle('Distribution of installation positions and net RMSEs of RBs')
    count = 0
    for spring_num in [2,3,6]:
        count+=1
        axis = plt.subplot(1,3,count)
        axis.set_title('RB {}'.format(spring_num))
        axis.set_xlabel('Installation position [m]')
        axis.set_ylabel('Net rubber band RMSE')
        axis.set_xlim(-0.4, 0.4)
        axis.set_ylim(0,30)
        ga = good_ga_data[good_ga_data['s_num']==spring_num]
        s_pos = [[float(x) for x in ga['s_pos'].iloc[i][1:-1].split(',')] for i in range(ga.shape[0])]
        s_c = [[float(x) for x in ga['s_c'].iloc[i][1:-1].split(',')] for i in range(ga.shape[0])]
        s_l = [[float(x) for x in ga['s_l'].iloc[i][1:-1].split(',')] for i in range(ga.shape[0])]
        
        t_count = 0
        around_0_count = 0

        s_pos_flattend = []
        for i in s_pos:
            for j in i:
                if not math.isnan(j):
                    t_count+=1
                    if(-0.05<=j<=0.05): around_0_count+=1
                    s_pos_flattend.append(j)
        
        print('Number of RBs close to zero:{}'.format(around_0_count))
        print("Total number of RBs: {}".format(t_count))
        
        s_c_flattend = []
        for i in s_c:
            for j in i:
                if not math.isnan(j):
                    s_c_flattend.append(j)

        s_l_flattend = []
        for i in s_l:
            for j in i:
                if not math.isnan(j):
                    s_l_flattend.append(j)

        rmse = []
        for ind in range(len(s_l_flattend)):
            dplm_instance.set_dplm_spring_constants([s_c_flattend[ind]])
            dplm_instance.set_dplm_spring_lengths([s_l_flattend[ind]])
            dplm_instance.set_springs_positions([s_pos_flattend[ind]])
            rmse.append(dplm_instance.current_rmse_only_springs())

        # xy = np.vstack([s_pos_flattend[0:100],rmse[0:100]])
        # z = gaussian_kde(xy)(xy)    
        # plt.scatter(s_pos_flattend[0:100], rmse[0:100], c=z, s=1, cmap='viridis')

        # plt.hist2d(s_pos_flattend,rmse,cmap='viridis')
        
        xy = np.vstack([s_pos_flattend,rmse])
        z = gaussian_kde(xy)(xy)   
        plt.scatter(s_pos_flattend, rmse, c=z, s=1, cmap='viridis')
        plt.axvline(0.2, color = 'red')
        plt.text(0, 30, 'Total RBs number:{}\n RBs close to zero\n(-0.05<=x<=0.05):\n{}'.format(t_count, around_0_count), horizontalalignment='center')
        plt.colorbar()
    # plt.colorbar(new_z, orientation='vertical')
    plt.show()
def rmse_dist_mul():
    #RMSE Distribution
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    pos = [[0,0], [0,1],[0,2], [1,0], [1,1],[1,2]]

    ga_data = pd.read_csv('./GA/GA data/combined_copy_proccessed.csv')
    ga_data.columns = ['s_num', 's_c', 's_l', 'e_l','s_pos', 'rmse']
    good_ga_data = ga_data[ga_data['rmse']<=2]
    # good_ga_data = pd.DataFrame(np.repeat(good_ga_data.values,5,axis=0))
    # good_ga_data.to_csv('./GA/GA data/good_ombined_copy_proccessed.csv', index=False)

    fig, axes = plt.subplots(1,3)
    fig.suptitle('RMSE distribution for 2 to 6 rubber bands', fontweight='semibold')
    plt.tight_layout(pad = 1)

    # ga_for_diff_s_num = []
    count = 0
    for i in [2,3,6]:
        count+=1
        ax = plt.subplot(1,3,count)
        # ax = axes[pos[i-2][0]][pos[i-2][1]]
        ax.set_title('{} RBs'.format(i))
        k = good_ga_data[good_ga_data['s_num'] == i]
        temp = k
        k = pd.DataFrame(np.repeat(temp.values,5,axis=0))
        k.columns = temp.columns
        print("the mean rmse for {} springs is {}".format(i, k['rmse'].mean()))
        # dis_str = 'Mean RMSE = {:.2f}\n Median RMSE = {:.2f}\n sample size: {}'.format(k['rmse'].mean(),k['rmse'].median(), (k.count()['rmse']+1)*5)
        dis_str = 'Mean \nRMSE\n={:.2f}'.format(k['rmse'].mean())

        ax.text(0.1, 0.9, horizontalalignment='left',
                                            verticalalignment='center', transform =                                               ax.transAxes,
                                            s = dis_str)
        ax.set_xlabel('RMSE')
        ax.set_ylabel('sample')
        ax.set_xlim(0,2)
        ax.hist(k['rmse'], bins = 25)
    plt.show()

pgs = [rmse_dist_tri,dens_dist_tri,rmse_dist_mul,dens_dist_mul]

main = dens_dist_tri


if __name__ == '__main__':
    main()
