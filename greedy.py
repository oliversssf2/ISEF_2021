import dplm_base
import numpy as np 
import matplotlib.pyplot as plt
import csv
import pandas as pd


plotting = True

if __name__ == '__main__':
    dplm_instance = dplm_base.dplm('para1.csv')
    dplm_instance.show_dplm_config()
    dplm_instance.set_dplm_slot_num(20)
    dplm_instance.set_dplm_spring_num(4)
    # dplm_instance.set_slot([-4, 13, 8])
    dplm_instance.set_dplm_spring_constants([600,300,250, 230])
    dplm_instance.set_dplm_spring_lengths([0.1, 0.2, 0.17, .13])
    dplm_instance.set_dplm_allowed_angle_range(-40, 60, 1)


    with open('greedy_graphs/4_springs/greedy.csv', mode='w+', newline='') as csvfile:
        csvfile.writelines('number of springs: {}\n'.format(dplm_instance.get_spring_num()))
        csvfile.writelines('Number of slots: {}\n'.format(dplm_instance.get_slot_num()))
        csvfile.writelines('spring_constants: {}\n'.format(dplm_instance.get_spring_constatnts()))
        csvfile.writelines('spring_lengths: {}\n'.format(dplm_instance.get_spring_init_lengths()))
        
        writer = csv.writer(csvfile, delimiter = ',', quotechar = '"')
        # writer.writerow('"number of springs: {}"'.format(dplm_instance.get_spring_num()))
        # writer.writerow('"Number of slots: {}"'.format(dplm_instance.get_slot_num()))
        # writer.writerow('"spring_constants: {}"'.format(dplm_instance.get_spring_constatnts()))
        # writer.writerow('"spring_lengths: {}"'.format(dplm_instance.get_spring_init_lengths()))
        writer.writerow(['rmse', 'initial guess', 'final install positions'])

        plt.ion()
        fig = plt.figure(figsize=[9.6, 6.4])
        # print(plt.gcf().number)
        # fig2 = plt.figure(2)
        # fig2, axs = plt.subplots(4,5)
        # fig2.number = 2
        # print(plt.gcf().number)
        for sample in range(100): #try 100 different initial states
            init_guess = np.random.randint(0, high=dplm_instance.get_slot_num()*2-1, size=dplm_instance.get_spring_num())-dplm_instance.get_slot_num()+1
            guess = np.array(init_guess, copy=True)
            for greedy_iter_num in range(1): #three iterations for each greedy
                for ind in range(dplm_instance.get_spring_num()):
                    rmse = np.zeros(dplm_instance.get_slot_num())
                    for slot in range(-dplm_instance.get_slot_num()+1, dplm_instance.get_slot_num()):
                        guess[ind] = slot
                        dplm_instance.set_slot(guess)
                        rmse[slot] = dplm_instance.current_rmse()
                        guess[ind] = np.argmin(rmse)
                dplm_instance.set_slot(guess)

            if plotting == True:
                plt.figure(1)
                print(plt.gcf().number)
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


                ax.text(-20,-13, 'RMSE={:.2f} \nInitial random state: {} \nFinal install positions: {}'.format(rmse,init_guess, guess))
                plt.savefig('greedy_graphs/4_springs_t/test_{}.png'.format(sample+1))
                # plt.show()
                plt.pause(0.001)
                # del fig
                
                writer.writerow([f'{rmse:.2f}', list(init_guess), list(guess)])

                print(rmse)

                # if (sample+1)%4==0:
                #     plt.figure(2)
                #     fig2.add_subplot(ax)

        # plt.figure(2)

    k = pd.read_csv('greedy_graphs/4_springs/greedy.csv', header = 4)

    k.index = np.arange(1, len(k)+1)
    k.index.name='sample'
    k.to_excel('greedy_graphs/4_springs/greedy.xlsx')
    k.to_csv('greedy_graphs/4_springs/greedy_processed.csv')
    k.to_clipboard(sep = ',')

    k