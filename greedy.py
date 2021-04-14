
import dplm_base
import matplotlib.pyplot as plt

matplotlib.use('agg')

plotting = True

if __name__=='__main__':
    dplm_instance = dplm_base.dplm('para1.csv')
    dplm_instance.show_dplm_config()
    dplm_instance.set_dplm_slot_num(20)
    dplm_instance.set_dplm_spring_num(3)
    dplm_instance.set_slot([-4, 13, 8])
    dplm_instance.set_dplm_spring_constants([300,300,300])
    dplm_instance.set_dplm_spring_lengths([0.2, 0.2, 0.2])
    dplm_instance.set_dplm_allowed_angle_range(-40, 60, 1)

    if plotting == True:
        lower_limit, upper_limit, step_size, total_angle_num = dplm_instance.get_allowed_angle_range().values()

        a,b,c, rmse = dplm_instance.calculate_current_moment()
        # plt.cla()    
        # plt.figure()
        ax = plt.gca()
        plt.plot(range(lower_limit, upper_limit+1), a, label = 'moment_weight', ls = '--', lw = 3, color = 'mediumaquamarine')

        ax = plt.gca()

        for i in range(len(b)):
            plt.plot(range(lower_limit,upper_limit+1), b[i], label = 'moment_spring_{}'.format(i+1), ls = '--', lw = 3, color = 'cornflowerblue')

        plt.plot(range(lower_limit, upper_limit+1), c, label = 'moment_total', ls = '--', lw = 3, color = 'mediumslateblue')
        plt.axhline(y = 0, ls = '-', lw = 3, color = 'darkgrey')

        plt.axis(ymin=-20, ymax=50)
        plt.legend()
        plt.xlabel('angle [degree]')
        plt.ylabel('moment [Nm]')
        ax.xaxis.set_major_formatter('{x}°')


        plt.text(-10,-10, r'$RMSE={:.2f}$'.format(rmse))

        plt.savefig('test.png')
        # plt.show()

        print(rmse)