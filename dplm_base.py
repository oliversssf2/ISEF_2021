import csv
import os
import math
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.arraysetops import in1d


def print_dir():
    print(os.getcwd())
    print("hi")


def RMSE():
    pass


class dplm:
    """
    A class used to represent a DPLM (Double parallelogram mechanism).
    This class stores the parameters and the current state of a DPLM instance.
    All the calculation of a DPLM instance is performed within this class

    Attributes:
    dplm_config -- dict -- a dictionary containing the configuration (name, length, and mass of linkages) of the DPLM instance.
    spring_positions -- list -- stores the current positions of all the springs installed on the dplm instance

    ------------
    """
    g = 9.80665  # constant of gravitational field

    def __init__(self, dplm_config_file):
        # the basic parameters (length, mass, etc.) of the DPLM instance
        self.dplm_config = {}
        self.slot_num = 0
        self.spring_constants = []
        self.spring_init_lengths = []
        self.spring_num = 0 
        self.triangle_list = []


        self.dplm_allowed_angle_range = \
            {
                'lower_limit': 0,
                'upper_limit': 0,
                'step_size': 0,
                'total_angle_num': 0
            }

        # The current spring installaion positions
        self.spring_positions = []

        self._import_parameter(dplm_config_file, self.dplm_config)

        #stores moment_weight, moment_string, moment_total to reduce computation
        #while learning
        #Structure:
        # {
        #     'moment_weight': list
        #     'moment_spring_dict':
        #       {
        #           (spring_constant, installation position, spring_initial_length): list
        #           .
        #           .
        #           .
        #       }
        #     'rmse':{
        #           (spring_constant, installation position, spring_initial_length): float 
        #       }
        #      
        # }
    
    def add_triangle(self, spring_constant,  init_length):
        self.spring_num +=2
        self.set_dplm_spring_lengths([*self.get_spring_init_lengths(), init_length, init_length])
        self.set_dplm_spring_constants([*self.get_spring_constatnts(), spring_constant, spring_constant])
        self.triangle_list.append([self.spring_num-2, self.spring_num-1])
        # print('triangle_list is: {}'.format(self.triangle_list))
        pass

    def rm_triangle(self):
        ind_1 = self.triangle_list[0][0]
        # ind_2 = self.triangle_list[0][1]
        self.triangle_list.clear()
        del self.spring_constants[ind_1]
        del self.spring_constants[ind_1]
        del self.spring_positions[ind_1]
        del self.spring_positions[ind_1]
        del self.spring_init_lengths[ind_1]
        del self.spring_init_lengths[ind_1]
        self.spring_num -=2

    def is_triangle(self, index):
        if ([index, index+1] in self.triangle_list) or ([index-1, index] in self.triangle_list):
            return True
        else:
            return False

    def get_another_triangle_index(self, index):
        if ([index, index+1] in self.triangle_list):
            return index+1
        elif([index-1, index] in self.triangle_list):
            return index-1
        else:
            raise ValueError("Wrong triangle index\n the triangle list is{}.\n \
                              but the input index is {}".format(self.triangle_list, index))
    def get_extension(self, index, l_p1p2, spring_init_length, phi):
        if index==-1:
            # print('returning the extension of a normal spring')
            val = l_p1p2 - spring_init_length
        elif self.is_triangle(index):
            another_index = self.get_another_triangle_index(index)
            installation_dif = abs(self.spring_positions[index]-self.spring_positions[another_index])
            if installation_dif >0: 
                l_another = math.sqrt(l_p1p2**2 + installation_dif**2 - 2*installation_dif*l_p1p2*math.cos(math.radians(phi))) 
                val = l_another + installation_dif + l_p1p2 - spring_init_length
            elif installation_dif <0:
                l_another = math.sqrt(l_p1p2**2 + installation_dif**2 + 2*installation_dif*l_p1p2*math.cos(math.radians(phi))) 
                val = l_another + installation_dif + l_p1p2 - spring_init_length
            elif installation_dif ==0:
                val = 2*l_p1p2 - spring_init_length 
            # print('returning the extension of a triangle')
        
        if val<0:
            return 0
        else:
            return val

    def show_dplm_config(self):
        for item in sorted(self.dplm_config.items()):
            print('{}:{}'.format(item[0], item[1]))

    def set_dplm_allowed_angle_range(self, lower_limit, upper_limit,  step_size):
        if ((upper_limit-lower_limit)/step_size).is_integer() == False:
            raise ValueError('The angle range and step value is not valid \n \
                The upper limit is {}. The lower limit is {}. The step size\
                    is {}'.format(upper_limit, lower_limit, step_size))
        else:
            self.dplm_allowed_angle_range = \
                {
                    'lower_limit': lower_limit,
                    'upper_limit': upper_limit,
                    'step_size': step_size,
                    'total_angle_num': int((upper_limit - lower_limit)/step_size+1)
                }

    def set_dplm_slot_num(self, n):
        self.slot_num = n
        print('The number of slots is set to {}'.format(n))

    def set_dplm_spring_num(self, n):
        if isinstance(n, int) == False:
            raise ValueError(
                "The number of spring n is {}, it should be an integer!!!".format(n))
        self.spring_num = n

    def set_dplm_spring_constants(self, spring_constants):
        if len(spring_constants) != self.spring_num:
            raise ValueError("The number of spring constants is incorrect\n \
                The number of spring is {} but the number of spring constants \
                    provided is {}".format(self.spring_num, len(spring_constants)))
        else:
            self.spring_constants = spring_constants

    def set_dplm_spring_lengths(self, spring_init_lengths):
        if len(spring_init_lengths) != self.spring_num:
            raise ValueError("The number of spring lengths is incorrect\n \
                The number of spring is {} but the number of spring lengths\
                    provided is {}".format(self.spring_num, len(spring_init_lengths)))
        else:
            self.spring_init_lengths = spring_init_lengths 

    def set_springs_positions(self, spring_positions):
        if len(spring_positions) != self.spring_num:
            raise ValueError("The number of spring positions is incorrect\n \
                The number of spring is {} but the number of spring positions\
                    provided is {}".format(self.spring_num, len(spring_positions)))
        else:
            self.spring_positions = spring_positions

    def _import_parameter(self, file, dest_dict):
        """Import the parameters of the DPLM intance from a csv file and write the paramters into a dictionary

        Args:
            file ([str]): The filename of the csv file containing the parameters of the DPLM:
            The .csv file should contain the name, lenght and mass of each linkages
            in the DPLM. Each line should contain the parameters of one linkage, 
            delimited by ",", in the following format:

            name,length,mass

            Example: 
                name,length,mass
                O1O_1,0.762,0.84734661
                O1O2,0.254,0.29870661
                O3O_3,0.648,0.72422661
                O2O4,0.615,0.67904403

            dest_dict ([dict]): The destination dictionary that stores the parameters
            of the DPLM instance.
        """

        with open(file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                # print(row)
                if line_count == 0:
                    print(f'Column names are {", ".join(row)}')
                    line_count += 1
                else:
                    dest_dict['l_'+row[0].lower()] = float(row[1])
                    dest_dict['m_'+row[0].lower()] = float(row[2])
                    #dest_dict[row[0].lower()] = [row[1],row[2]]
                    #print("name is{}".format(row[0]))
                    #print("length is {} and mass is {}".format(row[1], row[2]))
                    line_count += 1

            dest_dict['l_o1o3'] = dest_dict['l_o2o4'] - 2*dest_dict['l_o1o2']
            dest_dict['m_o2o_2'] = dest_dict['m_o1o_1']
            dest_dict['m_o4o_4'] = dest_dict['m_o3o_3']
            dest_dict['l_o2o_2'] = dest_dict['l_o1o_1']
            dest_dict['l_o_1o_2'] = dest_dict['l_o1o2']
            dest_dict['l_o4o_4'] = dest_dict['l_o3o_3']
            dest_dict['l_o3o4'] = dest_dict['l_o1o2']
            dest_dict['l_o_3o_4'] = dest_dict['l_o3o4']

            dest_dict['r_o1o_1'] = dest_dict['l_o1o_1']/2
            dest_dict['alpha_o1o_1'] = 0
            dest_dict['r_o2o_2'] = dest_dict['l_o2o_2']/2
            dest_dict['alpha_o2o_2'] = 0
            dest_dict['r_o3o_3'] = dest_dict['l_o3o_3']/2
            dest_dict['alpha_o3o_3'] = 0
            dest_dict['r_o4o_4'] = dest_dict['l_o4o_4']/2
            dest_dict['alpha_o4o_4'] = 0
            dest_dict['r_o_1o_2'] = dest_dict['l_o_1o_2']/2
            dest_dict['alpha_o_1o_2'] = 0
            dest_dict['r_o2o4'] = dest_dict['l_o2o4']/2
            dest_dict['alpha_o2o4'] = 0

    def _calculate_moment(self, inst_pos, spring_constant, spring_init_length, angle,
                          calculate_moment_weight=True, calculate_moment_spring=True,
                          index = -1):
        """Calculate the moment generated by a spring on the base hinge of the dplm instance.
            return moment_i, moment_g, moment_total

        Args:
            inst_pos (float): The installation position of a spring in meter, currently using the value
            of o1p1 - o2p2.

            spring_constant (float): Spring constant in N/m.
            spring_init_len (float): the initial (unextended) length of spring
            angle (float): The current angle of the dplm instance in degree. 
        """

        # calculate the coordinates all the points on the dplm
        o_4x = 0.0
        o_4y = 0.0
        o3x = -self.dplm_config['l_o3o_3']*math.cos(math.radians(angle))
        o3y = self.dplm_config['l_o_3o_4'] + \
            self.dplm_config['l_o3o_3']*math.sin(math.radians(angle))
        o_3x = 0.0
        o_3y = self.dplm_config['l_o_3o_4']
        o4x = -self.dplm_config['l_o4o_4']*math.cos(math.radians(angle))
        o4y = self.dplm_config['l_o4o_4']*math.sin(math.radians(angle))
        o1x = o3x
        o1y = o3y + self.dplm_config['l_o1o3']
        o2x = o1x
        o2y = o1y + self.dplm_config['l_o1o2']
        o_1x = o1x + self.dplm_config['l_o1o_1']*math.cos(math.radians(angle))
        o_1y = o1y + self.dplm_config['l_o1o_1']*math.sin(math.radians(angle))
        o_2x = o_1x
        o_2y = o_1y+self.dplm_config['l_o1o2']

        # calculate the coordinate of the COMs of the linkages
        # the prefix m_ stands for COM
        m_o1o_1y = o1y + self.dplm_config['r_o1o_1'] * math.sin(
            math.radians(self.dplm_config['alpha_o1o_1'] + angle))
        m_o1o_1x = o1x + self.dplm_config['r_o1o_1'] * math.cos(
            math.radians(self.dplm_config['alpha_o1o_1'] + angle))

        m_o2o_2x = o2x + self.dplm_config['r_o2o_2'] * math.cos(
            math.radians(self.dplm_config['alpha_o2o_2'] + angle))
        m_o2o_2y = o2y + self.dplm_config['r_o2o_2'] * math.sin(
            math.radians(self.dplm_config['alpha_o2o_2'] + angle))

        m_o3o_3x = o_3x - self.dplm_config['r_o3o_3'] * math.cos(
            math.radians(self.dplm_config['alpha_o3o_3'] + angle))
        m_o3o_3y = o_3y + self.dplm_config['r_o3o_3'] * math.sin(
            math.radians(self.dplm_config['alpha_o3o_3'] + angle))

        m_o4o_4x = o_4x - self.dplm_config['r_o4o_4'] * math.cos(
            math.radians(self.dplm_config['alpha_o4o_4'] + angle))
        m_o4o_4y = o_4y + self.dplm_config['r_o4o_4'] * math.sin(
            math.radians(self.dplm_config['alpha_o4o_4'] + angle))

        m_o_1o_2x = o_2x + self.dplm_config['r_o_1o_2'] * \
            math.sin(math.radians(self.dplm_config['alpha_o_1o_2']))
        m_o_1o_2y = o_2y - self.dplm_config['r_o_1o_2'] * \
            math.cos(math.radians(self.dplm_config['alpha_o_1o_2']))

        m_o2o4x = o2x - self.dplm_config['r_o2o4'] * \
            math.sin(math.radians(self.dplm_config['alpha_o2o4']))
        m_o2o4y = o2y - self.dplm_config['r_o2o4'] * \
            math.cos(math.radians(self.dplm_config['alpha_o2o4']))

        if inst_pos >= 0:
            p1x = o_1x
            p1y = o_1y
            p2x = o2x + (self.dplm_config['l_o2o_2'] -
                         inst_pos)*math.cos(math.radians(angle))
            p2y = o2y + (self.dplm_config['l_o2o_2'] -
                         inst_pos)*math.sin(math.radians(angle))
        else:
            p1x = o1x
            p1y = o1y
            p2x = o2x + ((-inst_pos)*math.cos(math.radians(angle)))
            p2y = o2y + ((-inst_pos))*math.sin(math.radians(angle))
        # print('p1x = {:5f}, p1y = {:5f}, p2x = {:5f}, p2y = {:5f} when theta = {}'.format(p1x, p1y, p2x,p2y, angle))

        v_p2p1 = [p1x - p2x, p1y - p2y]
        v_o1o_1 = [o_1x-o1x, o_1y-o1y]
        l_p1p2 = np.linalg.norm(v_p2p1)
        # print('v_p2p1 = {}, v_o1o_1 = {}, l_p1p2 = {} when theta = {}'.format(v_p2p1,v_o1o_1,l_p1p2,angle))

        phi = math.degrees(math.acos(np.dot(v_p2p1, v_o1o_1) /
                           (l_p1p2*self.dplm_config['l_o1o_1'])))
        # print('phi = {} when theta = {}'.format(phi,angle))

        if calculate_moment_weight == True:
            x_o1o_1 = m_o1o_1x - o1x
            x_o2o_2 = m_o2o_2x - o2x
            x_o3o_3 = o_3x - m_o3o_3x
            x_o4o_4 = o_4x - m_o4o_4x
            x_o2o4 = o2x - m_o2o4x
            x_o_1o_2 = m_o_1o_2x - o_2x

            M_g = self.g * (self.dplm_config['m_o2o_2']*x_o2o_2 + self.dplm_config['m_o1o_1'] * x_o1o_1
                            + self.dplm_config['m_o_1o_2']*self.dplm_config['l_o1o_1'] * math.cos(math.radians(angle))
                            + self.dplm_config['m_o3o_3'] * x_o3o_3 + self.dplm_config['m_o4o_4'] * x_o4o_4 +
                            + (self.dplm_config['m_o_1o_2'] + self.dplm_config['m_o2o_2'] + self.dplm_config['m_o1o_1']
                               + self.dplm_config['m_o2o4']) * self.dplm_config['l_o3o_3'] * math.cos(math.radians(angle)))
        else:
            M_g = 0

        if calculate_moment_spring == True:
            extended_length  = self.get_extension(index, l_p1p2, spring_init_length, phi)

            # extended_length = self.get_extension(index, l_p1p2, spring_init_length)
            # extended_length = l_p1p2-spring_init_length
            # if extended_length < 0:
            #     extended_length = 0
            M_i = extended_length * \
                math.sin(math.radians(phi))*inst_pos*spring_constant
        else:
            M_i = 0

        # print('M_i = {:5f}, M_g = {:5f} when theta = {}'.format(M_i, M_g, angle))

        # M_total = M_i - M_g

        return M_i, M_g
  
    def calculate_current_moment(self):
        """Return the lists of the moment of spring, moment of weight, and the total 
        moment across the allowed angle range in the current spring installation.
        The number of spring, the position of spring, and the allowed angle range must
        be set to run this function

        Returns:
            moment_weight: list
            moment_spring_list: a list containing multiple lists corresponding to the
                                moments produced by all the springs on in DPLM
            moment_total: a list: the sum of the moment_spring in moment_spring_list
                          minus moment_weight
        """
        
        angle_range = self.dplm_allowed_angle_range
        moment_spring_list = []

        moment_weight = [self._calculate_moment(0,0,0,i+angle_range['lower_limit'], True, False)[1]\
        for i in range(angle_range['total_angle_num'])]

        
        for index in range(self.spring_num):
            if not self.is_triangle(index):
                moment_spring  = [self._calculate_moment(
                    self.spring_positions[index],
                    self.spring_constants[index],
                    self.spring_init_lengths[index],
                    angle+angle_range['lower_limit'], False, True)[0] 
                    for angle in range(angle_range['total_angle_num'])]
            elif self.is_triangle(index):
                 moment_spring  = [self._calculate_moment(
                    self.spring_positions[index],
                    self.spring_constants[index],
                    self.spring_init_lengths[index],
                    angle+angle_range['lower_limit'], False, True,index)[0] 
                    for angle in range(angle_range['total_angle_num'])]
            moment_spring_list.append(moment_spring)
        
        moment_total = [sum(x) for x in zip(*moment_spring_list, [-y for y in moment_weight])]
        
        temp = 0
        for x in moment_total:
            temp+=x**2
        rmse = math.sqrt(temp/self.dplm_allowed_angle_range['total_angle_num'])
        # for i in moment_spring_list:
            # print('****************************************')
            # for k in i:
                # print(k)

        #The sum of the moments of all springs minus the moment of weight        
        return moment_weight, moment_spring_list, moment_total, rmse

    def get_allowed_angle_range(self):
        """return the dictionary containing the allowed angle range

        Returns:
           dict: example:
            allowed_angle_range = {
                'lower_limit': 0,
                'upper_limit': 0,
                'step_size': 0,
                'total_angle_num': 0
            }
        """
        return self.dplm_allowed_angle_range

    def current_rmse(self ):
        rmse = self.calculate_current_moment()[3]
        return rmse
    def get_spring_positions(self):
        return self.spring_positions
    def get_slot_num(self):
        return self.slot_num
    def get_spring_init_lengths(self):
        return self.spring_init_lengths
    def get_spring_num(self):
        return self.spring_num
    def get_spring_constatnts(self):
        return self.spring_constants
    def set_extremity_load(self, load_mass_kg):
        self.dplm_config['m_o_1o_2'] = load_mass_kg
        pass
    def set_slot(self, slots):
        """Change the installation slots of the springs on the dplm as specified
           by the incoming list [slots]. The lenght [slots] must be equal to the 
           number of slots on the dplm or exception would be raised. The 
           calculate_current_moment function is called automatically once new slots
           are set

        Args:
            slots (list): a list containing the slots the springs should be installed on
        """
        if len(slots) != self.spring_num:
            raise ValueError("The new slots specified do not match with the \
                number of spring on the dplm instatnce. There are {} springs on \
                    this dplm instance but the incoming slots list contains \
                        {} items".format(self.spring_num, len(slots)))
        if abs(max(slots))>(self.slot_num-1):
            raise ValueError("The incoming slots exceed the nubmer of slots\
                              on this dplm instance, the incoming list is {}\
                              and the number of slots on this dplm is {}".format(
                                  slots, self.slot_num
                              ))    
        linkage_length = self.dplm_config["l_o1o_1"]
        slot_num = self.slot_num

        #minus one because there are only slot_num-1 intervals
        self.set_springs_positions([(linkage_length/(slot_num-1))*x for x in slots])
        # print('Set_slot: spring positions are {}'.format(self.spring_positions))
        #self.calculate_current_moment()
        

# Testing code
if __name__ == "__main__":
    cwd = os.getcwd()
    dplm_1 = dplm(cwd + "/para1.csv")
    dplm_1.set_dplm_slot_num(5)
    dplm_1.set_dplm_spring_num(3)
    dplm_1.set_springs_positions([0.2,0.4,0.1])
    dplm_1.set_dplm_spring_constants([200,350,210])
    dplm_1.set_dplm_spring_lengths([0.2, 0.1, 0.4])
    dplm_1.set_dplm_allowed_angle_range(-20, 60, 1)
    dplm_1.calculate_current_moment()
    # moment_spring, moment_weight, moment_total = dplm_1.calculate_current_moment()
    # print(moment_spring)
    # print(moment_weight)
    # print(moment_total)

    # k = [dplm_1.calculate_moment(0.2,0.3,0.2, item)[1] for item in range(-50, 51)]
    # print(k)
    # dplm_1.calculate_moment(0.25)
    # plt.plot(range(-50,51), k)
    # plt.show()
