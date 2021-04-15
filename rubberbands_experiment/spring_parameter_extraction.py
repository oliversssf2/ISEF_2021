# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.1
#   kernelspec:
#     display_name: Python (isef2021_
#     language: python
#     name: isef_2021
# ---

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# +
column_names = ['time[s]', 'deflexion[mm]', 'force[kgf]']
raw_diamond = pd.read_excel("./tesile test raw data/raw_data_diamond_rubber_bands.xlsx", index_col=0)  
raw_diamond.attrs['name'] = 'Diamond'
raw_diamond_linear_range = [50,300]

raw_sannex = pd.read_excel("./tesile test raw data/raw_data_sannex_rubber_bands.xlsx", index_col=0)  
raw_sannex.attrs['name'] = 'Sannex'
raw_sannex_linear_range = [50,300]

# raw_orange_round = pd.read_excel("./tesile test raw data/raw_data_orange_round_rubber_band_100mm.xlsx", index_col=0)  
# raw_orange_round.attrs['name'] = 'orange_round_100mm'
# raw_orange_round_linear_range = [0,1000]

raw_orange_round = pd.read_excel("./tesile test raw data/raw_data_orange_round_rubber_band.xlsx", index_col=0)  
raw_orange_round.attrs['name'] = 'Orange round'
raw_orange_round_linear_range = [40,550]

raw_orange_rect = pd.read_excel("./tesile test raw data/raw_data_orange_rect_rubber_band.xlsx", index_col=0)  
raw_orange_rect.attrs['name'] = 'Orange rect'
raw_orange_rect_linear_range = [200,500]

raw_shortened_orange_round = pd.read_excel("./tesile test raw data/rwa_data_shortened_orange_round.xlsx", index_col=0)  
raw_shortened_orange_round.attrs['name'] = 'Shortened orange round'
raw_shortened_orange_round_linear_range = [90,400]
# -

# # Comparison of all four types of rubber bands

# +
# %gui osx
# %matplotlib
# # %matplotlib inline

raw_rubber_bands = []
linear_ranges = []
raw_rubber_bands.extend([raw_diamond, raw_sannex, raw_orange_round,raw_orange_rect])
linear_ranges.extend([raw_diamond_linear_range,raw_sannex_linear_range,raw_orange_round_linear_range,raw_orange_rect_linear_range])

for i in raw_rubber_bands:
    i.columns = column_names
#     print('this is rubber band {}'.format(i.attrs['name']))
#     print(i.describe()

show_linear_reg = True
show_linear_range = True

figure, axes = plt.subplots(2, 2,figsize=[9.6, 7.2], dpi=100)
figure.suptitle('Force against deflexion for four types of rubber bands', size = 15, weight = 'semibold')
figure.tight_layout(pad = 3)
plt.subplots_adjust(left=0.08, bottom=.08, right=.96, top=.9)

for index in range(len(raw_rubber_bands)):
    df = raw_rubber_bands[index]
    print('this is rubber band {}'.format(df.attrs['name']))
    k = df[df['deflexion[mm]'].between(*linear_ranges[index], inclusive=False)]
    print(k)
    
    x = df['deflexion[mm]'].values.reshape(-1,1)
    y = df['force[kgf]'].values.reshape(-1,1)
    
    linear_x = k['deflexion[mm]'].values.reshape(-1, 1)
    linear_y = k['force[kgf]'].values.reshape(-1, 1)
    
    axis = plt.subplot(2,2,index+1)
    axis.set_title("{} rubber band".format(df.attrs['name']), color = 'maroon', weight = 'semibold')
    axis.set_ylabel("Force[kgf]")
    axis.set_xlabel('Deflexion[mm]')
    
    axis.set_xlim(0,700)
    axis.set_ylim(0,13)
    
    if show_linear_reg:
        linear_regressor = LinearRegression()  # create object for the class
        linear_regressor.fit(linear_x, linear_y)  # perform linear regression
        Y_pred = linear_regressor.predict(linear_x)  # make predictions
        print('slope is {}, intercept is {}'.format(*linear_regressor.coef_, *linear_regressor.intercept_))
        spring_rate = linear_regressor.coef_*9.806*1000
        print('spring rate:\n{}'.format(float(spring_rate)))
        display_text = 'Spring rate:\n{:.2f}[N/m]\n\n Linear range:\n{:.0f}[mm] to {:.0f}[mm]: \n{:.0f}[mm]'\
                        .format(float(spring_rate), 
                                float(linear_x[0]), 
                                float(linear_x[-1]), 
                                float(linear_x[-1]-linear_x[0]))
        axis.plot(linear_x, Y_pred, color='red')
        axis.text(x=(linear_x[0]+linear_x[-1])/2, y=0.5*(sum(axis.get_ylim())),\
                  s=display_text,\
                  transform=axis.transData,\
                  horizontalalignment='center')
        axis.set_xticks(list(axis.get_xticks()))
#         axis.set_xticks([float(linear_x[0]),float(linear_x[-1])])
#         axis.set_xticks(list(axis.get_xticks())+[linear_x[0], linear_x[-1]])
#         axis.text(x=0.5, y=0.5,s='spring rate is {}'.format(spring_rate),transform=axis.transAxes)


        
    
    if show_linear_range:
        axis.axvline(linear_x[0], color = 'red')
        axis.axvline(linear_x[-1], color = 'red')
    
    
    
    axis.scatter(x, y, s =1)

plt.show()
# -

# # Comparison of the original and the shortened orange round rubber band

# +
# %gui osx
# %matplotlib
# # %matplotlib inline

raw_rubber_bands = []
linear_ranges = []
raw_rubber_bands.extend([raw_orange_round,raw_shortened_orange_round])
linear_ranges.extend([raw_orange_round_linear_range,raw_shortened_orange_round_linear_range])

for i in raw_rubber_bands:
    i.columns = column_names
#     print('this is rubber band {}'.format(i.attrs['name']))
#     print(i.describe()

show_linear_reg = True
show_linear_range = True

figure, axes = plt.subplots(1, 2,figsize=[9.6, 4.8], dpi=100)
figure.suptitle('Force against deflexion for the original and \nshortend orange round rubber band', size = 15, weight = 'semibold')
figure.tight_layout(pad = 3)
plt.subplots_adjust(left=0.08, bottom=.15, right=.96, top=.8)

for index in range(len(raw_rubber_bands)):
    df = raw_rubber_bands[index]
    print('this is rubber band {}'.format(df.attrs['name']))
    k = df[df['deflexion[mm]'].between(*linear_ranges[index], inclusive=False)]
    print(k)
    
    x = df['deflexion[mm]'].values.reshape(-1,1)
    y = df['force[kgf]'].values.reshape(-1,1)
    
    linear_x = k['deflexion[mm]'].values.reshape(-1, 1)
    linear_y = k['force[kgf]'].values.reshape(-1, 1)
    
    axis = plt.subplot(1,2,index+1)
    axis.set_title("{} rubber band".format(df.attrs['name']), color = 'maroon', weight = 'semibold')
    axis.set_ylabel("Force[kgf]")
    axis.set_xlabel('Deflexion[mm]')
    
    axis.set_xlim(0,700)
    axis.set_ylim(0,13)
    
    if show_linear_reg:
        linear_regressor = LinearRegression()  # create object for the class
        linear_regressor.fit(linear_x, linear_y)  # perform linear regression
        Y_pred = linear_regressor.predict(linear_x)  # make predictions
        print('slope is {}, intercept is {}'.format(*linear_regressor.coef_, *linear_regressor.intercept_))
        spring_rate = linear_regressor.coef_*9.806*1000
        print('spring rate:\n{}'.format(float(spring_rate)))
        display_text = 'Spring rate:\n{:.2f}[N/m]\n\n Linear range:\n{:.0f}[mm] to {:.0f}[mm]: \n{:.0f}[mm]'\
                        .format(float(spring_rate), 
                                float(linear_x[0]), 
                                float(linear_x[-1]), 
                                float(linear_x[-1]-linear_x[0]))
        axis.plot(linear_x, Y_pred, color='red')
        axis.text(x=(linear_x[0]+linear_x[-1])/2, y=0.5*(sum(axis.get_ylim())),\
                  s=display_text,\
                  transform=axis.transData,\
                  horizontalalignment='center')

    if show_linear_range:
        axis.axvline(linear_x[0], color = 'red')
        axis.axvline(linear_x[-1], color = 'red')
    
    
    
    axis.scatter(x, y, s =1)

plt.show()
# -


