# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 14:20:51 2016

@author: Sara Madaan
"""
## this file plots the speed plots for the two heart beats separately, but
## keeps the time axis different for both plots. It also plots the average speed 
# for the two heart beats

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
## read the training and test data from the csv file
from func_import_csv import readcsv_as_nparray


file_location_train = 'C:/Users/Sara Madaan/Documents/E.DATA/1USC/Dr. Fraser/Projects/Light feild Microscopy/Paper 1/Heart Data/BPS Heart Data/New speed files after editing tracks/Speed_file_all_numbers.csv'
csv_array_dict_train = readcsv_as_nparray(file_location_train)
all_data = csv_array_dict_train['csv_nparray']
size_all_data = np.asarray(np.shape(all_data))

unique_tracks = np.unique(all_data[:,2])
num_tracks = np.asarray(np.shape(unique_tracks))

min_time = int(np.round(np.min(all_data[:,1])))
max_time = int(np.round(np.max(all_data[:,1])))

speed_arr = np.zeros((max_time-min_time+1,np.size(unique_tracks)+1))
size_speed_arr =  np.asarray(np.shape(speed_arr))

for i in range(min_time-1,max_time):
    print(i)
    speed_arr[i,0] = i+1

for i in range(size_all_data[0]):
    print(i)
    
    for j in range(num_tracks):
        if all_data[i,2] ==  unique_tracks[j]:
           
           for k in range(size_speed_arr[0]):
               
               if all_data[i,1] == speed_arr[k,0]:
               
                   speed_arr[k,j+1] = all_data[i,0]
                   
#%%necessary modifications
#for track 6, set all speed values before t = 42 = 0
speed_arr[0:46,6] = np.zeros(46)
speed_arr[84:155,6] = np.zeros(70)
# for track 7, set all speed values before t = 42 = 0 because they correspond to the blood cell entering the atrium at a high speed
speed_arr[0:46,7] = np.zeros(46) 
# for track 8, set all speed values before t = 42 = 0 because they correspond to the blood cell entering the atrium at a high speed
speed_arr[0:46,8] = np.zeros(46) 
speed_arr[79:155,8] = np.zeros(75) 
# we don't really need track 110,11 and 12, so set them all equal to 0
speed_arr[:,10] = np.zeros(size_speed_arr[0])    
speed_arr[:,11] = np.zeros(size_speed_arr[0])   
speed_arr[:,12] = np.zeros(size_speed_arr[0]) 



#%% multiply speeds by 0.09 (90 frames/sec divided by 1000 for unit conversion from um to mm)
# to get speed in mm/sec

speed_arr[:,1:] = speed_arr[:,1:]*0.09




# create new speed_arr_new with only the first 92 rows
speed_arr_new = speed_arr[0:86,:]  
size_speed_arr_new =  np.asarray(np.shape(speed_arr_new))

#%% plot the figures
label_size = 20
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 

fig, ax = plt.subplots()

ax.plot(speed_arr_new[:,0]/90,speed_arr_new[:,1],color='blue',lw = 6,label = 'Track 1')
#ax.plot(speed_arr[:,0],speed_arr[:,2],color='green',lw = 10,label = 'Track 2')  # we only have the atrial systole part of track 2, so we shouldn't use it 
ax.plot(speed_arr_new[:,0]/90,speed_arr_new[:,3],color='red',lw = 6,label = 'Track 3')
ax.plot(speed_arr_new[:,0]/90,speed_arr_new[:,4],color='cyan',lw = 6,label = 'Track 4')
ax.plot(speed_arr_new[:,0]/90,speed_arr_new[:,5],color='magenta',lw = 6,label = 'Track 5')
ax.plot(speed_arr_new[:,0]/90,speed_arr_new[:,6],color='darkblue',lw = 6,label = 'Track 6')
ax.plot(speed_arr_new[:,0]/90,speed_arr_new[:,7],color='black',lw = 6,label = 'Track 7')
ax.plot(speed_arr_new[:,0]/90,speed_arr_new[:,8],color='darkorange',lw = 6,label = 'Track 8')    
ax.plot(speed_arr_new[:,0]/90,speed_arr_new[:,9],color='darkgreen',lw = 6,label = 'Track 9')
ax.plot(speed_arr_new[:,0]/90,speed_arr_new[:,10],color='chocolate',lw = 6,label = 'Track 10')
ax.plot(speed_arr_new[:,0]/90,speed_arr_new[:,11],color='hotpink',lw = 6,label = 'Track 11')
ax.plot(speed_arr_new[:,0]/90,speed_arr_new[:,12],color='yellowgreen',lw = 6,label = 'Track 12')
ax.xaxis.set_ticks( np.around((np.arange(1,size_speed_arr_new[0]+15,5))/90,decimals=2))

legend = ax.legend(loc='upper right', shadow=True)

# The frame is matplotlib.patches.Rectangle instance surrounding the legend.
frame = legend.get_frame()
frame.set_facecolor('0.90')

# Set the fontsize
for label in legend.get_texts():
    label.set_fontsize('large')

for label in legend.get_lines():
    label.set_linewidth(6)  # the legend line width
plt.show()
   
#%% create new speed_arr2

speed_arr_2 = np.ones((42,size_speed_arr[1]*3))
size_speed_arr_2 =  np.asarray(np.shape(speed_arr_2))

for i in range(42):
    speed_arr_2[i,0] = i+1
    
speed_arr_2[:,1:size_speed_arr[1]] = speed_arr[0:42,1:size_speed_arr[1]]

speed_arr_2[:,size_speed_arr[1]:size_speed_arr[1]*2-1] = speed_arr[42:84,1:size_speed_arr[1]]

speed_arr_2[:,size_speed_arr[1]*2:size_speed_arr[1]*3-1] = speed_arr[84:126,1:size_speed_arr[1]]


plt.figure(2)
plt.hold

for i in range(size_speed_arr[1]*3-1):
    plt.plot(speed_arr_2[:,0],speed_arr_2[:,i+1])
    
plt.xticks( range(size_speed_arr_2[0]), speed_arr_2[:,0])
plt.xlabel('Time', fontsize=50)
plt.ylabel('Speed of blood cells', fontsize=50)


speed_arr_3 = speed_arr_2[:,1:]

speed_arr_4 = np.transpose(speed_arr_3)


speed_arr_5 =  speed_arr_4[~np.all(speed_arr_4 == 0, axis=1)]
size_speed_arr_5 =  np.asarray(np.shape(speed_arr_5))

xx = np.arange(1,size_speed_arr_5[0]+1)
yy = np.arange(1,size_speed_arr_5[1]+1)

plt.figure(3)
plt.contourf(yy, xx, speed_arr_5, cmap=plt.cm.Reds, alpha=0.8)
    

#%% split the speed_arrays into two separate
## arrays that contain one heartbeat each

# heartbeat 1 contains tracks 3,4 and 5

speed_arr_new_1 = speed_arr_new[:,[0,3,4,5]]


x_ticks =  np.around((np.arange(1,size_speed_arr_new[0]+5,5))/90,decimals =2)
# plot the figures
fig, ax = plt.subplots()

ax.plot(speed_arr_new_1[0:43,0]/90,speed_arr_new_1[0:43,1],color='blue',lw = 6,label = 'Track 3')
ax.plot(speed_arr_new_1[0:43,0]/90,speed_arr_new_1[0:43,2],color='red',lw = 6,label = 'Track 4')
ax.plot(speed_arr_new_1[0:43,0]/90,speed_arr_new_1[0:43,3],color='chocolate',lw = 6,label = 'Track 5')

ax.xaxis.set_ticks(x_ticks[0:8])

legend = ax.legend(loc='upper right', shadow=True)

# The frame is matplotlib.patches.Rectangle instance surrounding the legend.
frame = legend.get_frame()
frame.set_facecolor('0.90')

# Set the fontsize
for label in legend.get_texts():
    label.set_fontsize('large')

for label in legend.get_lines():
    label.set_linewidth(6)  # the legend line width
plt.show()



# heartbeat 2 contains tracks 6,7,8 and 9

speed_arr_new_2 = speed_arr_new[:,[0,6,7,8,9]]

#plot the figures
fig, ax = plt.subplots()

ax.plot(speed_arr_new_2[43:,0]/90,speed_arr_new_2[43:,1],color='black',lw = 6,label = 'Track 6')
ax.plot(speed_arr_new_2[43:,0]/90,speed_arr_new_2[43:,2],color='brown',lw = 6,label = 'Track 7')
ax.plot(speed_arr_new_2[43:,0]/90,speed_arr_new_2[43:,3],color='darkgreen',lw = 6,label = 'Track 8')
ax.plot(speed_arr_new_2[43:,0]/90,speed_arr_new_2[43:,4],color='hotpink',lw = 6,label = 'Track 9')

ax.xaxis.set_ticks(x_ticks[8:])

legend = ax.legend(loc='upper right', shadow=True)

# The frame is matplotlib.patches.Rectangle instance surrounding the legend.
frame = legend.get_frame()
frame.set_facecolor('0.90')

# Set the fontsize
for label in legend.get_texts():
    label.set_fontsize('large')

for label in legend.get_lines():
    label.set_linewidth(6)  # the legend line width
plt.show()


#%% calculate average speed plots and plot them


# for speed_arr_new_1
size_speed_arr_new_1 = np.asarray(np.shape(speed_arr_new_1))
ave_new_1 = np.zeros((size_speed_arr_new_1[0],2))
ave_new_1[:,0] = speed_arr_new_1[:,0]

for i in range(size_speed_arr_new_1[0]):
    if np.count_nonzero(speed_arr_new_1[i,1:]) > 1:
        ave_new_1[i,1] = np.sum(speed_arr_new_1[i,1:])/np.count_nonzero(speed_arr_new_1[i,1:])
            
#plot the figures
fig, ax = plt.subplots()

ax.plot(ave_new_1[0:43,0]/90,ave_new_1[0:43,1],color='black',lw = 6,label = 'Avg_heartbeat_1')
ax.xaxis.set_ticks(x_ticks[0:8])



# for speed_arr_new_2
size_speed_arr_new_2 = np.asarray(np.shape(speed_arr_new_2))
ave_new_2 = np.zeros((size_speed_arr_new_2[0],2))
ave_new_2[:,0] = speed_arr_new_2[:,0]

for i in range(size_speed_arr_new_2[0]):
    if np.count_nonzero(speed_arr_new_2[i,1:]) > 1:
        ave_new_2[i,1] = np.sum(speed_arr_new_2[i,1:])/np.count_nonzero(speed_arr_new_2[i,1:])
            
#plot the figures
fig, ax = plt.subplots()

ax.plot(ave_new_2[43:,0]/90,ave_new_2[43:,1],color='black',lw = 6,label = 'Avg_heartbeat_1')
ax.xaxis.set_ticks( x_ticks[8:])




