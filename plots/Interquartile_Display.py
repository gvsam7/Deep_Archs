"""
Author: Georgios Voulgaris
Date: 29_11_2020
Description: Plots the interquartile/median test accuracy for each architecture, using 3 datasets (50x50, 100x100 and
226x226 pixels).
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import pandas
from pandas import DataFrame

# agg backend is used to create plot as a .png file
mpl.use('agg')

CNN_3_test_acc_50 = [0.814085574, 0.829231352, 0.781143506, 0.801211662, 0.816736085]
CNN_3_test_acc_100 = [0.840590685, 0.828095418, 0.842105263, 0.837182885, 0.850435441]
CNN_3_test_acc_226 = [0.807648618, 0.831881863, 0.829988641, 0.807648618, 0.78227944]
CNN_4_test_acc_50 = [0.863687997, 0.866717153, 0.860280197, 0.878076486, 0.876183264]
CNN_4_test_acc_100 = [0.892086331, 0.875804619, 0.868610375, 0.880348353, 0.887163953]
CNN_4_test_acc_226 = [0.866717153, 0.856493752, 0.863309353, 0.870882242, 0.868989019]
CNN_5_test_acc_100 = [0.881105642, 0.87921242, 0.887542598, 0.879969708, 0.881484286]
CNN_5_test_acc_226 = [0.887542598, 0.88262022, 0.883377509, 0.876561908, 0.883377509]
AlexNet_50 = [0.819386596, 0.804998107, 0.824687618, 0.792124195, 0.829988641]
AlexNet_100 = [0.7883377508519, 0.811813707, 0.796289284, 0.781143506, 0.795153351]
AlexNet_226 = [0.71942446, 0.745550928, 0.671336615]
VGG_50 = [0.817114729, 0.822415752, 0.80954184, 0.841726619, 0.84437713]
VGG_100 = [0.790609618, 0.770162817, 0.773191973, 0.772056039, 0.765997728]
VGG_226 = [0.705035971, 0.738356683, 0.726618705, 0.764861795, 0.726997349]
data_to_plot = [CNN_3_test_acc_50, CNN_3_test_acc_100, CNN_3_test_acc_226,
                CNN_4_test_acc_50, CNN_4_test_acc_100, CNN_4_test_acc_226,
                CNN_5_test_acc_100, CNN_5_test_acc_226,
                AlexNet_50, AlexNet_100, AlexNet_226,
                VGG_50, VGG_100, VGG_226]

# Create a figure instance
fig = plt.figure(1, figsize=(17, 12))

# Create an axes instance
ax = fig.add_subplot(111)

"""
# Create the boxplot
bp = ax.boxplot(data_to_plot)

# Save the figure
fig.savefig('fig1.png', bbox_inches='tight')"""

# Add patch_artist=True option to ax.boxplot() to get fill colour
bp = ax.boxplot(data_to_plot, patch_artist=True)

# Change outline colour, fill colour and linewidth of the boxes
for box in bp['boxes']:
     # Change outline colour
     box.set(color='#7570b3', linewidth=2)
     # Change fill colour
     box.set(facecolor='#1b9e77')

# Change colour and linewidth of the whiskers
for whisker in bp['whiskers']:
     whisker.set(color='#7570b3', linewidth=2)

# Change colour and linewidth of the caps
for cap in bp['caps']:
     cap.set(color='#7570b3', linewidth=2)

# Change colour and linewidth of the medians
for median in bp['medians']:
     median.set(color='#b2df8a', linewidth=2)

# Change the style of fliers and their fill
for flier in bp['fliers']:
     flier.set(marker='o', color='#e7298a', alpha=0.5)

# Custom x-axis labels
ax.set_xticklabels(['3_CNN_50', '3_CNN_100', '3_CNN_226',
                    '4_CNN_50', '4_CNN_100', '4_CNN_226',
                    '5_CNN_100', '5_CNN_226',
                    'AlexNet_50', 'AlexNet_100', 'AlexNet_226',
                    'VGG_50', 'VGG_100', 'VGG_226'])

# Custom axis labels
ax.set_axisbelow(True)
ax.set_title('Comparison of Test Accuracy for Various CNN Architectures', fontsize=18)
ax.set_xlabel('CNN Architectures', fontsize=14)
ax.set_ylabel('Test Accuracy', fontsize=14)

# Add a horizontal grid to the plot, but make it very light in colour
# thus, assist when reading data values but not too distracting
ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
ax.xaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)

# Remove top and right axes ticks
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()

# Save the figure
fig.savefig('Interquartile.png', bbox_inches='tight')



