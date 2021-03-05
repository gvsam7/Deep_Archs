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

CNN_3_test_acc_50 = [0.859256449, 0.878224583, 0.871396055, 0.863429439, 0.857359636]
CNN_3_test_acc_100 = [0.883156297, 0.869499241, 0.861532625, 0.857738998, 0.87215478]
CNN_3_test_acc_256 = [0.820940819, 0.820940819, 0.811456753, 0.817526555, 0.842943854]
CNN_4_test_acc_50 = [0.847116844, 0.878603945, 0.797420334, 0.876707132, 0.876707132]
CNN_4_test_acc_100 = [0.872534143, 0.880121396, 0.892640364, 0.881638847, 0.88353566]
CNN_4_test_acc_256 = [0.836494689, 0.85660091, 0.851669196, 0.846358118, 0.867981791]
VAE_CNN_4_test_acc_50 = [0.820182094, 0.821699545, 0.806525038, 0.827010622, 0.806145675]
VAE_CNN_4_test_acc_100 = [0.85660091, 0.849772382, 0.863050076, 0.864946889, 0.847116844]
VAE_CNN_4_test_acc_256 = [0.819802731, 0.722306525, 0.832701062, 0.847496206, 0.814491654]
CNN_5_test_acc_100 = [0.880121396, 0.889605463, 0.85508346, 0.889605463, 0.849772382]
CNN_5_test_acc_256 = [0.886570561, 0.880880121, 0.857738998, 0.892261002, 0.887708649]
VAE_CNN_5_test_acc_100 = [0.846737481, 0.886570561, 0.853566009, 0.86646434, 0.839150228]
VAE_CNN_5_test_acc_256 = [0.874051593, 0.85660091, 0.862291351, 0.868740516, 0.852807284]
AlexNet_50 = [0.819802731, 0.75, 0.797420334, 0.823596358, 0.793626707]
AlexNet_100 = [0.789074355, 0.770106222, 0.812594841, 0.79400607, 0.81107739]
AlexNet_256 = [0.718133536, 0.704097117, 0.713960546, 0.720030349, 0.712443096]
VGG_50 = [0.836494689, 0.853566009, 0.79552352, 0.81676783, 0.845978756]
VGG_100 = [0.781866464, 0.775037936, 0.695371775, 0.753793627, 0.751138088]
VGG_256 = [0.752276176, 0.741654021, 0.745827011, 0.72723824, 0.737481032]
data_to_plot = [CNN_3_test_acc_50, CNN_3_test_acc_100, CNN_3_test_acc_256,
                CNN_4_test_acc_50, CNN_4_test_acc_100, CNN_4_test_acc_256,
                VAE_CNN_4_test_acc_50, VAE_CNN_4_test_acc_100, VAE_CNN_4_test_acc_256,
                CNN_5_test_acc_100, CNN_5_test_acc_256,
                VAE_CNN_5_test_acc_100, VAE_CNN_5_test_acc_256,
                AlexNet_50, AlexNet_100, AlexNet_256,
                VGG_50, VGG_100, VGG_256]

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
ax.set_xticklabels(['3_CNN_50', '3_CNN_100', '3_CNN_256',
                    '4_CNN_50', '4_CNN_100', '4_CNN_256',
                    'VAE_4_CNN_50', 'VAE_4_CNN_100', 'VAE_4_CNN_256',
                    '5_CNN_100', '5_CNN_256',
                    'VAE_5_CNN_100', 'VAE_5_CNN_256',
                    'AlexNet_50', 'AlexNet_100', 'AlexNet_256',
                    'VGG_50', 'VGG_100', 'VGG_256'], rotation=45)

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



