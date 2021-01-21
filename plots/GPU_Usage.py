"""
Author: Georgios Voulgaris
Date: 07_12_2020
Description: Plots GPU power consumption, time running and architecture on average of 5 runs.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt

# agg backend is used to create plot as a .png file
mpl.use('agg')

pwr = [68.11, 71.394, 75.14,
       108.03, 140.51, 149.96,
       134.68, 71.12,
       138.57, 113.47,
       69.31, 108.63, 121.01]
time = [1.704, 6.532, 43.37,
        1.148, 5.05, 21.48,
        4.97, 37.82,
        15.83, 47.67,
        19.24, 42.44, 145.92]
arch = ['3_CNN_50', '3_CNN_100', '3_CNN_226',
        '4_CNN_50', '4_CNN_100', '4_CNN_226',
        '5_CNN_100', '5_CNN_226',
        'AlexNet_50', 'AlexNet_100',
        'VGG16_50', 'VGG16_100', 'VGG16_226']
# labels = ['3_CNN', '4_CNN', '5_CNN', 'AlexNet', 'VGG_16']

fig, ax = plt.subplots(figsize=(10, 10))
ax.scatter(time, pwr, c='r', marker='x', s=35)

for i, txt in enumerate(arch):
    ax.annotate(txt, (time[i], pwr[i]), xytext=(5, 5), textcoords='offset points')
    # plt.scatter(time, pwr, marker='x', color='red')

# Custom axis labels
ax.set_axisbelow(True)
# ax.legend(labels, loc='upper right', shadow=True)
ax.set_title('GPU Power Consumption', fontsize=18)
ax.set_xlabel('Time (minutes)', fontsize=14)
ax.set_ylabel('Power (watts)', fontsize=14)

# Add a horizontal grid to the plot, but make it very light in colour
# thus, assist when reading data values but not too distracting
ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
ax.xaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)

# Save the figure
fig.savefig('GPU_Usage.png', bbox_inches='tight')
