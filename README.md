# Deep_Architectures

Description: Test platform to validate various Deep architectures using SSRP dataset. The dataset is comprised of aerial images from Ghaziabad India. Stratification method was used to split the data to train/validate: 80% (out of which train: 80% and validation: 20%), and test: 20% data.

Architectures used: 3, 4, and 5 CNN, AlexNet, VGG_16.

Tested images: compressed 50x50, 100x100, and 226x 226 pixel images. Note that 50x50 was too small for the 5 CNNs.

Test Procedure: 5 runs for each architecture for each of the compressed data. That is 5x50x50 for each architecture. Then the Interquartile range, using the median was plotted.

Plots: Average GPU usage per architecture, Interquartile, and for each architecture an F1 Score heatmap for each class.
