# Plots the heat map of F1 Score per Deep Architecture for each class

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

classes = ["Agriculture", "Barren_Land", "Brick_Kilns", "Forest_Orchard", "Industry", "Roads", "Urban",
          "Urban_Green_Space", "Water"]
architectures = ['3_CNN_50', '3_CNN_100', '3_CNN_226', '4_CNN_50', '4_CNN_100', '4_CNN_226', '5_CNN_100', '5_CNN_226',
                'AlexNet_50', 'AlexNet_100', 'VGG16_50', 'VGG16_100', 'VGG16_226']
# f1_score = []


f1_score = np.array([[0.91, 0.94, 0.93, 0.95, 0.96, 0.95, 0.95, 0.96, 0.92, 0.85, 0.94, 0.92, 0.81],
                     [0.65, 0.67, 0.64, 0.76, 0.77, 0.73, 0.76, 0.76, 0.65, 0.59, 0.65, 0.56, 0.50],
                     [0.73, 0.79, 0.76, 0.83, 0.85, 0.83, 0.85, 0.85, 0.75, 0.76, 0.80, 0.63, 0.58],
                     [0.90, 0.93, 0.92, 0.92, 0.95, 0.94, 0.95, 0.95, 0.91, 0.85, 0.92, 0.92, 0.82],
                     [0.64, 0.66, 0.62, 0.75, 0.75, 0.73, 0.77, 0.77, 0.64, 0.57, 0.61, 0.58, 0.44],
                     [0.59, 0.63, 0.48, 0.68, 0.71, 0.64, 0.72, 0.71, 0.53, 0.56, 0.51, 0.43, 0.41],
                     [0.77, 0.80, 0.76, 0.84, 0.85, 0.83, 0.86, 0.84, 0.77, 0.70, 0.80, 0.70, 0.61],
                     [0.59, 0.67, 0.57, 0.72, 0.74, 0.72, 0.77, 0.74, 0.62, 0.67, 0.61, 0.49, 0.54],
                     [0.96, 0.97, 0.98, 0.96, 0.97, 0.97, 0.96, 0.94, 0.94, 0.96, 0.99, 0.87, 0.94]])


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on bottom.
    ax.tick_params(top=False, bottom=True,
                   labeltop=False, labelbottom=True)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)+.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)+.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


fig, ax = plt.subplots(1, figsize=(11, 6))

im, cbar = heatmap(f1_score, classes, architectures, ax=ax,
                   cmap="PuOr", cbarlabel="F1 Score")
texts = annotate_heatmap(im, valfmt="{x:.2f}", size=9)

fig.tight_layout()

# Save the figure
fig.savefig('F1_Score_Heatmap.png', bbox_inches='tight')
