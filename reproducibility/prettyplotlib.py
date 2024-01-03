# Modified from https://github.com/samfway/samplotlib/blob/master/samplotlib/util.py

import numpy as np
from matplotlib import rcParams

# Constants
SINGLE_FIG_SIZE = (6,4)
BAR_WIDTH = 0.6
TICK_SIZE = 15
XLABEL_PAD = 10
LABEL_SIZE = 14
TITLE_SIZE = 16
LEGEND_SIZE = 12
LINE_WIDTH = 2
LIGHT_COLOR = '0.8'
LIGHT_COLOR_V = np.array([float(LIGHT_COLOR) for i in range(3)])
DARK_COLOR = '0.4'
DARK_COLOR_V = np.array([float(DARK_COLOR) for i in range(3)])
ALMOST_BLACK = '0.125'
ALMOST_BLACK_V = np.array([float(ALMOST_BLACK) for i in range(3)])
ACCENT_COLOR_1 = np.array([255., 145., 48.]) / 255.

# Configuration
#rcParams['text.usetex'] = True #Let TeX do the typsetting
rcParams['pdf.use14corefonts'] = True
rcParams['ps.useafm'] = True
#rcParams['text.latex.preamble'] = [r'\usepackage{sansmath}', r'\sansmath'] #Force sans-serif math mode (for axes labels)
rcParams['font.family'] = 'sans-serif' # ... for regular text
rcParams['font.sans-serif'] = ['Helvetica Neue', 'HelveticaNeue', 'Helvetica'] #, Avant Garde, Computer Modern Sans serif' # Choose a nice font here
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42
rcParams['text.color'] = ALMOST_BLACK
rcParams['axes.unicode_minus'] = False

rcParams['xtick.major.pad'] = '8'
rcParams['axes.edgecolor']  = ALMOST_BLACK
rcParams['axes.labelcolor'] = ALMOST_BLACK
rcParams['lines.color']     = ALMOST_BLACK
rcParams['xtick.color']     = ALMOST_BLACK
rcParams['ytick.color']     = ALMOST_BLACK
rcParams['text.color']      = ALMOST_BLACK
rcParams['lines.solid_capstyle'] = 'butt'

# Imports
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D


def single_fig(figsize=SINGLE_FIG_SIZE):
    return plt.subplots(1,1,figsize=figsize)


def color_bp(bp, color):
    """ Helper function for making prettier boxplots """
    c = np.array(color) # * 0.5
    c = tuple(c)

    for x in bp['boxes']:
        plt.setp(x, color=c)
        x.set_facecolor(color)
    for x in bp['medians']:
        plt.setp(x, color='w')
    for x in bp['whiskers']:
        plt.setp(x, color=c)
    for x in bp['fliers']:
        plt.setp(x, color=c)
    for x in bp['caps']:
        plt.setp(x, color=c)


def adjust_spines(ax, spines):
    """ From http://matplotlib.org/examples/pylab_examples/spine_placement_demo.html """
    for loc, spine in ax.spines.items():
        if loc in spines:
            spine.set_position(('outward', 10))  # outward by 10 points
            spine.set_smart_bounds(True)
        else:
            spine.set_color('none')  # don't draw spine

    # turn off ticks where there is no spine
    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')
    else:
        # no yaxis ticks
        ax.yaxis.set_ticks([])

    if 'bottom' in spines:
        ax.xaxis.set_ticks_position('bottom')
    else:
        # no xaxis ticks
        ax.xaxis.set_ticks([])

def share_axes(axs,xticks=None,yticks=None):
    """ Remove tick labels, adjust axis limits, and label axes for shared axes in axs
        assumes axs is in the form of [[a1,a2,a3],[b1,b2,b3],[c1,c2,c3]], etc where left and bottom axes should be labeled """
    # remove x ticks for all except last row
    for ax_list in axs[:-1]:
        for ax in ax_list:
            ax.xaxis.set_ticks([])
    # remove y ticks for all except left column
    for ax_list in axs:
        for ax in ax_list[1:]:
            ax.yaxis.set_ticks([])
    # add x and y ticks
    if xticks != None:
        for ax in axs[-1]:
            ax.xaxis.set_ticks(xticks)
    if yticks != None:
        for ax_list in axs:
            ax = ax_list[0]
            ax.yaxis.set_ticks(yticks)


def hide_right_top_axis(ax):
    """ Remove the top and right axis """
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

def no_frame(ax):
    """ Remove all axes """
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

def no_ticks(ax):
    """ Remove ticks from both axes """
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])

def remove_ax(ax):
    """ Remove frame and ticks """
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])

def finalize_keep_frame(ax, fontsize=LABEL_SIZE, labelpad=7, aspect=1):
    """ Apply final adjustments """
    ax.tick_params(direction='out')
    ax.yaxis.label.set_size(fontsize)
    ax.xaxis.label.set_size(fontsize)
    ax.tick_params(axis='both', which='major', labelsize=fontsize, pad=labelpad)
    # force_aspect(ax,aspect)

def finalize(ax, fontsize=LABEL_SIZE, labelpad=7, aspect=1):
    """ Apply final adjustments """
    ax.tick_params(direction='out')
    hide_right_top_axis(ax)
    ax.yaxis.label.set_size(fontsize)
    ax.xaxis.label.set_size(fontsize)
    ax.tick_params(axis='both', which='major', labelsize=fontsize, pad=labelpad)
    # force_aspect(ax,aspect)

def force_aspect(ax,aspect=1):
    # set aspect ratio - default is square
    # ys = ax.get_ylim(); xs = ax.get_xlim()
    # asp =  (abs((xs[1]-xs[0])/(ys[1]-ys[0]))/aspect)
    # ax.set_aspect(asp)
    im = ax.get_images()
    extent =  im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)

def calculate_aspect(ax,aspect=1):
    # set aspect ratio - default is square
    ys = ax.get_ylim(); xs = ax.get_xlim()
    return (abs((xs[1]-xs[0])/(ys[1]-ys[0]))/aspect)

def lineswap_axis(fig, ax, zorder=-1000, lw=1, alpha=0.2, skip_zero=False):
    """ Replace y-axis ticks with horizontal lines running through the background.
        Sometimes this looks really cool. Worth having in the bag 'o tricks.
    """
    fig.canvas.draw()  # Populate the tick vals/labels. Required for get_[xy]ticklabel calls.

    ylabels = [str(t.get_text()) for t in ax.get_yticklabels()]
    yticks = [t for t in ax.get_yticks()]
    xlabels = [str(t.get_text()) for t in ax.get_xticklabels()]
    xticks = [t for t in ax.get_xticks()]

    x_draw = [tick for label, tick in zip(ylabels, yticks) if label != '']  # Which ones are real, though?
    y_draw = [tick for label, tick in zip(ylabels, yticks) if label != '']

    xmin = x_draw[0]
    xmax = x_draw[-1]

    # Draw all the lines
    for val in y_draw:
        if val == 0 and skip_zero:
            continue  # Don't draw over the bottom axis
        ax.plot([xmin, xmax], [val, val], color=ALMOST_BLACK, zorder=zorder, lw=lw, alpha=alpha)

    ax.spines["left"].set_visible(False)  # Remove the spine
    ax.tick_params(axis=u'y', which=u'both',length=0)  # Erase ticks by setting length=0
    ax.set_xlim(xmin, xmax)  # Retain original xlims

def create_extent(data,centers):
    '''
    Create extent for heatmap (imshow) with centered tickmarks
    data: dataframe for imshow
    centers: [min(x),max(x),min(y),max(y)]
    '''
    dx, = np.diff(centers[:2])/(data.shape[1]-1)
    dy, = -np.diff(centers[2:])/(data.shape[0]-1)
    extent = [centers[0]-dx/2, centers[1]+dx/2, centers[2]+dy/2, centers[3]-dy/2]
    return extent

def label_subplots(
    axes,
    x_pads,
    y_pad=1.15,
    labels=[],
    horizontal_alignments=[],
    fontsize=10,
    fontweight="bold",
    # fontname=font.name['subplot_label'],
):
    import string

    if not horizontal_alignments:
        horizontal_alignments = ['left' for i in range(len(axes))]

    if not labels:
        labels = list(string.ascii_uppercase)[:len(axes)]

    for ax, label, x_pad, ha in zip(axes, labels, x_pads, horizontal_alignments):
        axes[ax].text(
            x_pad,
            y_pad,
            label.upper(),
            transform=axes[ax].transAxes,
            # fontname=fontname,
            fontsize=fontsize,
            fontweight=fontweight,
            va='top',
            ha=ha,
        )
