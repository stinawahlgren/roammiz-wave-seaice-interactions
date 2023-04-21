import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, BoundaryNorm, ListedColormap
from matplotlib.dates import DateFormatter, DayLocator
import cmocean.cm as cmo

import waves


def plot_panel(swift, axes, vmin, vmax, y_left_visible = True, 
               y_right_visible = True, wave_height_ylim = None,
               ):
    
    plot_sic_and_waveheight(swift, axes[0], ylim = wave_height_ylim, y_right_visible = y_right_visible)
    plot_ice_code(swift, axes[1])
    plot_wave_spectra(swift, axes[2], vmin, vmax)
    
    if not y_left_visible:
        axes[0].yaxis.set_visible(False)
        axes[1].yaxis.set_visible(False)
        axes[2].yaxis.set_visible(False)

    return


def plot_wave_spectra(swift, ax, vmin, vmax):    
    ds = waves.get_wavespectra(swift)    
    ax.pcolormesh(ds.time, ds.freq, ds.energy, 
                  norm = LogNorm(vmin = vmin, vmax = vmax),
                  shading = 'nearest')
    ax.set_ylabel('$f$ (Hz)')

    return


def plot_ice_code(swift, ax):
    (cmap,norm) = icetype_cmap()    
    ax.pcolor(swift.timestamp, [0,1], np.tile([swift.icetype],(2,1)), 
              cmap = cmap, norm = norm,
              shading = 'nearest')
    ax.yaxis.set_visible(False)
    return


def plot_sic_and_waveheight(swift, ax, waveheight_threshold = 0.1 , ylim = None, y_right_visible = True):
    
    kwargs = {'linewidth': 3}
    ycol = 'sigwaveheight'
    xcol = 'timestamp'
    
    # Plot wave height  
    is_valid = swift[ycol] >= waveheight_threshold
    ax.plot(swift[xcol], swift[ycol], c = 'gray')
    ax.plot(swift[xcol].where(is_valid, np.nan), 
            swift[ycol].where(is_valid, np.nan),
            **kwargs)      
    
    # Plot sea ice concentration
    ax_sic = ax.twinx()
    ax_sic.fill_between(swift[xcol], swift['sic'],
                        step = "mid", color = "silver")
    ax_sic.set_ylim([0,100])
    ax_sic.zorder = 1
    ax.zorder=2
    ax.patch.set_visible(False)
    
    ax.set_ylabel('$H_s$ (m)')
    if ylim is not None:
        ax.set_ylim(ylim)
    
    if y_right_visible:
        ax_sic.set_ylabel('sic (%)', color = 'gray')
        ax_sic.tick_params(axis = 'y', labelcolor = 'gray')
    else:
        ax_sic.yaxis.set_visible(False)
    
    return


def icetype_cmap(false_colors = False):
    """
    Custom cmap for the swift photo ice type scale
    
    -9 : No data (night)
    -8 : No data (frozen lens)
    -7 : No data (missing photos)
    -1 : Lead
    1-7: Sea ice type (1: open water, 7: tightly packed floes)
    
    Returns (cmap,norm)
    Example useage : plt.pcolormesh(icetype, cmap=cmap, norm=norm)
    """       
    color_no_data = [0.5, 0.5, 0.5] # grey
    color_lead    = [0.5, 0.0, 0.5] # purple     
    if false_colors:
        color_ice = matplotlib.cm.get_cmap('gist_rainbow')(np.linspace(0,1,7))[:,0:3]
    else:    
        color_ice = cmo.ice(np.linspace(0.05,0.90,7))[:,0:3]
    colors = np.concatenate([[color_no_data], [color_lead], color_ice]);   
    cmap = ListedColormap(colors)
    
    boundaries = [-10,-2,0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5]
    norm = BoundaryNorm(boundaries, cmap.N, clip=True)
    
    return (cmap, norm) 


def icetype_colorbar(im = None, false_colors=False, label="", labelpad=0, **kwargs):
    """
    Creates sea ice type colorbar, with appropriate ticks and tick labels.
    **kwargs are passed to plt.colorbar()
    """    
    (cmap,norm) = icetype_cmap(false_colors)

    ticks = [-6,-1,1,2,3,4,5,6,7]
    ticklabels = ['no data', '-1', '1', '2', '3', '4' ,'5', '6', '7']
    
    if im is None:
        im = plt.cm.ScalarMappable(cmap=cmap, 
                                   norm=norm)
    
    cbar = plt.colorbar(im, ticks=ticks, **kwargs)
       
    cbar.ax.set_yticklabels(ticklabels);  # vertically oriented colorbar
    cbar.set_label(label, labelpad=labelpad)
    
    return cbar


def subplot_time_grid(deployments, rows, cols, sharex="col", sharey=True, 
                      height_ratios = None, hspace=None, wspace=None, time="timestamp"):
    """
    Creates a layout of subplots with timestamps as x-axis, based on time spans. 
    Each column of subplots have the same x-limits. The width of columns are 
    based on the time span.
    
    Parameters:
        deployments    List of swift dataframes 
        rows           Number of rows of subplots
        cols           Number of columns of subplots
        
    Retruns a list of axes, one for each subplot.  
    """
    
    # get x limits
    (xmin, xmax) = get_common_time_limits(deployments, rows, cols, time = time)

    width_ratios = []
    for (tmin,tmax) in zip(xmin,xmax):      
        width_ratios.append((tmax-tmin).astype('timedelta64[m]').astype(int) / 60)
   
    # create grid and adjust width to time span
    fig, axes = plt.subplots(nrows=rows, ncols=cols, sharex=sharex, sharey=sharey,
                             gridspec_kw={'width_ratios': width_ratios,
                                          'height_ratios': height_ratios,
                                          'hspace': hspace,
                                          'wspace': wspace},
                             figsize=(20,10))    
    axes = axes.flatten()
    
    # create axes and adjust time axis
    for (i,df) in enumerate(deployments):
        col = i % cols;
        axes[i].set_xlim([xmin[col],xmax[col]])    
    fig.tight_layout()
    
    return (fig, axes)


def nice_time_axis(ax=None):
    if ax is None:
        ax = plt.gca()
    ax.xaxis.set_major_formatter(DateFormatter('%b %d'))
    ax.xaxis.set_major_locator(DayLocator())
    rotateTickLabels(ax)
    return


def rotateTickLabels(ax, rotation=30, which='x', rotation_mode='anchor', ha='right'):
    """ Rotates the ticklabels of a matplotlib Axes
    Parameters
    ----------
    ax : matplotlib Axes
        The Axes object that will be modified.
    rotation : float
        The amount of rotation, in degrees, to be applied to the labels.
    which : string
        The axis whose ticklabels will be rotated. Valid values are 'x',
        'y', or 'both'.
    rotation_mode : string, optional
        The rotation point for the ticklabels. Highly recommended to use
        the default value ('anchor').
    ha : string
        The horizontal alignment of the ticks. Again, recommended to use
        the default ('right').
    Returns
    -------
    None
    
    Source: https://github.com/matplotlib/matplotlib/issues/8509
    """

    if which == 'both':
        rotateTickLabels(ax, rotation, 'x', rotation_mode=rotation_mode, ha=ha)
        rotateTickLabels(ax, rotation, 'y', rotation_mode=rotation_mode, ha=ha)
    else:
        if which == 'x':
            axis = ax.xaxis

        elif which == 'y':
            axis = ax.yaxis

        for t in axis.get_ticklabels():
            t.set_horizontalalignment(ha)
            t.set_rotation(rotation)
            t.set_rotation_mode(rotation_mode)
    return


def get_common_time_limits(deployments, rows, cols, time="timestamp"):
    """
    Returns min and max timestamp for each column. Suitable for ajdusting
    xlim in a figure with rows x cols subplots, one for each deployment.
    
    Parameters:
        deployments    List of SWIFT_df or xarray datasets
        rows           Number of subplot rows
        cols           Number of subplot cols
        adjust_to_days If True, timespan will be extendend to midnights.
    Returns: Tuple of lists of timestamps (xmin_per_column, xmax_per_column)    
    """
    # get suitable x-axis for each column
    # Note: for simplicity, values are stored in [col][row], not the usual [row][col]!
    tmin = [[pd.Timestamp.max for c in range(rows)] for r in range(cols)]
    tmax = [[pd.Timestamp.min for c in range(rows)] for r in range(cols)]

    for (i,df) in enumerate(deployments):
        r = i // cols;
        c = i % cols;

        tmin[c][r] = min(df[time])
        tmax[c][r] = max(df[time])
  
    tmin_per_column = []    
    for col in tmin:
        tmin_per_column.append(min(col).to_datetime64()) 

    tmax_per_column = []
    for col in tmax:
        tmax_per_column.append(max(col).to_datetime64())
        
    return (tmin_per_column, tmax_per_column)    