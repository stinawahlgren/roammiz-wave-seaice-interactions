import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.ticker import AutoLocator


def distance_in_wave_direction(swift20, swift21):
    """
    Compute the projected distance in wave direction (in meters) bewteen the two buoys. 
    For each time instance, the wave direction is taken as the mean of the
    swell direction of the two buoys. Distance is computed in polar stereographic
    coordinates (x,y). The bouys are assumed to be at close to 0 deg longitude, thus
    no correction for that wave direction is given relative north and not relative 
    positive x-axis is performed.
    """
    if len(swift20) != len(swift21):
        raise RuntimeError('Number of data points do not match')

    if not all(swift20.timestamp.values == swift21.timestamp.values):
        raise RuntimeError('Timestamps do not match')    
    
    # Use the mean of the measured swell directions
    wave_direction = np.deg2rad((swift20['swell.meandirT']+swift21['swell.meandirT'])/2);
 
    # We don't correct for longitude. This is reasonable since we're approx at 0 longitude
    direction_vector_x = 1/np.sqrt(2) * np.sin(wave_direction)
    direction_vector_y = 1/np.sqrt(2) * np.cos(wave_direction)
    
    # Compute projected distance
    xdiff = swift20.x-swift21.x
    ydiff = swift20.y-swift21.y
    
    projected_distance_x = xdiff*direction_vector_x
    projected_distance_y = ydiff*direction_vector_y
    
    return np.sqrt(projected_distance_x**2 + projected_distance_y**2)


def get_pointwise_attenuation(spectra, spectra_ref, frequency_range, distance):
    """
    Computes the ratio of the power spectral density of two wave spectra  
    datasets (from waves.get_wavespectra) in the given frequency range. 
    This ratio, together with the distances provided, is then used to derive
    the attenuation coefficient for each frequency and timestamp, assuming
    exponential attenuation.
    """  
    # Compute the power spectral density ratio
    ratio = (spectra.energy/spectra_ref.energy).where(
                (spectra_ref.freq >= frequency_range[0]) 
                & (spectra_ref.freq <= frequency_range[1])
             ).rename('ratio')
    ds = ratio.to_dataset()
    
    # Compute pointwise attenuation
    ds['distance'] = (('time'), distance)
    ds['alpha'] = -np.log(ds['ratio'])/ds['distance']

    return ds


def fit_alpha(alpha, a0, b0, b_constraint = None):
    """
    Non linear least squares to a power law 
        alpha = a * b ^ freq
    using scipy
    """    

    # prepare data    
    freq = xr.broadcast(alpha.freq,alpha)[0]
    x = np.ravel(freq.values)
    y = np.ravel(alpha.values)

    used_points = (~np.isnan(y) )
    xdata = x[used_points]
    ydata = y[used_points]
    
    # Compute fit 
    if b_constraint:
        kwargs = {'p0' : (a0, b0),
                  'bounds': ([-np.inf, b_constraint[0]], [np.inf, b_constraint[1]])}
    else:
        kwargs = {'p0' : (a0, b0)}
    
    popt, pcov = curve_fit(_power_law, xdata, ydata, **kwargs)

    return (popt, pcov)

def alpha_boxplot(alpha):
    """
    Make a box-and-whisker plot with the derived frequency dependent attenuation
    """
    # Extract non-nan data points
    freqs = []
    data  = []

    for f in alpha.freq.values:

        x = alpha.sel(freq=f).values
        x = x[~np.isnan(x)]*1e5

        if len(x)>0:
            freqs.append(f)
            data.append(x)

    # Make boxplot        
    widths = 0.5*(alpha.freq.values[1]-alpha.freq.values[0])
    box_plot = plt.boxplot(data, positions=freqs, widths=widths)

    # Nicer median line
    for median in box_plot['medians']:
        median.set_color('black')
        median.set_zorder(-1)
        
    # Make x-axis nice again
    xlim = [freqs[0]-widths, freqs[-1]+widths]
    plt.xlim(xlim)
    plt.gca().xaxis.set_major_locator(AutoLocator())
    xticks = plt.xticks()[0]
    plt.xticks(ticks=xticks, labels=[str(round(tick,5)) for tick in xticks])
    plt.xlim(xlim)
    
    plt.xlabel('frequency (Hz)')
    plt.ylabel('$\\alpha$ ($10^{-5}$ m$^{-1}$)')
       
    return


def plot_fit(popt, **kwargs):
    """
    Plot fit from fit_alpha
    """    
    xlim = plt.gca().get_xlim()
    x = np.linspace(xlim[0],xlim[1],20)
    y = _power_law(x, *popt)
    plt.plot(x,y*1e5, **kwargs)
        
    return


def _power_law(f,a,b):
    return a*f**b