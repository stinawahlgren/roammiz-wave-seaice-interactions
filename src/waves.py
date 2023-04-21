import numpy as np
import xarray as xr

import misc

def get_wavespectra(swift_df): 
    """
    Takes a swift dataframe as input and returns an xarray dataset with wavespectra data
    (all columns called wavespectra.<variable> in the swift dataframe). The xarray dataset will have
    coordinates 'time' and 'freq'.
    """
        
    group = 'wavespectra'
    coordinate_name = 'freq'    
    coordinate_column = (group + '.' + coordinate_name)
    
    # Find all columns with wavespectra data
    variable_columns = []   
    for col in swift_df.columns.difference({coordinate_column}):
        if col.startswith(group):
            variable_columns.append(col)
            
    # Convert wavespectra data to 2D-array and create a dictionary
    coordinate = swift_df[coordinate_column].iloc[0]
    data = {}
    for col in variable_columns:
        variable_name   = col[len(group)+1:]
        values = swift_df[col].to_numpy()       
        
        # If data is missing for a row,replace None with array of NaN (so that all rows has same length)
        values = [i if i is not (None or np.NaN) else np.NaN*np.zeros(len(coordinate)) for i in values]
        
        data[variable_name] = ([coordinate_name, 'time'], np.vstack(values).T)

    # Create xarray dataset                
    wave_ds = xr.Dataset(data_vars = data,
                         coords = dict(freq = coordinate,
                                       time = swift_df['timestamp'].values)
                        )

    return wave_ds


def swift_add_waveband_properties(swift, f_start = 0.05, f_stop = 0.14, bandname = "swell"):
    """
    Computes bulk wave parameters of part of the wavespectra and adds the following 
    column to the swift dataframe:
    
        <bandname>.peakdirT      : peak dominant direction of the specified frequency band 
                                   (deg in nautical convention)
        <bandname>.meandirT      : energy-weighted dominant direction of the specified 
                                   frequency band (deg in nautical convention)
        <bandname>.dirspread     : spread of the enery-weigthed dominant direction (deg)
        <bandname>.peakperiod    : peak period of the specified frequency band
        <bandname>.meanperiod    : energy-weighted period of the specified frequency band
        <bandname>.sigwaveheight : significant wave height of the specified frequency band
        
    Parameters:
        swift     pandas.DataFrame with SWIFT data (created using swiftutils.import_SWIFT)
        f_start   lower limit of the frequency band (inclusive)
        f_stop    higher limit of the frequency band (inclusive)
        bandname  string, used for naming the new columns
    
    Returns:
        Nothing (modifies swift)
    """
    
    # Get index of desired frequencies
    freq  = swift['wavespectra.freq'].loc[0]
    index = (freq >= f_start) & (freq <= f_stop)
    df    = np.mean(np.diff(freq))

    # Loop through rows in swift
    peak_dir = []
    mean_dir = []
    spread   = []
    swh      = []
    peak_per = []
    mean_per = []

    for (i,row) in swift.iterrows():
        swh.append(significant_wave_height(row['wavespectra.energy'][index],df)) 

        peak_per.append(peak_period(freq[index], row['wavespectra.energy'][index]))
        
        mean_per.append(mean_period(freq[index], row['wavespectra.energy'][index]))

        if all(np.isnan(row['wavespectra.a1'])): 
            peak_dir_this_row = np.NaN
            mean_dir_this_row = np.NaN
            spread_this_row   = np.NaN
            
        else:
            peak_dir_this_row = peak_direction(row['wavespectra.a1'][index], 
                                               row['wavespectra.b1'][index],
                                               row['wavespectra.energy'][index])

            mean_dir_this_row = mean_direction(row['wavespectra.a1'][index], 
                                               row['wavespectra.b1'][index],
                                               row['wavespectra.energy'][index])

            spread_this_row = mean_directional_spread(row['wavespectra.a1'][index], 
                                                      row['wavespectra.b1'][index],
                                                      row['wavespectra.energy'][index])
        peak_dir.append(peak_dir_this_row)
        mean_dir.append(mean_dir_this_row)
        spread.append(spread_this_row)
        
    # Add new columns
    misc.pandas_insert_with_overwrite(swift, "end", bandname+".peakdirT", peak_dir) 
    misc.pandas_insert_with_overwrite(swift, "end", bandname+".meandirT", mean_dir) 
    misc.pandas_insert_with_overwrite(swift, "end", bandname+".dirspread", spread)    
    misc.pandas_insert_with_overwrite(swift, "end", bandname+".peakperiod", peak_per) 
    misc.pandas_insert_with_overwrite(swift, "end", bandname+".meanperiod", mean_per) 
    misc.pandas_insert_with_overwrite(swift, "end", bandname+".sigwaveheight", swh) 
    
    return

def significant_wave_height(energy, df):
    """
    Computes the significant wave height based on energy spectra.
    Assumes an evenly sampled spectra.
    
    Input:
        energy : array with energy spectra
        df     : size of frequency bins
    """    
    return 4 * np.sqrt(df*nansum(energy))

def directional_moments2direction(a1,b1):
    """
    Computes direction in nautical convention using the first two directional moments
    See appendix in https://doi.org/10.1175/JTECH-D-17-0091.1 for a detailed axplanation
    of the directional moments.
    
    Input:
        a1    : array with first directional moment in east-west direction
        b1    : array with first directional moment in north-south direction   
    """
    dominant_direction__rad = np.arctan2(b1,a1) # Note "role" reversal, the syntax is np.arctan2(y,x)   
    # Convert to nautic convention
    dominant_direction = 90 - np.rad2deg(dominant_direction__rad)  
    return misc.symmetric_wrap(dominant_direction, 360)

def peak_direction(a1,b1,energy):  
    """
    Returns the peak dominant direction in nautical convention using 
    the first two directional moments and energy spectra.
    
    Note: will take the average over the discontinuity at 180 deg (South).
    
    Input:
        a1    : array with first directional moment in east-west direction
        b1    : array with first directional moment in north-south direction
        energy: array with energy spectra
    """   
    dominant_direction = directional_moments2direction(a1,b1)
    return dominant_direction[np.nanargmax(energy)]

def mean_direction(a1,b1,energy):
    """
    Computes energy-weighted dominant direction in nautical convention 
    using the first two directional moments and energy spectra.
    
    Note: Computes energy-weighted moments first, and then uses those
    to compute dominant direction.
    
    Input:
        a1    : array with first directional moment in east-west direction
        b1    : array with first directional moment in north-south direction
        energy: array with energy spectra
    """      
    a1_mean = nansum(energy * a1)/nansum(energy)
    b1_mean = nansum(energy * b1)/nansum(energy)
    mean_dir = directional_moments2direction(a1_mean,b1_mean)    
    return misc.symmetric_wrap(mean_dir, 360)


def mean_directional_spread(a1,b1,energy):
    """
    Computes energy-weighted directional spread in degrees
    using the first two directional moments and energy spectra.
    
    Note: Computes energy-weighted moments first, and then uses those
    to compute directional spread.
    
    Input:
        a1    : array with first directional moment in east-west direction
        b1    : array with first directional moment in north-south direction
        energy: array with energy spectra
    """        
    a1_mean = nansum(energy * a1)/nansum(energy)
    b1_mean = nansum(energy * b1)/nansum(energy)       
    return directional_spread(a1_mean, b1_mean)


def directional_spread(a1,b1):
    """
    Computes directional spread in degrees using the 
    first two directional moments.
    
    Input:
        a1    : array with first directional moment in east-west direction
        b1    : array with first directional moment in north-south direction
    
    """    
    directional_spread__rad = np.sqrt(2*(1-np.sqrt(a1**2 + b1**2)))    
    return np.rad2deg(directional_spread__rad)
                   

def peak_period(freq, energy):
    """
    Returns peak wave period. 
    
    Input:
        freq  : array with frequencies
        energy: array with energy spectra (same length as freq)
    """   
    if np.all(np.isnan(energy)):
        return np.nan
    else:
        return 1/freq[np.nanargmax(energy)]

    
def mean_period(freq, energy): 
    """
    Returns energy weighted wave period. 
    
    Input:
        freq  : array with (evenly spaced) frequencies
        energy: array with energy spectra (same length as freq)
    """
    return nansum(energy)/nansum(energy*freq)


def nansum(a, **kwargs):
    """
    Wrapper of numpy.nansum, since numpy.nansum will return 0
    if all elements are nan.
    """
    if np.isnan(a).all():
        return np.nan
    else:
        return np.nansum(a, **kwargs)
