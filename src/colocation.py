import xarray as xr
import pandas as pd
import numpy as np
from scipy.interpolate import griddata

import misc

def swift_collocate_sic(swift, sic, method="nearest", column="sic", loc=4):
    """
    Adds a column with colocated sea ice concentration to SWIFT dataframe. 
    Note: SWIFT_df must have columns x,y. Add those by running 
    colocation.add_xy_to_SWIFT first.  

    Parameters:
        swift   pandas.DataFrame with SWIFT data (created using swiftutils.import_SWIFT) 
        sic     xarray.Dataset with sea ice concentrations and and dimensions time,y,x.
        method  colocation method "nearest" or "linear" (passed to xarray.Dataset.interp)
        column  Name of the created column with sea ice concentration. Default: "sic"
        loc     Location of inserted column. Default: 4
        
    Returns:
        Nothing. (Modifies swift dataframe)
    """
    
    # Create a new dimension along the SWIFT track to collocate with
    x    = xr.DataArray(swift["x"].values, dims="track")
    y    = xr.DataArray(swift["y"].values, dims="track")
    time = xr.DataArray(swift["timestamp"].values, dims="track")

    # Colocate data with this track
    colocated_sic = sic.interp(x = x,
                               y = y,
                               time = time,
                               method = method,
                              )
    
    # Add column to SWIFT_df
    misc.pandas_insert_with_overwrite(swift, loc, column, colocated_sic.z.values)   
    
    return


def add_xy_to_swift(swift, sic, loc=2):
    """
    Adds polar stereographic coordinates xy to SWIFT_df
    
    Parameters:
        swift   pandas.DataFrame with SWIFT data (created using swiftutils.import_SWIFT)
        sic     xarray.Dataset with variables Longitudes, Latitudes
                   and dimensions y,x.    
    Returns:
        Nothing (Modifies swift dataframe)
    """
    (x,y) = interp_xy_from_lola(sic, swift['lon'], swift['lat'])
    
    misc.pandas_insert_with_overwrite(swift, loc, "x", x)
    misc.pandas_insert_with_overwrite(swift, loc+1, "y", y)
       
    return

def interp_xy_from_lola(lola_ds, lon, lat):
    """
    Returns coordinates that best match lon lat using linear interpolation.
    
    Parameters:
        lola_ds    xarray.Dataset with variables Longitudes, Latitudes
                   and dimensions y,x.
        lon        1D array with longitudes that will be interpolated
        lat        1D array with latitudes that be interpolated
        
    Returns:       Mx2 ndarray with x,y-values corresponding to lon lat
    """    
    
    # Create matrices with x- and y-values for all lon/lat in lola_ds
    xy = np.meshgrid(lola_ds.x, lola_ds.y)
    
    # Interpolate x- and y-values
    x = griddata(np.array([lola_ds.Longitudes.values.ravel(), 
                           lola_ds.Latitudes.values.ravel()]).T, 
                           np.ravel(xy[0]),
                           np.array([lon,lat]).T,
                           method="linear")

    y = griddata(np.array([lola_ds.Longitudes.values.ravel(), 
                           lola_ds.Latitudes.values.ravel()]).T, 
                           np.ravel(xy[1]),
                           np.array([lon,lat]).T,
                           method="linear")
    
    return (x,y)


def swift_add_icetype(swift, icefile):
    """
    Adds ice type from file to swift dataframe.
    
    Ice type is stored in a csv file with the following columns:
        date: YYYY/mm/dd
        hour: int between 0 and 23
        icetype: int   
    
    Parameters:
        swift   : swift dataframe to be modified
        icefile : path to the ice file
        
    Returns: nothing (adds column 'icetype' to swift)    
    """  

    """
    First, add an icetype column to swift. We will start by setting
    the icetype to -7 (missing data)
    """
    misc.pandas_insert_with_overwrite(swift, 5, 'icetype', -7)
    
    print("Adding icetype...")
    if icefile is None:
        print("No data")
        return
    
    """
    Convert date+hour from icefile to pandas timestamp
    """
    timestamp = []
    icetype_df = pd.read_csv(icefile, sep=',', header=0)
    for (i,row) in icetype_df.iterrows():
        timestamp.append(pd.to_datetime(row.date+"T"+str(row.hour).zfill(2), format='%Y/%m/%dT%H'))

    icetype_df['timestamp'] = timestamp
    
    """
    Now we add an icetype column to the swift dataframe
    """
    icetype = swift['icetype'].copy() # We shouldn't update things we're iterating over. 
                                      # That's why this temporary copy
    for (i,row) in swift.iterrows():

        timestamp_match = (row.timestamp == icetype_df.timestamp)
        number_of_matches = sum(timestamp_match)

        if number_of_matches == 1:
            icetype.iloc[i] = icetype_df.loc[timestamp_match].icetype

        elif number_of_matches == 0:
            print(f"Didn't find an icetype for {row.timestamp}")

        elif number_of_matches > 1:
            print(f"Found multiple icetypes for {row.timestamp}, ignoring all")

    swift['icetype'] = icetype    
    return