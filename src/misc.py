import numpy as np
import os
from scipy.ndimage import gaussian_filter

def get_xy_limits_of_region(lola_ds, lon_min, lon_max, lat_min, lat_max, lon="Longitudes", lat="Latitudes"):
    """
    Find the limits of the smallest rectangle that enclose the region defined
    by the longitude-latitude limits.
    
    Parameters:
        lola_ds    xarray.Dataset with variables Longitudes, Latitudes
                   and dimensions y,x.
        lon_min    lower longitude limit
        lon_max    upper longitude limit
        lat_min    lower latitude limit
        lat_max    upper latitude limit
        lon        name of longitude variable in lola_ds
        lat        name of latitude variable in lola_ds
        
    Returns:
        list [ix_min, ix_max, iy_min, iy_max] with index of the edges of the
        smallest rectangle that encloses the region. Note: all values are inclusive.
        (thus, depending on function, might need to use [ix_min:(ix_max+1)])
    """
    
    lon = lola_ds[lon].values
    lat = lola_ds[lat].values
    
    lon_in_region = (lon >= lon_min) & (lon <= lon_max)
    lat_in_region = (lat >= lat_min) & (lat <= lat_max)
    region = lon_in_region & lat_in_region

    index_of_rows_in_region = np.flatnonzero(np.any(region, axis=1))
    index_of_cols_in_region = np.flatnonzero(np.any(region, axis=0))
    
    return [index_of_cols_in_region[0], 
            index_of_cols_in_region[-1],
            index_of_rows_in_region[0], 
            index_of_rows_in_region[-1]]


def nan_gaussian_filter(U, sigma):
    """
    Like scipy.ndimage.gaussian_filter(), but handles NaN values in input data
    Source: https://stackoverflow.com/a/36307291
    """
    # First check if sigma is 0, which means that no smoothing should be performed.
    if sigma == 0: 
        return U
    
    # Replace NaN with zeros and filter
    V = U.copy()
    V[np.isnan(U)] = 0
    VV = gaussian_filter(V,sigma=sigma)

    # The zeros will cause a bias in the filtered image, which is
    # compensated by this weighting
    W = 0*U.copy() + 1
    W[np.isnan(U)] = 0
    WW = gaussian_filter(W, sigma = sigma)

    ZZ = VV/WW
    
    # Finally, we add the NaN-values again
    ZZ[np.isnan(U)] = np.NaN

    return ZZ

def pandas_insert_with_overwrite(df, loc, column, value):
    """
    Add a new column to a pandas.DataFrame at a special location.
    """
        
    # Drop column if it already exists
    if column in df.columns:
        df.drop(labels = column, axis = "columns", inplace = True)
        print(f"Overwrote column {column}")
        
    if loc == "end":
        loc = len(df.columns)
    
    # Insert column
    df.insert(loc, column, value)
    
    return 

def symmetric_wrap(X, Y):
    """
    Like X%Y, but returns values Y/2 <= x < Y/2 instead of 0 <= x < Y 
    """  
    if isinstance(X,list):
        x = (np.array(X)+Y/2) % Y - Y/2
        return x.tolist()
    else:
        return (X+Y/2) % Y - Y/2