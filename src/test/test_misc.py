import numpy as np
import xarray as xr
import misc


def test__get_xy_limits_of_region():

    # Mockup lolagrid:    
    x = np.array(np.arange(-5,-2))
    y = np.array(np.arange(0,3,0.5))

    lon = np.array([[5.  , 6.  , 7.  ],
                    [5.05, 6.05, 7.05],
                    [5.1 , 6.1 , 7.1 ],
                    [5.15, 6.15, 7.15],
                    [5.2 , 6.2 , 7.2 ],
                    [5.25, 6.25, 7.25]])

    lat = np.array([[-0.5, -0.4, -0.3],
                    [ 0. ,  0.1,  0.2],
                    [ 0.5,  0.6,  0.7],
                    [ 1. ,  1.1,  1.2],
                    [ 1.5,  1.6,  1.7],
                    [ 2. ,  2.1,  2.2]])

    lola_ds = xr.Dataset({"Longitudes": (["y","x"],lon),
                          "Latitudes":  (["y","x"],lat)}
                         ).assign_coords(x=x,y=y)
    
    # Testcases
    assert misc.get_xy_limits_of_region(lola_ds, 6.0, 7.2, -np.inf, np.inf) == [1,2,0,5]
    assert misc.get_xy_limits_of_region(lola_ds, -np.inf, np.inf, 0.5, 1.2) == [0,2,2,3]
    assert misc.get_xy_limits_of_region(lola_ds, 6.0, 7.2, 0.5, 1.2) == [1,2,2,3]
    

def test__symmetric_wrap():
    
    x = [-90, 360, 400, 320]
    y = 360
    assert np.all(np.isclose(misc.symmetric_wrap(x,y),
                             [-90,0,40,-40]))