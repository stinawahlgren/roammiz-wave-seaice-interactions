import colocation as co
import numpy as np
import xarray as xr

def test__interp_xy_from_lola():

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

    # Test case
    ix = 1
    iy = 2
    x_orig = x[ix]
    y_orig = y[iy]
    lon_orig = lon[iy,ix]
    lat_orig = lat[iy,ix]
    (x_interp,y_interp) = co.interp_xy_from_lola(lola_ds, lon_orig, lat_orig)

    precision = 5
    assert np.round(x_orig - x_interp, precision) == 0
    assert np.round(y_orig - y_interp, precision) == 0