import numpy as np
import pandas as pd
from shapely.geometry import Point, LineString, Polygon
from shapely.ops import nearest_points
import matplotlib.pyplot as plt
import warnings

import misc

def get_contour_path(x, y, data, contour_level):
    """
    Returns the vertices of the contour line at level contour_level. If more than
    one contour line exists, the one with the most vertices (probably corresponding
    to the longest line) is chosen.
    """    
    
    cs = plt.contour(x,y,data, levels=[contour_level])
    plt.close() # close the figure
    
    # Loop through contour lines and pick the one with most vertices
    longest_path = 0
    vertices_longest_path = []
    for p in cs.collections[0].get_paths():
        if len(p.vertices) > longest_path:
            longest_path = len(p.vertices)
            vertices_longest_path = p.vertices 

    return vertices_longest_path.T


def add_distance_to_edge(swift_df, edge_df, data_limits, in_direction, label, 
                         time_col = 'timestamp', make_plots = False, 
                         xlim = None, ylim = None, lon_col = 'lon'):
    """
    Computes distance to the ice edge. Distance will be in the same unit as the columns "x", "y" in 
    swift_df and all directions are in "compass degrees", i.e. 0 along positive y-axis.
    The function will add (or overwrite) the following columns to swift_df:
    
        dist_(label): Distance from buoy location to the ice edge. Negative if bouy 
                      is outside the ice.
                              
        dir_(label):  The direction in where dist_(label) is taken. This is defined
                      as the direction from buoy to the edge if distance is positive, and
                      the direction from edge to buoy if distance is negative. 
                                                          
     
     Parameters:
         
         swift_df:        pandas.DataFrame with columns 'timestamp' and 'x', 'y' (with buoy location)
         
         edge_df:         pandas.DataFrame with timestamp as index and columns 'x', 'y' with arrays 
                          representing the ice edge.
        
         data_limits:     pandas.DataFrame with columns 'x_min', 'x_max', 'y_min', 'y_max' defining the 
                          area where the ice edge is valid. If no intersection with the ice edge is found 
                          in this area, distance will be set to NaN and a warning will be thrown.
         
         in_direction:    If in_direction = "closest", the shortest distance to the edge will be computed. 
                          Otherwise, the distance in a specified direction will be computed. The direction
                          can be given either as a number, or a string. If a number, that wave direction 
                          will be used to derive distance in wave direction for all data points. If a string,
                          the direction will be taken from the column swift_df[in_direction]. Direction is given
                          in degrees north.
                          
         label:           Suffix of the new columns added to swift_df
         
         make_plots:      If True, a figure with ice edge, buoy location, and the found distances to 
                          the edge will be made for every 12th row in swift_df. Default: False
         
         xlim, ylim:      Optional plot limits.
         
         time_col:        Column with timestamps used for picking correct ice edge. Default: "timestamp"
         
         lon_col:         Column with longitudes used for converting derived distance to edge in polar
                          stereorgaphic proejection to nautical convention (degrees relative to north)
    
    Returns: Nothing (modifies swift_df)
    """
    
    # Things get much simpler if we use the time column in the ice edge data frame as index
    _edge_df = edge_df.set_index('time')

    distance_all  = []
    direction_all = []

    for (i,row) in swift_df.iterrows():

        if pd.isnull(row.x) or pd.isnull(row.y):
            distance = np.nan
            direction = np.nan
            
        else:
            # find index in ice_edge correspondning to current time
            edge_index = _edge_df.index.get_loc(row[time_col], method ='nearest')
            edge_x = _edge_df['x'][edge_index]
            edge_y = _edge_df['y'][edge_index]
            edge   = np.array([edge_x, edge_y]).T
            
            if in_direction == "closest": # Find shortest distance

                (distance, direction_xy) = get_distance_to_edge([row.x, row.y], 
                                                                edge, 
                                                                data_limits, 
                                                                direction = "closest"
                                                               )
            else: # Find distance in specified direction

                if isinstance(in_direction, str):
                    wave_dir = row[in_direction]
                else:
                    wave_dir = in_direction

                # Adjust to polar stereographic bearing (0 deg pointing in positive y-direction) 
                wave_dir_xy = wave_dir + row[lon_col]
                # Change to meterological convention (direction where waves are coming from)
                wave_dir_xy = wave_dir_xy + 180
                if ~np.isnan(wave_dir):
                    (distance, direction_xy) = get_distance_to_edge([row.x, row.y], 
                                                                    edge, 
                                                                    data_limits, 
                                                                    direction = compass2math(wave_dir_xy)
                                                                   )
                else:
                    distance = np.nan
                    direction_xy = np.nan

            # Convert direction from polar stereographic to compass bearing
            if ~np.isnan(direction_xy):
                direction = math2compass(direction_xy) - row[lon_col]
            else:
                direction = np.nan
                
        direction_all.append(direction);
        distance_all.append(distance);

        if make_plots:
            if i%12 == 0:

                # Make a plot with position, ice edge, and distances as a sanity check
                ax = plt.subplot(1,1,1)
                ax.plot(edge_x, edge_y)
                
                if xlim is not None:
                    ax.set_xlim(xlim)
                
                if ylim is not None:
                    ax.set_ylim(ylim)    
                
                ax.set_aspect('equal', 'box')

                ax.plot(row.x, row.y, '*r', markersize=10)
                plot_distance(ax, [row.x, row.y], distance, direction, linestyle='--')

                plt.xlabel("x")
                plt.ylabel("y")
                plt.title(str(row[time_col]))
                plt.legend(['ice edge',
                            'buoy location',
                            'distance to edge'])
                plt.show()
     
    # Add new columns to swift_df    
    misc.pandas_insert_with_overwrite(swift_df, "end", "dist_" + label, distance_all)
    misc.pandas_insert_with_overwrite(swift_df, "end", "dir_" + label, direction_all)
   
    return 

def get_distance_to_edge(location, edge, data_limits, direction = "closest"):
    """
    Computes distance from location to the edge in the specified direction, or if
    no direction is given, to the closest point on the edge. If location is above 
    the edge, distance is negative and direction is flipped. (This represents the 
    case where the location is in open water instead of sea ice. The ice is assumed 
    to extend down from the edge.)
    
    Parameters:
    
        location     array [x,y] with x- and y- coordinates representing the location.
        
        edge         X-by-2 numpy.ndarray representing the edge. This can for 
                     example be a contour line generated from matlplotlib.pyplot with:
                       cs = matplotlib.pyplot.contour(x,y,data)
                       path = cs.collections[0].get_paths()[0].vertices

        data_limits  pandas.DataFrame with columns 'x_min', 'x_max', 'y_min', 'y_max' defining the 
                      area where the ice edge is valid. If no intersection with the ice edge is found 
                      in this area, distance will be set to NaN and a warning will be thrown.
                     
        direction    The direction from the location to the edge, in which the distance
                     will be measured. The direction is given in radians in "math 
                     coordinate system", i.e. 0 means to the right of the location.
                     If omitted, the closest direction to the edge will be used.
                     
    Returns tuple (distance, direction) where
        
        distance   Distance from the lolcation to the edge. If the point is outside
                   the ice, distance is negative.

        direction  Direction from location to the edge, corresponding to the distance. 
                   The direction is given in radians in "math coordinate system", i.e.
                   0 means to the right of the location. If distance is negative, the
                   direction is flipped, i.e. the direction from the edge to the 
                   location
    """
    
    # Define polygon from edge
    lower_polygon_limit = 2*data_limits.y_min[0] - data_limits.y_max[0]  # Same as ymin - (ymax - ymin)
    edge_polygon = polygon_from_path(edge, lower_polygon_limit)
    
    # Get distance (and direction)
    if type(direction) is str:
        
        if direction.lower() == "closest":
            
            (distance, direction) = closest_distance_and_direction_to_polygon_edge(Point(location), edge_polygon)
            
        else:
            raise ValueError("Unknown value for kwarg direction: " +  direction)
            
    else:
        
        distance = distance_to_polygon_edge(Point(location), edge_polygon, direction)
    
    # Check if result is valid
    if not np.isnan(distance): 
        if not is_valid(location, distance, direction, data_limits):   
            warnings.warn("The found point on edge is outside data limits. Please adjust direction or provide a longer section of the edge.")
            distance = np.nan
            direction = np.nan
        
    return (distance, direction)   


def polygon_from_path(path, y_lim):
    """
    Creates a shaply polygon with path as upper (or lower) edge and and extending downwards 
    (or upwards) to y_lim. Also adds a little bit of padding to the sides, so that it will be
    easy to tell if detected points on the edge is outside the data limit or not.   
    
    Parameters:
        path    X-by-2 numpy.ndarray representing the path. This can for example be a contour
                line generated from matlplotlib.pyplot with:
                    cs = matplotlib.pyplot.contour(x,y,data)
                    path = cs.collections[0].get_paths()[0].vertices
                
        y_lim    desired y-coordinate of the lower (or upper) boundary of the polygon
        
    Returns: a shaply.geometry.polygon.Polygon
    
    """
    
    x_padding = (path[-1,0] - path[0,0])/10
    
    # Pad the path with vertices that will create an extension downwards (or upwards)
    verticies_before = [[path[0,0] - x_padding, y_lim],
                        [path[0,0] - x_padding, path[0,1]]]
    verticies_after  = [[path[-1,0] + x_padding, path[-1,1]],
                        [path[-1,0] + x_padding, y_lim]]    
    path = np.concatenate((verticies_before, path, verticies_after), axis=0)

    # Create a polygon
    return Polygon([(i[0], i[1]) for i in zip(path[:,0],path[:,1])])



def direction_between_points(point1, point2):
    """
    Returns the direction to point2 from point1. Direction is given in 
    radians in "math coordinate system", i.e 0 means that point2 lies 
    to the right of point1
    
    Parameters:
        point1, point2:  shapely.geometry.Point objects
        
    Returns:
        angel in radians
    """
    
    dx = point2.x - point1.x
    dy = point2.y - point1.y
    
    return np.arctan2(dy,dx)%(2*np.pi)


def closest_distance_and_direction_to_polygon_edge(p0, edge):
    """
    Returns closest distance and direction from the point to 
    to the edge. If the point lies outside the polygon, the
    distance is given as a negative number.

    Parameters:

        p0        shapely.geometry.point representing buoy location

        edge      shapely.geometry.polygon representing the ice edge

    Returns tuple (distance, direction):

        distance:  Distance to the closest part of the edge

        direction: Direction from p0 to closest part of 
                   the edge. The direction is given in radians 
                   in "math coordinate system", i.e. 0 means 
                   to the right of the point p0. 

    """

    # Get closets distance between p0 and edge
    distance = edge.exterior.distance(p0)

    # Get the point on edge closest to the p0
    p1 = nearest_points(edge.exterior, p0)[0]

    # Get direction to closest point
    direction = direction_between_points(p0, p1)  
    
    # Change sign of distance if the original point p0
    # is not in sea ice
    if not p0.within(edge):
        distance = -distance
        direction = (direction + np.pi) % (2*np.pi)

    return (distance, direction)


def distance_to_polygon_edge(p0, edge, direction):
    """
    Returns distance from the point p0 to the edge in the
    given direction. If the point lies outside the polygon, the
    distance is given as a negative number and the direction
    is flipped, so that it is from the edge to the point.

    Parameters:

        p0         shapely.geometry.point.Point representing buoy location

        edge       shapely.geometry.polygon.Polygon representing the ice edge
        
        direction  Direction from p0 to the edge. The direction is given in 
                   radians in "math coordinate system", i.e. 0 means 
                   to the right of the point p0. If p0 is 
                   outside the polygon, the direction is reversed.

    Returns distance to the edge in the given direction.
    
    """
    
    # Check if point is inside the area bounded by edge
    if not p0.within(edge):
        # flip direction
        direction = (direction + np.pi) % (2*np.pi)
        distance_sign = -1
    else:
        distance_sign = 1
    
    """
    Create a line in given direction that is guaranteed to be longer 
    than any distance between a point in the interior and any point 
    on the edge.
    """
    (xmin, ymin, xmax, ymax) = edge.bounds
    line_length = 2*np.max([xmax-xmin, ymax-ymin])

    help_line = LineString([p0, Point(p0.x+np.cos(direction)*line_length, p0.y+np.sin(direction)*line_length)])

    """
    Find the distance to the closest point on the edge that intersects with the help line
    """
    crossings = help_line.intersection(edge.exterior)
    
    return distance_sign*p0.distance(crossings)


def is_valid(position, distance, direction, data_limits):
    """
    Check if a point distance away in direction is within the data limits. Useful for 
    making sure that the derived distance and direction the ice edge is valid.
    
    Parameters:
        position:      Array with x- and y-position (of e.g. a buoy)
        
        distance:      Distance from position to the point
        
        direction:     Direction from position to the point. The direction is given in 
                       radians in "math coordinate system", i.e. 0 means 
                       to the right of the position.
                   
        data_limits:   pandas.DataFrame with columns 'x_min', 'x_max', 'y_min', 'y_max' defining the 
                        area where the ice edge is valid. If no intersection with the ice edge is found 
                        in this area, distance will be set to NaN and a warning will be thrown.
        
    Returns a boolean: True if the point is within data limits, False otherwise
    """    
    x = position[0] + distance*np.cos(direction)
    y = position[1] + distance*np.sin(direction)
    ok_x = (x > data_limits.x_min[0]) & (x < data_limits.x_max[0])
    ok_y = (y > data_limits.y_min[0]) & (y < data_limits.y_max[0])
    return  ok_x & ok_y


def compass2math(direction_compass_deg):
    """
    Converts direction from the compass degrees to radians
    in "math coordinate system", i.e with 0 at positive
    x-axis and going counterclockwise.
    
    Parameter:
    
        direction_compass_deg:  number or np.array with directions
                                in compass degrees
                              
    Returns: number or np.array with direction in "math" radians
    
    """
    direction_math_deg = -direction_compass_deg + 90;
    direction_math_rad = np.deg2rad(direction_math_deg)
    
    # Note: modulus with nan generates an annoying warning
    direction_math_rad = direction_math_rad % (2*np.pi)
    
    return direction_math_rad


def math2compass(direction_math_rad):
    """
    Converts direction from radians in "math coordinate system", 
    i.e with 0 at positive x-axis and going counterclockwise, to
    compass degrees.
    
    Parameter:
    
        direction_math_rad:  number or np.array with directions
                             in "math" radians
                              
    Returns: number or np.array with direction in compass degrees
    
    """
    direction_compass_rad = -direction_math_rad + np.pi/2;
    direction_compass_deg = np.rad2deg(direction_compass_rad)
            
    # Note: modulus with nan generates an annoying warning
    direction_compass_deg = direction_compass_deg%360
    
    return direction_compass_deg


def plot_distance(ax, location, distance, direction, **kwargs):
    """
    Note: direction in navigation convention
    """
    direction = compass2math(direction)
    x = location[0] + np.array([0, distance*np.cos(direction)])
    y = location[1] + np.array([0, distance*np.sin(direction)])
    
    ax.plot(x,y, **kwargs)
    return