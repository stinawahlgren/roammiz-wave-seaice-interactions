import icegeometry as ig
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import Point, Polygon


def test__add_distance_to_edge():

    x = [0, -2]
    y = [0, 2]
    lon = [0,90]
    edge = [np.array([[-2,0],[2,4]]), np.array([[-2,0],[2,4]])]
    wave_direction = [270,180];
    data_limits = pd.DataFrame({'x_min': -3, 'x_max': 3,
                                'y_min': -3, 'y_max': 6},
                              index = [0])
    timestamp = [pd.to_datetime("2021-09-22 13:30", format='%Y-%m-%d %H:%M'),
                 pd.to_datetime("2021-09-22 13:45", format='%Y-%m-%d %H:%M')]

    swift_df = pd.DataFrame({'timestamp': timestamp,
                             'x': x, 
                             'y': y,
                             'lon': lon,
                             'wave_direction':wave_direction})
    
    edge_df  = pd.DataFrame({'time': timestamp,
                             'x': [np.array([-2,2]), np.array([-2,2])],
                             'y': [np.array([0,4]), np.array([0,4])]})
    #edge_df.set_index('timestamp', inplace=True)

    ig.add_distance_to_edge(swift_df, edge_df, data_limits, 'wave_direction', 'wavedir')
    ig.add_distance_to_edge(swift_df, edge_df, data_limits, 'closest', 'closest')

    assert np.all(np.isclose(swift_df.dist_closest.values, [np.sqrt(2), -np.sqrt(2)]))
    assert np.all(np.isclose(swift_df.dist_wavedir.values, [2, -2]))
    assert np.all(np.isclose(swift_df.dir_wavedir.values, [270, 180]))
    assert np.all(np.isclose(swift_df.dir_closest.values, [315, 225]))

    
def test__get_distance_to_edge():
    
    # Interior point (positive distance)
    position     = [0,0]
    edge         = np.array([[-2,0],[2,4]])
    limits       = pd.DataFrame({'x_min': -2.2, 'x_max':  2.2,
                                 'y_min': -2.2, 'y_max': 6},
                               index = [0])
    direction_in = 3/4*np.pi

    (distance, direction) = ig.get_distance_to_edge(position, 
                                                    edge,
                                                    limits,
                                                    direction=direction_in)
                       
    assert np.isclose(distance, np.sqrt(2))
    assert np.isclose(direction, direction_in)
    
    # Exterior point (negative distance)
    position     = [-2,2]
    edge         = np.array([[-2,0],[2,4]])
    limits       = pd.DataFrame({'x_min': -2.2, 'x_max':  2.2,
                                 'y_min': -2.2, 'y_max': 6},
                                index = [0])
    direction_in = 3/4*np.pi

    (distance, direction) = ig.get_distance_to_edge(position, 
                                                    edge,
                                                    limits,
                                                    direction=direction_in)
                       
    assert np.isclose(distance, -np.sqrt(2))
    assert np.isclose(direction, direction_in)


def test__direction_between_points():

    p0 = Point(1,1)
    p1 = Point(2,1)    
    p2 = Point(0,1)    
    p3 = Point(2,2)
    
    assert np.isclose(0, np.rad2deg(ig.direction_between_points(p0,p1)))
    assert np.isclose(180, np.rad2deg(ig.direction_between_points(p0,p2)))
    assert np.isclose(45, np.rad2deg(ig.direction_between_points(p0,p3)))


def test__closest_distance_and_direction_to_polygon_edge():
    
    # First, test a simple and non-ambiguous geometry:
    
    edge = Polygon([(0,0),(0,1),(1,1),(1,0)])
    p0 = Point(0.5,0.8)
    
    (distance, direction) = ig.closest_distance_and_direction_to_polygon_edge(p0, edge)
    
    assert np.isclose(distance, 0.2)
    assert np.isclose(direction, np.pi/2)
    
    # Now let's test an ambiguous geometry as well, in
    # order to detect if shaply makes any changes. This
    # time, we have four different points on the edge 
    # that are equally close
    
    edge = Polygon([(0,0),(0,1),(1,1),(1,0)])
    p0 = Point(0.5,0.5)
    
    (distance, direction) = ig.closest_distance_and_direction_to_polygon_edge(p0, edge)
    
    assert np.isclose(distance, 0.5)
    assert np.isclose(direction, np.pi)
    
    # And finally, test for desired behaviour when p0 is outside the edge
    
    edge = Polygon([(0,0),(0,1),(1,1),(1,0)])
    p0 = Point(0.5,1.2)
    (distance, direction) = ig.closest_distance_and_direction_to_polygon_edge(p0, edge)

    assert np.isclose(distance, -0.2)
    assert np.isclose(direction, np.pi/2)


def test__get_contour_path():
    
    x = np.arange(5,7,0.5)
    y = np.arange(-1,1,0.5)
    
    data = np.array([[0.27915529, 0.41027838, 0.98384172, 0.82837876],
                     [0.8487735 , 0.55181228, 0.64400003, 0.16879045],
                     [0.34415605, 0.33608549, 0.74277705, 0.28150659],
                     [0.07607782, 0.1523038 , 0.77378087, 0.70323463]])
    
    contour_level = 0.5
    
    contour = np.array([[ 6.5       , -0.75107294],
                        [ 6.15151213, -0.5       ],
                        [ 6.26316128,  0.        ],
                        [ 6.5       ,  0.25904539]]).T
    
    assert np.all(np.isclose(contour, ig.get_contour_path(x,y,data, contour_level)))


def test__distance_to_polygon_edge():
    
    # First, test a simple geometry:
    
    edge = Polygon([(0,0),(0,1),(1,1),(1,0)])
    p0 = Point(0.5,0.5)
    direction = np.pi/4
    
    distance = ig.distance_to_polygon_edge(p0, edge, direction)
    
    assert np.isclose(distance, 0.5*np.sqrt(2))
    
    # Now let's test a geometr with multiple intersections, to check
    # that the distance to the closet intersection is returned
    
    edge = Polygon([(10,0), (10,8), (15,3), (20,10), (0,10), (0,0)])
    p0 = Point(5,5)
    direction = 0;
    
    distance = ig.distance_to_polygon_edge(p0, edge, direction)
    
    assert np.isclose(distance, 5)
                      
    # And finally, test for desired behaviour when p0 is outside the edge
    
    edge = Polygon([(0,0),(0,1),(1,1),(1,0)])
    p0 = Point(0.5,1.2)
    direction = np.pi/2
    distance = ig.distance_to_polygon_edge(p0, edge, direction)

    assert np.isclose(distance, -0.2)


def test__polygon_from_path():
    
    path = np.array([[10, 14.22],
                 [6, 13],
                 [4, 14.2],
                 [2, 14.5], 
                 [0, 15],])
    y_lim = 7
      
    poly = Polygon(([11, 7], 
                    [11, 14.22],
                    [10, 14.22],
                    [6, 13],
                    [4, 14.2],
                    [2, 14.5], 
                    [0, 15],
                    [-1, 15],
                    [-1, 7]))
    
    assert poly.equals(ig.polygon_from_path(path, y_lim))


def test__is_valid():
    
    position = np.array([3,2])
    limits = pd.DataFrame({'x_min': 0.3, 'x_max': 5, 'y_min': 1, 'y_max': 3}, index = [0])

    assert ig.is_valid(position, 1, np.pi/4, limits)
    assert ~ig.is_valid(position, 5, 0, limits)
    assert ~ig.is_valid(position, 5, np.pi/4, limits)
    assert ~ig.is_valid(position, 5, np.pi, limits)
    assert ~ig.is_valid(position, 5, -np.pi/2, limits)

    
def test__compass2math():
    
    compass_deg = np.array([0,90,180,270])
    math_rad    = np.array([np.pi/2, 0, 3/2*np.pi, np.pi])
    
    assert np.all(np.isclose(ig.compass2math(compass_deg), math_rad))


def test__math2compass():
    
    compass_deg = np.array([0,90,180,270])
    math_rad    = np.array([np.pi/2, 0, 3/2*np.pi, np.pi])
    
    assert np.all(np.isclose(ig.math2compass(math_rad), compass_deg))