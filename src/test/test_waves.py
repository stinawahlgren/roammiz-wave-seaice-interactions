import waves
import numpy as np

def test__significant_wave_height():
    
    # Test nan-behaviour
    energy = np.array([np.nan, np.nan, np.nan])
    df = 0.1   
    assert np.isnan(waves.significant_wave_height(energy, df))
    
    # Test wave height
    energy = np.array([0.1, 0.2, 0.5, 0.2, np.nan])
    df = 0.1  
    swh_expected = 1.26491
    atol = 5
    assert np.isclose(waves.significant_wave_height(energy, df), swh_expected, atol=atol)
    
    
def test__directional_moments2direction():
    
    # Test nan-behaviour
    a1 = np.array([np.nan, np.nan, np.nan])
    b1 = np.array([0, np.nan, -1])   
    assert np.isnan(waves.directional_moments2direction(a1,b1)).all()
    
    # Test direction    
    a1 = np.array([1, -np.sqrt(2), 0])
    b1 = np.array([0, -np.sqrt(2), 1])    
    dir_expected = np.array([90, -135, 0])   
    atol=5
    assert np.allclose(waves.directional_moments2direction(a1,b1), dir_expected, atol=atol)

    
def test__peak_direction():
    
    # Test nan-behaviour
    a1 = np.array([np.nan, np.nan, np.nan])
    b1 = np.array([0, np.nan, -1])
    energy = np.array([0, 1, 0])
    assert np.isnan(waves.peak_direction(a1,b1,energy))
    
    # Test direction
    a1 = np.array([1, -np.sqrt(2), 0])
    b1 = np.array([0, -np.sqrt(2), 1])    
    energy = np.array([0, 1, 0])
    dir_expected = -135
    atol=5
    assert np.isclose(waves.peak_direction(a1,b1,energy), dir_expected, atol=atol)
    
    
def test__mean_direction():    

    # Test nan-behaviour
    a1 = np.array([np.nan, np.nan, 0])
    b1 = np.array([np.nan, np.nan, -1])
    energy = np.array([0, 1, np.nan])
    assert np.isnan(waves.mean_direction(a1,b1,energy))
    
    # Test direction
    a1 = np.array([-0.67, -0.7, -0.6])
    b1 = np.array([0.33,0.67,0.4])
    energy = np.array([0.1, 0.5, 0.2])
    dir_expected = -50.16
    atol = 2
    assert np.isclose(dir_expected,waves.mean_direction(a1,b1,energy), atol=atol)
    

def test__directional_spread():
    
    # Test nan-behaviour
    a1 = np.array([np.nan, np.nan, 0])
    b1 = np.array([-1, np.nan, np.nan])   
    assert np.isnan(waves.directional_spread(a1,b1)).all()
    
    # Test spread
    a1 = np.array([-1, 0, 0.3])
    b1 = np.array([0,  0, 0.4])
    spread_expected = np.array([0, np.rad2deg(np.sqrt(2)), np.rad2deg(1)])
    atol = 2   
    assert np.allclose(spread_expected, waves.directional_spread(a1,b1), atol=atol)
    

def test__mean_directional_spread():
    
    # Test nan-behaviour
    energy = [1,1.2,0]
    a1 = np.array([np.nan, np.nan, 0])
    b1 = np.array([np.nan, np.nan, np.nan])   
    assert np.isnan(waves.mean_directional_spread(a1,b1,energy))
    
    # Test spread
    a1 = np.array([-0.67, -0.7, -0.6])
    b1 = np.array([0.33,0.67,0.4])
    energy = np.array([0.1, 0.5, 0.2])
    spread_expected = 28.74
    atol = 2   
    assert np.isclose(spread_expected, waves.mean_directional_spread(a1,b1,energy), atol=atol)    
    
def test__peak_period(): 
    
    # Test nan-behaviour
    freq = np.array([0.1, 0.2, 0.3])
    energy = np.array([np.nan, np.nan, np.nan])   
    assert np.isnan(waves.peak_period(freq, energy))
    
    # Test period
    freq = np.array([0.1, 0.2, 0.3])
    energy = np.array([1, np.nan, 0.2])
    expected_period = 10
    atol = 5
    assert np.isclose(waves.peak_period(freq, energy), expected_period, atol=atol)
    
    
def test__mean_period(): 
    
    # Test nan-behaviour
    freq = np.array([0.1, 0.2, 0.3])
    energy = np.array([np.nan, np.nan, np.nan])   
    assert np.isnan(waves.mean_period(freq, energy))
    
    # Test period
    freq = np.array([0.1, 0.2, 0.3])
    energy = np.array([1, 2, np.nan])
    expected_period = 6
    atol = 5
    assert np.isclose(waves.mean_period(freq, energy), expected_period, atol=atol)    
    
    
def test__nansum():
    
    assert np.isnan(waves.nansum([np.nan, np.nan]))
    assert waves.nansum([np.nan, 1.2, 0.8]) == 2       
               