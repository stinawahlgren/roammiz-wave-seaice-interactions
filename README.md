# Direct observations of wave-sea ice interactions in the Antarctic Marginal Ice Zone
*Authors: S.Wahlgren, J. Thomson, L. Biddle, S. Swart*

This is a repository for the data analysis performed in Wahlgren et al. (submitted to JGR:Oceans). Preprint DOI: [10.22541/essoar.168201718.84053333/v1](https://doi.org/10.22541/essoar.168201718.84053333/v1)  


## Summary
Frequency dependent attenuation of swell waves in the Antarctic Marginal Ice Zone is studied using in-situ data from two [SWIFT wave buoys](https://apl.uw.edu/project/project.php?id=swift). The variation in swell wave direction with distance from the sea ice edge is also examined.

## Workflow
The workflow is divided into preprocessing of data and analysis. Preprocessed data is provided in this repository, so it is possible to jump right to the analysis part.

The conda environment which lists the dependencies for this code is provided [here](./environment.yml). 

Notebooks and functions are provided in the [src](./src/) folder, together with unit tests for some of the functions. The unit tests can be run with`pytest`.

### Preprocessing of data
SWIFT buoy data need to be converted to python-friendly format using read_swift in [swifttools.py](https://github.com/SASlabgroup/SWIFT-codes/blob/master/Python/swifttools.py) beforehand.

1) Preprocessing of sea ice concentration - [01_preprocess_sea_ice_concentration.ipynb](src/01_preprocess_sea_ice_concentration.ipynb)
2) Finding the ice edge [02_defining_the_ice_edge.ipynb](src/02_defining_the_ice_edge.ipynb)
3) Co-location and data fusion [03_data_fusion.ipynb](src/03_data_fusion.ipynb)

The preprocessed data is found in [the processed data folder](./processed_data/).

### Analysis
4) Data overview (Figure 4) [04_overview.ipynb](src/04_overview.ipynb)
5) Winter attenuation (Figure 5a) [05_wave_attenuation_winter.ipynb](src/05_wave_attenuation_winter.ipynb)
6) Spring attenuation (Figure 5b) [06_wave_attenuation_spring.ipynb](src/06_wave_attenuation_spring.ipynb)
7) Wave direction (Figure 7) [07_wave_direction_vs_distance.ipynb](src/07_wave_direction_vs_distance.ipynb)

## Data source
The SWIFT buoy data is published [here](https://doi.org/10.5281/zenodo.7845764). In addition, [AMSR sea ice concentration](https://seaice.uni-bremen.de/databrowser/#p=sic) is used in the analysis.

## Citation
### Data
The provided data should be cited as

Thomson, Jim, Biddle, Louise C, Wahlgren, Stina, & Swart, Sebastiaan. (2023). Direct observations of wave-sea ice interactions in the Antarctic Marginal Ice Zone [Data set]. Zenodo. https://doi.org/10.5281/zenodo.7845764

### Code
If you find the methods presented here useful, please consider citing the paper.