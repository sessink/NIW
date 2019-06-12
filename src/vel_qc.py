import os
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import seaborn as sns

testdata = xr.open_dataset('./data/xarray/xr_7701b_grid.nc')

testdata['u'] = 0.5*(testdata.u1 + testdata.u2)
testdata['v'] = 0.5*(testdata.v1 + testdata.v2)
