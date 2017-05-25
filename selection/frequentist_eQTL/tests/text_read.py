import glob
import os, numpy as np, pandas, statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.stats import probplot, uniform

path =r'/Users/snigdhapanigrahi/Results_freq_EQTL'
allFiles = glob.glob(path + "/*.txt")

for file_ in allFiles:
    df = np.loadtxt(file_, usecols=(1,))

print("mean",np.mean(df, axis =0))
print("median",np.median(df, axis =0))

