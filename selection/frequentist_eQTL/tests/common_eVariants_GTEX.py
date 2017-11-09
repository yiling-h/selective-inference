import numpy as np
import glob, os
import pandas as pd

corr_path = r'/Users/snigdhapanigrahi/inference_liver/correlations_10/'
allFiles = glob.glob(corr_path + "/*.txt")
count_common = 0.
count_tot = 0.
sel_tot = 0.
count_empty = 0.
for file_ in allFiles:
    df = np.loadtxt(file_)
    if df.ndim!= 0 and df.shape[0] != 0:
        if df.ndim > 1:
            sel_tot += df.shape[0]
            for k in range(df.shape[1]):
                count_common += np.any(df[:, k] > 0.50)
                #print("count", np.any(df[:, k] > 0.50))
                count_tot += 1.
        else:
            sel_tot += df.shape[0]
            count_common += np.any(df > 0.50)
            count_tot += 1.
    elif df.ndim == 0:
        df = np.asarray([df])
        count_common += np.any(df > 0.50)
        count_tot += 1.

print("count_common", count_common, count_tot, sel_tot, count_empty)