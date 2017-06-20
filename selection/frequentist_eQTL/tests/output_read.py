import glob
import os, numpy as np, pandas, statsmodels.api as sm
from collections import Counter

f=open('/Users/snigdhapanigrahi/Jason-results/Liver_97_genes_simes_BH.txt',"r")
lines=f.readlines()
result=[]
for x in lines:
    result.append(x.rstrip('\n').split('\t')[4])
f.close()

a = dict(Counter(result))

print("result", a)