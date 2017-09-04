import glob
import os, numpy as np, pandas, statsmodels.api as sm
from collections import Counter

f=open('/Users/snigdhapanigrahi/Jason-sim-results/Liver_97_genes_simes_BH.txt',"r")
lines=f.readlines()
result=[]
current_indx = 0
indices = []
for x in lines:
    result.append(x.rstrip('\n').split('\t')[4])
    if '1' in x.rstrip('\n').split('\t')[4]:
        indices.append(current_indx )

    current_indx += 1
f.close()

a = dict(Counter(result))

print("result", a)

indices = np.asarray(indices) -1

def BH_selection_egenes(p_simes, level):

    m = p_simes.shape[0]
    p_sorted = np.sort(p_simes)
    indices = np.arange(m)
    indices_order = np.argsort(p_simes)

    #if np.any(p_sorted - np.true_divide(level * (np.arange(m) + 1.), m) <= np.zeros(m)):

    order_sig = np.max(indices[p_sorted - np.true_divide(level * (np.arange(m) + 1.), m) <= 0])
    E_sel = indices_order[:(order_sig+1)]

    return order_sig+1, E_sel


path='/Users/snigdhapanigrahi/sim_randomized_Bon_Z/'
allFiles = glob.glob(path + "/*.txt")
list_ = []
shapes = []
for file_ in allFiles:
    dataArray = np.loadtxt(file_)
    shapes.append(dataArray.shape[0])
    list_.append(dataArray)
length = len(list_)

print("length", length)

shapes = np.asarray(shapes)
print("shapes", shapes)
v = np.cumsum(shapes)
print("vector", v)
#print("shape", shapes.shape, shapes)
simes_output = np.vstack(list_)
print("dimensions", simes_output.shape)

p_simes = simes_output[:,2]
print("number of genes", p_simes.shape[0])
sig = BH_selection_egenes(p_simes, 0.10)

print("no of egenes selected", sig[0])

K = sig[0]
E_sel = np.sort(sig[1])
print("selected indices", E_sel, E_sel.shape[0])

print("intersection", np.intersect1d(E_sel, indices).shape)
