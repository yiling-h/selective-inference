import glob
import os, numpy as np, pandas, statsmodels.api as sm

def BH_selection_egenes(p_simes, level):

    m = p_simes.shape[0]
    p_sorted = np.sort(p_simes)
    indices = np.arange(m)
    indices_order = np.argsort(p_simes)

    #if np.any(p_sorted - np.true_divide(level * (np.arange(m) + 1.), m) <= np.zeros(m)):

    order_sig = np.max(indices[p_sorted - np.true_divide(level * (np.arange(m) + 1.), m) <= 0])
    E_sel = indices_order[:(order_sig+1)]

    return order_sig+1, E_sel

path='/Users/snigdhapanigrahi/simes_output/'

allFiles = glob.glob(path + "/*.txt")
list_ = []
for file_ in allFiles:
    dataArray = np.loadtxt(file_)
    list_.append(dataArray)
length = len(list_)

simes_output = np.vstack(list_)

p_simes = simes_output[:,2]
sig = BH_selection_egenes(p_simes, 0.10)

true = (simes_output[:,1]>0.001).sum()
true_1 = (simes_output[:,1]==1).sum()
true_2 = (simes_output[:,1]==2).sum()
true_3 = (simes_output[:,1]==3).sum()
true_4 = (simes_output[:,1]==4).sum()
true_5 = (simes_output[:,1]==5).sum()
true_6 = (simes_output[:,1]==6).sum()
true_7 = (simes_output[:,1]==7).sum()
true_8 = (simes_output[:,1]==8).sum()
true_9 = (simes_output[:,1]==9).sum()
true_10 = (simes_output[:,1]==10).sum()

print("egenes", true)

K = sig[0]
E_sel = np.sort(sig[1])
p_0 = 0.
false = 0.
regime_1 = 0.
p_1 = 0.
regime_2 = 0.
p_2 = 0.
regime_3 = 0.
p_3 = 0.
regime_4 = 0.
p_4 = 0.
regime_5 = 0.
p_5 = 0.
regime_6 = 0.
p_6 = 0.
regime_7 = 0.
p_7 = 0.
regime_8 = 0.
p_8 = 0.
regime_9 = 0.
p_9 = 0.
regime_10 = 0.
p_10 = 0.

n = (simes_output[:, 0] < 5000).sum()
print("genes<5000 SNPs", n)

egene_5000 = (simes_output[E_sel, 0] < 5000).sum()
print("genes<5000 SNPs", egene_5000)

for j in range(K):

    if simes_output[E_sel[j],1] == 0:
        false +=1.
        p_0 += simes_output[E_sel[j],0]
    elif simes_output[E_sel[j],1] == 1:
        regime_1 += 1.
        p_1 += simes_output[E_sel[j], 0]
    elif simes_output[E_sel[j],1] == 2:
        regime_2 += 1.
        p_2 += simes_output[E_sel[j], 0]
    elif simes_output[E_sel[j],1] == 3:
        regime_3 += 1.
        p_3 += simes_output[E_sel[j], 0]
    elif simes_output[E_sel[j],1] == 4:
        regime_4 += 1.
        p_4 += simes_output[E_sel[j], 0]
    elif simes_output[E_sel[j],1] == 5:
        regime_5 += 1.
        p_5 += simes_output[E_sel[j], 0]
    elif simes_output[E_sel[j],1] == 6:
        regime_6 += 1.
        p_6 += simes_output[E_sel[j], 0]
    elif simes_output[E_sel[j],1] == 7:
        regime_7 += 1.
        p_7 += simes_output[E_sel[j], 0]
    elif simes_output[E_sel[j],1] == 8:
        regime_8 += 1.
        p_8 += simes_output[E_sel[j], 0]
    elif simes_output[E_sel[j],1] == 9:
        regime_9 += 1.
        p_9 += simes_output[E_sel[j], 0]
    elif simes_output[E_sel[j],1] == 10:
        regime_10 += 1.
        p_10 += simes_output[E_sel[j], 0]

print("FDR", false/K)
print("power", (K-false)/true)

print("break up of selected egenes", regime_1, regime_2, regime_3, regime_4, regime_5,
      regime_6, regime_7, regime_8, regime_9, regime_10)

print("break up of egenes", true_1, true_2, true_3, true_4, true_5,
      true_6, true_7, true_8, true_9, true_10)

print("average num of SNPs", p_0/false, p_1/regime_1, p_2/regime_2, p_3/regime_3, p_4/regime_4, p_5/regime_5,
      p_6/regime_6, p_7/regime_7, p_8/regime_8, p_9/regime_9)
