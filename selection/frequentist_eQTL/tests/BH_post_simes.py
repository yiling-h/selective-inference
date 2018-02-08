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

path='/Users/snigdhapanigrahi/sim_nonrandomized_Bon_Z'

allFiles = glob.glob(path + "/*.txt")
list_ = []
for file_ in allFiles:
    dataArray = np.loadtxt(file_)
    list_.append(dataArray)
length = len(list_)
print("length", length)
simes_output = np.vstack(list_)
print("dimension", simes_output.shape)

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

K = sig[0]
E_sel = np.sort(sig[1])

false_rej = (simes_output[E_sel,1]<0.05)

print("false rejections", false_rej.sum())

print("fdr", false_rej.sum()/float(K))
print("power", (K-false_rej.sum())/float(true))


p_0 = 0.
false = 0.
regime_1 = 0.
p_1 = 0.
f_1 = 0.
regime_2 = 0.
p_2 = 0.
f_2 = 0.
regime_3 = 0.
p_3 = 0.
f_3 = 0.
regime_4 = 0.
p_4 = 0.
f_4 = 0.
regime_5 = 0.
p_5 = 0.
f_5 = 0.
regime_6 = 0.
p_6 = 0.
f_6 = 0.
regime_7 = 0.
p_7 = 0.
f_7 = 0.
regime_8 = 0.
p_8 = 0.
f_8 = 0.
regime_9 = 0.
p_9 = 0.
f_9 = 0.
regime_10 = 0.
p_10 = 0.
f_10 = 0.

nsig = (simes_output[E_sel,1]/0.30).astype(int)
tsig = (simes_output[:,1]/0.30).astype(int)

p1 = (nsig==1).sum()/float((tsig==1).sum())
p2 = (nsig==2).sum() /float((tsig==2).sum())
p3 = (nsig == 3).sum() / float((tsig == 3).sum())
p4 = (nsig == 4).sum() / float((tsig == 4).sum())
p5 = (nsig == 5).sum() / float((tsig == 5).sum())
p6 = (nsig == 6).sum() / float((tsig == 6).sum())
p7 = (nsig == 7).sum() / float((tsig == 7).sum())
p8 = (nsig == 8).sum() / float((tsig == 8).sum())
p9 = (nsig == 9).sum() / float((tsig == 9).sum())
print("breakup of power", p1, p2, p3, p4, p5, p6, p7, p8, p9)

print("breakup of fdr", )
# print(simes_output[E_sel,1])
# for j in range(int(K)):
#
#     if simes_output[E_sel[j],1]/0.3 <= 0.05:
#         false +=1.
#         p_0 += simes_output[E_sel[j],0]
#     elif simes_output[E_sel[j],1]/0.3 > 0.95 and simes_output[E_sel[j],1]/0.3 < 1.05:
#         regime_1 += 1.
#         p_1 += simes_output[E_sel[j], 0]
#     elif simes_output[E_sel[j],1] > 1.95 and simes_output[E_sel[j],1] < 2.05:
#         regime_2 += 1.
#         p_2 += simes_output[E_sel[j], 0]
#     elif simes_output[E_sel[j],1] > 2.95 and simes_output[E_sel[j],1] < 3.05:
#         regime_3 += 1.
#         p_3 += simes_output[E_sel[j], 0]
#     elif int(simes_output[E_sel[j],1]) == 4:
#         regime_4 += 1.
#         p_4 += simes_output[E_sel[j], 0]
#     elif int(simes_output[E_sel[j],1]) == 5:
#         regime_5 += 1.
#         p_5 += simes_output[E_sel[j], 0]
#     elif int(simes_output[E_sel[j],1]) == 6:
#         regime_6 += 1.
#         p_6 += simes_output[E_sel[j], 0]
#     elif int(simes_output[E_sel[j],1]) == 7:
#         regime_7 += 1.
#         p_7 += simes_output[E_sel[j], 0]
#     elif int(simes_output[E_sel[j],1]) == 8:
#         regime_8 += 1.
#         p_8 += simes_output[E_sel[j], 0]
#     elif int(simes_output[E_sel[j],1]) == 9:
#         regime_9 += 1.
#         p_9 += simes_output[E_sel[j], 0]
#     elif int(simes_output[E_sel[j],1]) == 10:
#         regime_10 += 1.
#         p_10 += simes_output[E_sel[j], 0]
#
# print("break up of selected egenes", regime_1, regime_2, regime_3, regime_4, regime_5,
#       regime_6, regime_7, regime_8, regime_9, regime_10)
#
# print("break up of egenes", true_1, true_2, true_3, true_4, true_5,
#       true_6, true_7, true_8, true_9, true_10)
#
# print("average num of SNPs", p_0/false, p_1/regime_1, p_2/regime_2, p_3/regime_3, p_4/regime_4, p_5/regime_5,
#       p_6/regime_6, p_7/regime_7, p_8/regime_8, p_9/regime_9)
