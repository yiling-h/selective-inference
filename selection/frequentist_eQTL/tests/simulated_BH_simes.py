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


path='/Users/snigdhapanigrahi/sim_hs_randomized_Bon_Z/'
allFiles = glob.glob(path + "/*.txt")
list_ = []
shapes = []
for file_ in sorted(allFiles):
    dataArray = np.loadtxt(file_)
    #print "Current File Being Processed is: " + file_
    shapes.append(dataArray.shape[0])
    list_.append(dataArray)
length = len(list_)

shapes = np.asarray(shapes)
v = np.cumsum(shapes)
simes_output = np.vstack(list_)
print("dimensions", simes_output.shape)

indices = simes_output[:,0] >0
sel_indices = np.asarray([i for i in range(indices.shape[0]) if indices[i]])
print("indices", sel_indices)
simes_output_sel = simes_output[indices, :]

###do a break up to see percentages
ind_0 = (simes_output_sel[:,1] ==0).sum()
ind_1 = (simes_output_sel[:,1] ==1).sum()
ind_2 = (simes_output_sel[:,1] ==2).sum()
ind_3 = (simes_output_sel[:,1] ==3).sum()
ind_5 = (simes_output_sel[:,1] ==5).sum()
ind_10 = (simes_output_sel[:,1] ==10).sum()

print("break-up", ind_0/19555., ind_1/19555., ind_2/19555., ind_3/19555., ind_5/19555., ind_10/19555.)

p_simes = simes_output_sel[:,2]
print("number of genes after removing the zeros", p_simes.shape[0])

sig = BH_selection_egenes(p_simes, 0.10)

print("no of egenes selected", sig[0])

K = sig[0]
print("K", K)
E_sel = np.sort(sig[1])
#print("selected indices", E_sel, E_sel.shape[0])
egene_p = simes_output_sel[E_sel, 0]
#print("SNPs", egene_p)

egene_0 = (simes_output_sel[E_sel, 1] == 0).sum()
print("FDR", egene_0/float(K))
print("power", (K-egene_0)/(19555.-ind_0))

indices_0 = np.asarray([i for i in range(indices.shape[0]) if indices[i]])
E_sel_0 = indices_0[E_sel]

print("original indices sel", E_sel_0)

# outdir = '/Users/snigdhapanigrahi/sim_hs_bon_output_liver/randomized_egenes/'
# for i in range(v.shape[0]):
#
#     if i == 0:
#         outfile = os.path.join(outdir, "egene_index_" + str(format(i+1, '03')) + ".txt")
#         E = E_sel_0[(E_sel_0< v[0])]
#         if E.size>0:
#             np.savetxt(outfile, E)
#         else:
#             E = (-1* np.ones(1)).reshape((1,))
#             np.savetxt(outfile, E)
#     elif i>0 and i<length-1:
#         outfile = os.path.join(outdir, "egene_index_" + str(format(i+1, '03')) + ".txt")
#         E = E_sel_0[(E_sel_0>= v[i-1]) & (E_sel_0< v[i])]-v[i-1]
#         if E.size>0:
#             np.savetxt(outfile, E)
#         else:
#             E = (-1* np.ones(1)).reshape((1,))
#             np.savetxt(outfile, E)
#     elif i == length-1:
#         outfile = os.path.join(outdir, "egene_index_" + str(format(i+1, '03')) + ".txt")
#         E = E_sel_0[(E_sel_0>= v[i-1])]-v[i-1]
#         if E.size > 0:
#             np.savetxt(outfile, E)
#         else:
#             E = (-1 * np.ones(1)).reshape((1,))
#             np.savetxt(outfile, E)

# outdir = '/Users/snigdhapanigrahi/sim_hs_bon_output_Liver/randomized_egene_boninfo/'
# for i in range(v.shape[0]):
#
#     if i == 0:
#         outfile = os.path.join(outdir, "egene_index_" + str(format(i+1, '03')) + ".txt")
#         E = E_sel_0[(E_sel_0< v[0])]
#         if E.size>0:
#             np.savetxt(outfile, simes_output[E,:])
#     elif i>0 and i<length-1:
#         outfile = os.path.join(outdir, "egene_index_" + str(format(i+1, '03')) + ".txt")
#         E = E_sel_0[(E_sel_0>= v[i-1]) & (E_sel_0< v[i])]
#         if E.size>0:
#             np.savetxt(outfile, simes_output[E,:])
#     elif i == length-1:
#         outfile = os.path.join(outdir, "egene_index_" + str(format(i+1, '03')) + ".txt")
#         E = E_sel_0[(E_sel_0>= v[i-1])]
#         if E.size > 0:
#             np.savetxt(outfile, simes_output[E,:])