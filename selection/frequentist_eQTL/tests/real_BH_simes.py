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


path='/Users/snigdhapanigrahi/simes_F/'
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

p_simes = simes_output[:,1]
print("number of genes", p_simes.shape[0])
sig = BH_selection_egenes(p_simes, 0.10)

print("no of egenes selected", sig[0])

K = sig[0]
E_sel = np.sort(sig[1])
print("selected indices", E_sel, E_sel.shape[0])
egene_p = simes_output[E_sel, 0]
print("SNPs", egene_p)
#egene_p = (simes_output[E_sel, 0]).sum()/float(K)
#print("average number of SNPs", egene_p)


# outdir = '/Users/snigdhapanigrahi/simes_output_Liver/simes_output_norand/egenes/'
# for i in range(v.shape[0]):
#
#     if i == 0:
#         outfile = os.path.join(outdir, "egene_index_" + str(format(i+1, '03')) + ".txt")
#         E = E_sel[(E_sel< v[0])]
#         if E.size>0:
#             np.savetxt(outfile, E)
#         else:
#             E = (-1* np.ones(1)).reshape((1,))
#             np.savetxt(outfile, E)
#     elif i>0 and i<length-1:
#         outfile = os.path.join(outdir, "egene_index_" + str(format(i+1, '03')) + ".txt")
#         E = E_sel[(E_sel>= v[i-1]) & (E_sel< v[i])]-v[i-1]
#         if E.size>0:
#             np.savetxt(outfile, E)
#         else:
#             E = (-1* np.ones(1)).reshape((1,))
#             np.savetxt(outfile, E)
#     elif i == length-1:
#         outfile = os.path.join(outdir, "egene_index_" + str(format(i+1, '03')) + ".txt")
#         E = E_sel[(E_sel>= v[i-1])]-v[i-1]
#         if E.size > 0:
#             np.savetxt(outfile, E)
#         else:
#             E = (-1 * np.ones(1)).reshape((1,))
#             np.savetxt(outfile, E)

# outdir = '/Users/snigdhapanigrahi/simes_output_Liver/egenes/egene_index_'
# indices = np.loadtxt(outdir + str(format(2, '03'))+ ".txt")
# print("dims", indices)



