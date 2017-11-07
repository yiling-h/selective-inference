import glob
import os, numpy as np

def BH_selection_egenes(p_simes, level):

    m = p_simes.shape[0]
    p_sorted = np.sort(p_simes)
    indices = np.arange(m)
    indices_order = np.argsort(p_simes)
    order_sig = np.max(indices[p_sorted - np.true_divide(level * (np.arange(m) + 1.), m) <= 0])
    E_sel = indices_order[:(order_sig+1)]

    return order_sig+1, E_sel


path='/Users/snigdhapanigrahi/randomized_Bon_Z/'
allFiles = glob.glob(path + "/*.txt")
list_ = []
shapes = []
for file_ in sorted(allFiles):
    dataArray = np.loadtxt(file_)
    shapes.append(dataArray.shape[0])
    list_.append(dataArray)
length = len(list_)

print("length", length)

shapes = np.asarray(shapes)
v = np.cumsum(shapes)

simes_output = np.vstack(list_)
print("dimensions", simes_output.shape)

p_simes = simes_output[:,1]
sig = BH_selection_egenes(p_simes, 0.05)

K = sig[0]
print("K", K)
E_sel = np.sort(sig[1])
print("selected indices", E_sel, E_sel.shape[0])
egene_p = simes_output[E_sel, 0].sum()/float(K)
print("average number of SNPs", egene_p)
#
#
outdir = '/Users/snigdhapanigrahi/bon_output_liver/randomized_egenes_05/'
for i in range(v.shape[0]):

    if i == 0:
        outfile = os.path.join(outdir, "egene_index_" + str(format(i+1, '03')) + ".txt")
        E = E_sel[(E_sel< v[0])]
        if E.size>0:
            np.savetxt(outfile, E)
        else:
            E = (-1* np.ones(1)).reshape((1,))
            np.savetxt(outfile, E)
    elif i>0 and i<length-1:
        outfile = os.path.join(outdir, "egene_index_" + str(format(i+1, '03')) + ".txt")
        E = E_sel[(E_sel>= v[i-1]) & (E_sel< v[i])]-v[i-1]
        if E.size>0:
            np.savetxt(outfile, E)
        else:
            E = (-1* np.ones(1)).reshape((1,))
            np.savetxt(outfile, E)
    elif i == length-1:
        outfile = os.path.join(outdir, "egene_index_" + str(format(i+1, '03')) + ".txt")
        E = E_sel[(E_sel>= v[i-1])]-v[i-1]
        if E.size > 0:
            np.savetxt(outfile, E)
        else:
            E = (-1 * np.ones(1)).reshape((1,))
            np.savetxt(outfile, E)


outdir = '/Users/snigdhapanigrahi/bon_output_Liver/randomized_egene_boninfo_05/'
for i in range(v.shape[0]):

    if i == 0:
        outfile = os.path.join(outdir, "egene_index_" + str(format(i+1, '03')) + ".txt")
        E = E_sel[(E_sel< v[0])]
        if E.size>0:
            np.savetxt(outfile, simes_output[E,:])
    elif i>0 and i<length-1:
        outfile = os.path.join(outdir, "egene_index_" + str(format(i+1, '03')) + ".txt")
        E = E_sel[(E_sel>= v[i-1]) & (E_sel< v[i])]
        if E.size>0:
            np.savetxt(outfile, simes_output[E,:])
    elif i == length-1:
        outfile = os.path.join(outdir, "egene_index_" + str(format(i+1, '03')) + ".txt")
        E = E_sel[(E_sel>= v[i-1])]
        if E.size > 0:
            np.savetxt(outfile, simes_output[E,:])


# path='/Users/snigdhapanigrahi/nonrandomized_Bon_Z/'
# allFiles = glob.glob(path + "/*.txt")
# list_ = []
# shapes = []
# for file_ in sorted(allFiles):
#     dataArray = np.loadtxt(file_)
#     shapes.append(dataArray.shape[0])
#     list_.append(dataArray)
# length = len(list_)
#
# simes_output_1 = np.vstack(list_)
# print("dimensions", simes_output_1.shape)
#
# p_simes_1 = simes_output_1[:,1]
# sig_1 = BH_selection_egenes(p_simes_1, 0.10)
#
# E_sel_1 = np.sort(sig_1[1])
#
# print(np.intersect1d(E_sel, E_sel_1).shape)