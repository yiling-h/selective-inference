import glob
import numpy as np, os

def BH_selection_egenes(p_simes, level):

    m = p_simes.shape[0]
    p_sorted = np.sort(p_simes)
    indices = np.arange(m)
    indices_order = np.argsort(p_simes)
    order_sig = np.max(indices[p_sorted - np.true_divide(level * (np.arange(m) + 1.), m) <= 0])
    E_sel = indices_order[:(order_sig+1)]

    return order_sig+1, E_sel

def egene_selection(inpath):

    allFiles = glob.glob(inpath + "/*.txt")
    list_ = []
    shapes = []
    for file_ in sorted(allFiles):
        dataArray = np.loadtxt(file_)
        shapes.append(dataArray.shape[0])
        list_.append(dataArray)
    length = len(list_)
    print("check 0: number of files", length)

    shapes = np.asarray(shapes)
    v = np.cumsum(shapes)
    simes_output = np.vstack(list_)
    print("check 1: dimensions of concatenated bon outputs", simes_output.shape)

    p_simes = simes_output[:, 2]
    sig = BH_selection_egenes(p_simes, 0.10)

    K = sig[0]
    print("BH rejections", K)
    E_sel = np.sort(sig[1])
    print("selected indices", E_sel, E_sel.shape[0])
    egene_p = simes_output[E_sel, 0].sum() / float(K)
    print("average number of SNPs in selected egenes", egene_p)

    return K, E_sel, v, length, simes_output

BH_output = egene_selection('/Users/snigdhapanigrahi/sim_randomized_Bon_Z/')
E_sel = BH_output[1]
v = BH_output[2]
length = BH_output[3]
simes_output = BH_output[4]

# outdir = '/Users/snigdhapanigrahi/sim_bon_output_liver/randomized_egenes/'
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


# outdir = '/Users/snigdhapanigrahi/sim_bon_output_Liver/randomized_egene_boninfo/'
# for i in range(v.shape[0]):
#
#     if i == 0:
#         outfile = os.path.join(outdir, "egene_index_" + str(format(i+1, '03')) + ".txt")
#         E = E_sel[(E_sel< v[0])]
#         if E.size>0:
#             np.savetxt(outfile, simes_output[E,:])
#     elif i>0 and i<length-1:
#         outfile = os.path.join(outdir, "egene_index_" + str(format(i+1, '03')) + ".txt")
#         E = E_sel[(E_sel>= v[i-1]) & (E_sel< v[i])]
#         if E.size>0:
#             np.savetxt(outfile, simes_output[E,:])
#     elif i == length-1:
#         outfile = os.path.join(outdir, "egene_index_" + str(format(i+1, '03')) + ".txt")
#         E = E_sel[(E_sel>= v[i-1])]
#         if E.size > 0:
#             np.savetxt(outfile, simes_output[E,:])
