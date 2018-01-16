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

    shapes = np.asarray(shapes)
    v = np.cumsum(shapes)
    simes_output = np.vstack(list_)

    p_simes = simes_output[:, 2]
    true_egenes = (simes_output[:, 1]>0.05).sum()
    print("true egenes", true_egenes)
    sig = BH_selection_egenes(p_simes, 0.10)
    print("sig", sig)

    K = sig[0]
    print("BH rejections", K)
    E_sel = np.sort(sig[1])
    print("selected indices", E_sel, E_sel.shape[0])
    egene_p = simes_output[E_sel, 0].sum() / float(K)

    egene_sel = simes_output[E_sel,:]
    false_rej = egene_sel[:,1]==0

    print("fdr", false_rej.sum()/float(K))
    print("power", (K-false_rej.sum())/(float(true_egenes)))

    print("average number of SNPs in selected egenes", egene_p)

    return K, E_sel, v, length, simes_output

BH_output = egene_selection('/Users/snigdhapanigrahi/selective-inference/selection/frequentist_eQTL/simulation_prototype/bonferroni_output/')
selected_indices = BH_output[1]
outfile = os.path.join("/Users/snigdhapanigrahi/selective-inference/selection/frequentist_eQTL/simulation_prototype/" + "eGenes.txt")
np.savetxt(outfile, selected_indices)
