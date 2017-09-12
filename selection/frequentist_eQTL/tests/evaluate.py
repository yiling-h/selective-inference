import glob
import numpy as np

path =r'/Users/snigdhapanigrahi/inference_liver/inference0'

allFiles = glob.glob(path + "/*.txt")
list_ = []
for file_ in allFiles:
    df = np.loadtxt(file_)
    list_.append(df)

def summary_files(list_):

    length_ad = 0.
    length_unad = 0.
    length = len(list_)
    print("number of simulations", length)
    count = 0.

    for i in range(length):
        print("iteration", i)
        results = list_[i]
        count_0 = 0
        if results.size>0:
            if results.ndim > 1:
                nactive = results.shape[0]
            else:
                nactive = 1.
            print("nactive", nactive)

            if nactive > 1:
                ci_sel_l = results[:, 0]
                ci_sel_u = results[:, 1]

                unad_l = results[:, 2]
                unad_u = results[:, 3]

                length_adj = (ci_sel_u - ci_sel_l).sum() / float(nactive)
                length_unadj = (unad_u - unad_l).sum() / float(nactive)
                length_ad += length_adj
                length_unad += length_unadj
                print("lengths", length_adj, length_unadj)
                if length_adj == length_unadj:
                    count += 1.

            elif nactive == 1:
                ci_sel_l = results[0]
                ci_sel_u = results[1]

                unad_l = results[2]
                unad_u = results[3]

                length_adj = (ci_sel_u - ci_sel_l).sum() / float(nactive)
                length_unadj = (unad_u - unad_l).sum() / float(nactive)
                length_ad += length_adj
                length_unad += length_unadj
                print("lengths", length_adj, length_unadj)
                if length_adj == length_unadj:
                    count += 1.
        else:
            count_0 += 1

    return length_ad/length, length_unad/length, count_0, count

print("results", summary_files(list_))