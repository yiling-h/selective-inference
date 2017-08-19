import glob
import os, numpy as np, pandas, statsmodels.api as sm

# path='/Users/snigdhapanigrahi/simes_output_Liver/sigma_est_output/'
# outdir = '/Users/snigdhapanigrahi/simes_output_Liver/sigma_est_output/combined'
#
# for i in range(100):
#
#     i = i + 1
#
#     list = []
#     list.append(np.loadtxt(os.path.join(path, "1_simes_output_sigma_estimated_"+ str(format(i, '03')) + ".txt")))
#     list.append(np.loadtxt(os.path.join(path, "2_simes_output_sigma_estimated_"+ str(format(i, '03')) + ".txt")))
#
#     file = np.vstack(list)
#
#     print("file shape", file.shape)
#     outfile = os.path.join(outdir, "simes_output_sigma_estimated_" + str(format(i, '03')) + ".txt")
#     np.savetxt(outfile, file)

# path = '/Users/snigdhapanigrahi/Test_simes'
#
# allFiles = glob.glob(path + "/*.txt")
# list_ = []
# for file_ in allFiles:
#     df = np.loadtxt(file_)
#     list_.append(df)
#
# simes_output = np.vstack(list_)
# print("p", simes_output[:,0])
# print("simes output", simes_output[:,1])

import glob
import os, numpy as np, pandas, statsmodels.api as sm

path =r'/Users/snigdhapanigrahi/inference_liver/inference'

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
        if results.ndim> 1:
            nactive = results.shape[0]
        else:
            nactive = 1.
        print("nactive", nactive)

        if nactive>1:
            ci_sel_l = results[:, 0]
            ci_sel_u = results[:, 1]

            unad_l = results[:, 2]
            unad_u = results[:, 3]

        else:
            ci_sel_l = results[0]
            ci_sel_u = results[1]

            unad_l = results[2]
            unad_u = results[3]

        length_adj = (ci_sel_u - ci_sel_l).sum()/float(nactive)
        length_unadj = (unad_u - unad_l).sum()/float(nactive)
        length_ad += length_adj
        length_unad += length_unadj
        print("lengths", length_adj, length_unadj)
        if length_adj == length_unadj:
            count += 1.

    return length_ad/length, length_unad/length, count

print("results", summary_files(list_))