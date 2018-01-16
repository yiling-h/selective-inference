import glob
import numpy as np

path =r'/Users/snigdhapanigrahi/simulation_prototype/inference_simulation/'

allFiles = glob.glob(path + "/*.txt")
list_ = []
for file_ in allFiles:
    df = np.loadtxt(file_)
    list_.append(df)

def summary_files(list_):
    count = 0.
    count_0  = 0.
    length_ad = 0.
    length_unad = 0.
    cov_ad = 0.
    cov_unad = 0.
    risk_ad = 0.
    risk_unad = 0.

    for i in range(len(list_)):
        print("iteration", i)
        results = list_[i]
        if results.size>0:
            count += 1.
            if results.ndim > 1:
                nactive = results.shape[0]
            else:
                nactive = 1
            print("nactive", nactive)

            if nactive > 1:
                print("results", results, results.shape)
                length_ad += results[:,9].sum()/float(nactive)
                length_unad += results[:,10].sum()/float(nactive)
                cov_ad += results[:,5].sum()/float(nactive)
                cov_unad += results[:,6].sum()/float(nactive)
                risk_ad += results[:,7].sum()/float(nactive)
                risk_unad += results[:,8].sum()/float(nactive)

            elif nactive == 1:
                length_ad += results[9] / float(nactive)
                length_unad += results[10] / float(nactive)
                cov_ad += results[5] / float(nactive)
                cov_unad += results[6] / float(nactive)
                risk_ad += results[7] / float(nactive)
                risk_unad += results[8] / float(nactive)
        else:
            count_0 += 1

    return count, length_ad/count, length_unad/count, cov_ad/count, cov_unad/count, risk_ad/count, risk_unad/count

print("results", summary_files(list_))

# def summary_files(list_):
#
#     length_ad = 0.
#     length_unad = 0.
#     prop_ad = 0.
#     prop_unad = 0.
#     length = len(list_)
#     print("number of simulations", length)
#     count = 0.
#     avg_sel = 0.
#     mag_sel = 0.
#     mag_unad = 0.
#
#     for i in range(length):
#         print("iteration", i)
#         results = list_[i]
#         count_0 = 0
#         if results.size>0:
#             if results.ndim > 1:
#                 nactive = results.shape[0]
#             else:
#                 nactive = 1.
#             print("nactive", nactive)
#
#             if nactive > 1:
#                 ci_sel_l = results[:, 0]
#                 ci_sel_u = results[:, 1]
#                 count_ad_l = ci_sel_l > 0
#                 count_ad_u = ci_sel_u < 0
#                 count_ad = np.logical_or(count_ad_l, count_ad_u).sum()
#                 prop_ad += count_ad/float(nactive)
#
#                 avg_sel += float(nactive)
#
#                 unad_l = results[:, 2]
#                 unad_u = results[:, 3]
#                 count_unad_l = unad_l > 0
#                 count_unad_u = unad_u < 0
#                 count_unad = np.logical_or(count_unad_l, count_unad_u).sum()
#                 prop_unad += count_unad/float(nactive)
#
#                 sel_MLE = results[:, 4]
#                 mag_sel += ((sel_MLE-np.zeros(nactive))**2.).sum()/float(nactive)
#
#                 unad_MLE = results[:, 5]
#                 mag_unad += ((unad_MLE-np.zeros(nactive))**2.).sum()/float(nactive)
#
#                 length_adj = (ci_sel_u - ci_sel_l).sum() / float(nactive)
#                 length_unadj = (unad_u - unad_l).sum() / float(nactive)
#                 length_ad += length_adj
#                 length_unad += length_unadj
#
#                 print("magnitudes", ((sel_MLE-np.zeros(nactive))**2.).sum()/float(nactive),
#                       ((unad_MLE-np.zeros(nactive))**2.).sum()/float(nactive))
#                 if length_adj == length_unadj:
#                     count += 1.
#
#             elif nactive == 1:
#                 ci_sel_l = results[0]
#                 ci_sel_u = results[1]
#                 count_ad_l = ci_sel_l > 0
#                 count_ad_u = ci_sel_u < 0
#                 count_ad = np.logical_or(count_ad_l, count_ad_u).sum()
#                 prop_ad += count_ad / float(nactive)
#
#                 avg_sel += float(nactive)
#
#                 unad_l = results[2]
#                 unad_u = results[3]
#                 count_unad_l = unad_l > 0
#                 count_unad_u = unad_u < 0
#                 count_unad = np.logical_or(count_unad_l, count_unad_u).sum()
#                 prop_unad += count_unad / float(nactive)
#
#                 sel_MLE = results[4]
#                 mag_sel += (sel_MLE - 0.) ** 2.
#
#                 unad_MLE = results[5]
#                 mag_unad += (unad_MLE - 0.) ** 2.
#
#                 length_adj = (ci_sel_u - ci_sel_l).sum() / float(nactive)
#                 length_unadj = (unad_u - unad_l).sum() / float(nactive)
#                 length_ad += length_adj
#                 length_unad += length_unadj
#
#                 print("magnitudes", (sel_MLE - 0.) ** 2., (unad_MLE - 0.) ** 2.)
#                 if length_adj == length_unadj:
#                     count += 1.
#         else:
#             count_0 += 1
#
#     return avg_sel/length, length_ad/length, length_unad/length, prop_ad/length, prop_unad/length, \
#            mag_sel/length, mag_unad/length, count_0, count
#
# print("results", summary_files(list_))

# nsig = 0.
# norm = 0.
# length_unad = 0.
# count_sel = 0
# for i in range(len(list_)):
#     results = list_[i]
#
#     nactive = int(results[0])
#     if nactive == 0:
#         count_sel += 1
#
#     nsig += results[4]
#     norm += results[3]
#     length_unad += results[2]
#
# print("summary stats",  nsig/len(list_), norm/len(list_), length_unad/len(list_), count_sel)
#
