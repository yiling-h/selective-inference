import glob
import numpy as np

path =r'/Users/snigdhapanigrahi/inference_liver/nonrand_sel'

allFiles = glob.glob(path + "/*.txt")
list_ = []
for file_ in allFiles:
    df = np.loadtxt(file_)
    list_.append(df)

def summary_files(list_):

    length = len(list_)
    print("number of simulations", length)

    frac_overlap = 0.
    total_sel = 0.
    nactive_rand = 0.
    nactive_nonrand = 0.
    for i in range(length):

        results = list_[i]
        total_sel += results[1]
        nactive_rand += results[3]
        nactive_nonrand += results[4]
        frac_overlap += results[2]

    return frac_overlap/length, nactive_rand/length, nactive_nonrand/length, total_sel/length

print(summary_files(list_))


