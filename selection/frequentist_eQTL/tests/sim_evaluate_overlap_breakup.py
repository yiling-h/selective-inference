import glob
import numpy as np
path =r'/Users/snigdhapanigrahi/sim_inference_liver/nonrand_sel'

allFiles = glob.glob(path + "/*.txt")
list_ = []
for file_ in allFiles:
    df = np.loadtxt(file_)
    list_.append(df)

def summary_files(list_):

    length = len(list_)
    print("number of simulations", length)

    power_rand_extra = np.zeros(10)
    power_nonrand_extra = np.zeros(10)
    power_rand = np.zeros(10)
    power_nonrand = np.zeros(10)
    intersect_prop = np.zeros(10)
    diff_prop_nonrand = np.zeros(10)
    diff_prop_rand = np.zeros(10)
    negenes = np.zeros(10)

    for i in range(length):
        results = list_[i]

        nsignals = int(results[6])

        negenes[nsignals] += 1.
        power_rand_extra += results[2]/max(1., float(nsignals))
        power_nonrand_extra += results[3] / max(1., float(nsignals))

        power_rand[nsignals] += results[9]
        power_nonrand[nsignals] += results[10]

        true_rand_sel = results[7]
        true_nonrand_sel = results[8]

        #intersect_prop[nsignals]  += (max(true_nonrand_sel,1.)-results[3])/max(1., float(true_nonrand_sel))
        #diff_prop_nonrand[nsignals] += results[3]/max(1., float(true_nonrand_sel))
        #diff_prop_rand[nsignals] += results[2]/max(1., float(true_rand_sel))

    return np.true_divide(power_rand, negenes), np.true_divide(power_nonrand, negenes)
           #intersect_prop/length, diff_prop_nonrand/length, diff_prop_rand/length

print(summary_files(list_))