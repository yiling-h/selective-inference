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

    extra_rand = 0.
    extra_nonrand = 0.
    power_rand_extra = 0.
    power_nonrand_extra = 0.
    power_rand = 0.
    power_nonrand = 0.
    true_rand_sel = 0.
    true_nonrand_sel = 0.
    intersect_rand = 0.
    intersect_nonrand = 0.
    intersect_prop = 0.
    diff_prop_nonrand = 0.
    diff_prop_rand = 0.

    for i in range(length):
        results = list_[i]

        nsignals = results[6]
        extra_rand += results[0]
        extra_nonrand += results[1]

        power_rand_extra += results[2]/max(1., float(nsignals))
        power_nonrand_extra += results[3] / max(1., float(nsignals))

        power_rand += results[9]
        power_nonrand += results[10]

        true_rand_sel = results[7]
        true_nonrand_sel = results[8]

        nactive_rand = results[11]
        nactive_nonrand = results[12]

        intersect_rand += results[13]/max(1., float(nsignals))
        intersect_nonrand += results[14]/max(1., float(nsignals))

        intersect_prop += (max(true_nonrand_sel,1.)-results[3])/max(1., float(true_nonrand_sel))
        diff_prop_nonrand += results[3]/max(1., float(true_nonrand_sel))
        diff_prop_rand += results[2]/max(1., float(true_rand_sel))
        #print("results", results[9], results[10], nsignals, true_rand_sel, true_nonrand_sel, nactive_rand, nactive_nonrand)

    return power_rand_extra/length, power_nonrand_extra/length, power_rand/length, power_nonrand/length, intersect_nonrand/length, \
           intersect_prop/length, diff_prop_nonrand/length, diff_prop_rand/length

print(summary_files(list_))