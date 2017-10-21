import glob, os, numpy as np

path = '/Users/snigdhapanigrahi/fwd_bwd_inference/inference/'
allFiles = glob.glob(path + "/*.txt")
list_ = []
for file_ in allFiles:
    df = np.loadtxt(file_)
    list_.append(df)

coverage_unad = np.zeros(10)
risk_unad = np.zeros(10)
length_unad = np.zeros(10)
#power = np.zeros(10)
negenes = np.zeros(10)
#nactive = 0.
#coverage_unad = 0.
#risk_unad = 0.
#length_unad = 0.
for i in range(len(list_)):
    results = list_[i]

    nsignals = int(results[0])
    print(nsignals)
    #power[nsignals] += results[1]
    coverage_unad[nsignals] += results[2]
    risk_unad[nsignals] += results[4]
    length_unad[nsignals] += results[3]
    negenes[nsignals] += 1.
    #nactive += nsignals

print("break-up", np.true_divide(coverage_unad, negenes), np.true_divide(risk_unad, negenes),
      np.true_divide(length_unad, negenes), np.true_divide(negenes, float(len(list_))))

#print("break-up",  coverage_unad/float(len(list_)), risk_unad/float(len(list_)), length_unad/float(len(list_)))