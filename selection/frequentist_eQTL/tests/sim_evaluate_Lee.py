import os, numpy as np, pandas, statsmodels.api as sm
import glob

inf_path = '/Users/snigdhapanigrahi/sim_inference_liver/Lee_inf/'

allFiles = glob.glob(inf_path + "/*.txt")
list_ = []
for file_ in allFiles:
    df = np.loadtxt(file_)
    list_.append(df)

coverage_ad = np.zeros(10)
risk_ad = np.zeros(10)
length_ad = np.zeros(10)
coverage_unad = np.zeros(10)
risk_unad = np.zeros(10)
length_unad = np.zeros(10)
negenes = np.zeros(10)
print("length", len(list_))
for i in range(len(list_)):

    results = list_[i]
    nsignals = int(results[6])
    coverage_ad[nsignals] += results[0]
    coverage_unad[nsignals] += results[1]
    length_ad[nsignals] += results[2]
    length_unad[nsignals] += results[3]
    risk_ad[nsignals] += results[4]
    risk_unad[nsignals] += results[5]
    negenes[nsignals] += 1.

print("break-up", np.true_divide(coverage_unad, negenes),
          np.true_divide(risk_unad, negenes),
          np.true_divide(length_unad, negenes),
          negenes/len(list_))