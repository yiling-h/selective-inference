import os, numpy as np, pandas, statsmodels.api as sm
import glob

inf_path = '/Users/snigdhapanigrahi/sim_inference_liver/nonrand_inference/'

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
count = np.zeros(10)
print("length", len(list_))
for i in range(len(list_)):
    results = list_[i]
    print(results.shape, results.ndim)
    if results.ndim == 1 and results[3] == 0.:
        nsignals = int(results[6])
        count[nsignals] += 1
    elif results.ndim == 1 and results[3] > 0.:
        nsignals = int(results[6])
        nactive = 1
        coverage_ad[nsignals] += results[0]
        # if nsignals == 0.:
        #    print("coverage ad", coverage_ad)
        coverage_unad[nsignals] += results[1]
        length_ad[nsignals] += results[2]
        length_unad[nsignals] += results[3]
        risk_ad[nsignals] += results[4]
        risk_unad[nsignals] += results[5]
        negenes[nsignals] += 1.
    else:
        results = list_[i]
        nactive = float(results[:,7][0])
        nsignals = int(results[:,6][0])
        coverage_ad[nsignals] += results[:,0].sum()/nactive
        #if nsignals == 0.:
        #    print("coverage ad", coverage_ad)
        coverage_unad[nsignals] += results[:,1].sum()/nactive
        length_ad[nsignals] += results[:,2].sum()/nactive
        length_unad[nsignals] += results[:,3].sum()/nactive
        risk_ad[nsignals] += results[:,4].sum()/nactive
        risk_unad[nsignals] += results[:,5].sum()/nactive
        negenes[nsignals] += 1.

#print("ngenes", negenes)
#negenes[0] = negenes[0] - 50.
print("break-up", np.true_divide(coverage_ad, negenes),
          np.true_divide(risk_ad, negenes),
          np.true_divide(length_ad, negenes),
          negenes, count)