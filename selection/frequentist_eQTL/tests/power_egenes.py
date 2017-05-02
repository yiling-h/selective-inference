from __future__ import print_function
import time
from scipy.stats import norm as normal
import os, numpy as np, pandas, statsmodels.api as sm
from selection.randomized.api import randomization
from selection.tests.instance import gaussian_instance

from selection.frequentist_eQTL.simes_BH_selection import simes_selection, BH_simes, BH_selection_egenes, simes_selection_egenes

def power_test():

    # get all the Simes p_values
    p_simes = np.zeros(5000)

    for i in range(1000):
        X, y, true_beta, nonzero, noise_variance = gaussian_instance(n=350, p=700, s=0, sigma=1, rho=0, snr=5.)
        p_simes[i] = simes_selection(X, y)

    print("regime 0 done")

    for i in range(1000):
        X, y, true_beta, nonzero, noise_variance = gaussian_instance(n=350, p=700, s=1, sigma=1, rho=0, snr=5.)
        p_simes[i+1000] = simes_selection(X, y)

    print("regime 1 done")

    for i in range(1000):
        X, y, true_beta, nonzero, noise_variance = gaussian_instance(n=350, p=700, s=3, sigma=1, rho=0, snr=5.)
        p_simes[i+2000] = simes_selection(X, y)

    print("regime 3 done")

    for i in range(1000):
        X, y, true_beta, nonzero, noise_variance = gaussian_instance(n=350, p=700, s=5, sigma=1, rho=0, snr=5.)
        p_simes[i+3000] = simes_selection(X, y)

    print("regime 5 done")

    for i in range(1000):
        X, y, true_beta, nonzero, noise_variance = gaussian_instance(n=350, p=700, s=10, sigma=1, rho=0, snr=5.)
        p_simes[i+4000] = simes_selection(X, y)

    print("regime 10 done")

    nsig = BH_simes(p_simes, 0.10)

    print("power", nsig[0])

    print("false rejections", nsig[1])

    return nsig[0], nsig[1]

#power_test()

def test_simes(bh_level, ngenes =5000):

    p_simes = np.zeros(ngenes)

    for i in range(1000):
        X, y, true_beta, nonzero, noise_variance = gaussian_instance(n=350, p=700, s=0, sigma=1, rho=0, snr=5.)
        simes = simes_selection_egenes(X, y)
        p_simes[i] = simes.simes_p_value()

    print("regime 0 done")

    for i in range(1000):
        X, y, true_beta, nonzero, noise_variance = gaussian_instance(n=350, p=700, s=1, sigma=1, rho=0, snr=5.)
        simes = simes_selection_egenes(X, y)
        p_simes[i + 1000] = simes.simes_p_value()

    print("regime 1 done")

    for i in range(1000):
        X, y, true_beta, nonzero, noise_variance = gaussian_instance(n=350, p=700, s=3, sigma=1, rho=0, snr=5.)
        simes = simes_selection_egenes(X, y)
        p_simes[i + 2000] = simes.simes_p_value()

    print("regime 3 done")

    for i in range(1000):
        X, y, true_beta, nonzero, noise_variance = gaussian_instance(n=350, p=700, s=5, sigma=1, rho=0, snr=5.)
        simes = simes_selection_egenes(X, y)
        p_simes[i + 3000] = simes.simes_p_value()

    print("regime 5 done")

    for i in range(1000):
        X, y, true_beta, nonzero, noise_variance = gaussian_instance(n=350, p=700, s=10, sigma=1, rho=0, snr=5.)
        simes = simes_selection_egenes(X, y)
        p_simes[i + 4000] = simes.simes_p_value()

    print("regime 10 done")

    sig = BH_selection_egenes(p_simes, 0.10)

    K = sig[0]

    E_sel = np.sort(sig[1])

    simes_level = K*bh_level/ngenes

    return simes_level, E_sel

def selection_information(simes_level= 0.065):

    X, y, true_beta, nonzero, noise_variance = gaussian_instance(n=350, p=700, s=50, sigma=1, rho=0, snr=5.)

    simes = simes_selection_egenes(X, y)

    return simes.post_BH_selection(simes_level)

print(selection_information())