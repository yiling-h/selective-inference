from __future__ import print_function
import time
from scipy.stats import norm as normal
import os, numpy as np, pandas, statsmodels.api as sm
from selection.randomized.api import randomization
from selection.tests.instance import gaussian_instance

from selection.frequentist_eQTL.instance import instance

from selection.frequentist_eQTL.simes_BH_selection import simes_selection, BH_simes, BH_selection_egenes, simes_selection_egenes
from selection.frequentist_eQTL.tests.power_randomized_lasso import random_lasso

def power_test():

    # get all the Simes p_values
    p_simes = np.zeros(5000)

    for i in range(1000):
        X, y, true_beta, nonzero, noise_variance = instance(n=350, p=700, s=0, sigma=1, rho=0, snr=5.)
        p_simes[i] = simes_selection(X, y)

    print("regime 0 done")

    for i in range(1000):
        X, y, true_beta, nonzero, noise_variance = instance(n=350, p=700, s=1, sigma=1, rho=0, snr=5.)
        p_simes[i+1000] = simes_selection(X, y)

    print("regime 1 done")

    for i in range(1000):
        X, y, true_beta, nonzero, noise_variance = instance(n=350, p=700, s=3, sigma=1, rho=0, snr=5.)
        p_simes[i+2000] = simes_selection(X, y)

    print("regime 3 done")

    for i in range(1000):
        X, y, true_beta, nonzero, noise_variance = instance(n=350, p=700, s=5, sigma=1, rho=0, snr=5.)
        p_simes[i+3000] = simes_selection(X, y)

    print("regime 5 done")

    for i in range(1000):
        X, y, true_beta, nonzero, noise_variance = instance(n=350, p=700, s=10, sigma=1, rho=0, snr=5.)
        p_simes[i+4000] = simes_selection(X, y)

    print("regime 10 done")

    nsig = BH_simes(p_simes, 0.10)

    print("power", nsig[0])

    print("false rejections", nsig[1])

    return nsig[0], nsig[1]

#power_test()

def test_simes(bh_level=0.10, ngenes =5000, n= 350, p=1000):

    p_simes = np.zeros(ngenes)

    for i in range(500):
        np.random.seed(i)  # ensures same X
        sample = instance(n=n, p=p, s=5, snr=5., sigma=1., rho=0)
        np.random.seed(i)
        X, y, beta, nonzero, sigma = sample.generate_response()
        simes = simes_selection_egenes(X, y)
        p_simes[i] = simes.simes_p_value()

    print("regime 5 done")

    for i in range(500):
        np.random.seed(i+ 500)  # ensures same X
        sample = instance(n=n, p=p, s=4, snr=5., sigma=1., rho=0)
        np.random.seed(i+ 500)
        X, y, beta, nonzero, sigma = sample.generate_response()
        simes = simes_selection_egenes(X, y)
        p_simes[i + 500] = simes.simes_p_value()

    print("regime 4 done")

    for i in range(800):
        np.random.seed(i + 1000)  # ensures same X
        sample = instance(n=n, p=p, s=4, snr=5., sigma=1., rho=0)
        np.random.seed(i + 1000)
        X, y, beta, nonzero, sigma = sample.generate_response()
        simes = simes_selection_egenes(X, y)
        p_simes[i + 1000] = simes.simes_p_value()

    print("regime 3 done")

    for i in range(1200):
        np.random.seed(i + 1800)  # ensures same X
        sample = instance(n=n, p=p, s=4, snr=5., sigma=1., rho=0)
        np.random.seed(i + 1800)
        X, y, beta, nonzero, sigma = sample.generate_response()
        simes = simes_selection_egenes(X, y)
        p_simes[i + 1800] = simes.simes_p_value()

    print("regime 2 done")

    for i in range(1500):
        np.random.seed(i + 3000)  # ensures same X
        sample = instance(n=n, p=p, s=4, snr=5., sigma=1., rho=0)
        np.random.seed(i + 3000)
        X, y, beta, nonzero, sigma = sample.generate_response()
        simes = simes_selection_egenes(X, y)
        p_simes[i + 3000] = simes.simes_p_value()

    print("regime 1 done")

    for i in range(500):
        np.random.seed(i + 4500)  # ensures same X
        sample = instance(n=n, p=p, s=4, snr=5., sigma=1., rho=0)
        np.random.seed(i + 4500)
        X, y, beta, nonzero, sigma = sample.generate_response()
        simes = simes_selection_egenes(X, y)
        p_simes[i + 4500] = simes.simes_p_value()

    print("regime 0 done")

    sig = BH_selection_egenes(p_simes, 0.10)

    K = sig[0]

    E_sel = np.sort(sig[1])

    false_rej = (E_sel[E_sel>4500]).shape[0]

    simes_level = K*bh_level/ngenes

    print("power", (K-false_rej)/4500.)

    print("fdr", false_rej/float(K))

    print("regime 5 selected", E_sel[E_sel<500].shape[0])

    print("regime 4 selected", E_sel[(E_sel >= 500) & (E_sel < 1000)].shape[0])

    print("regime 3 selected", E_sel[(E_sel >= 1000) & (E_sel < 1800)].shape[0])

    print("regime 2 selected", E_sel[(E_sel >= 1800) & (E_sel < 3000)].shape[0])

    print("regime 1 selected", E_sel[(E_sel >= 3000) & (E_sel < 4500)].shape[0])

    print("regime 0 selected", E_sel[(E_sel >= 4500) & (E_sel < 5000)].shape[0])

    return simes_level, E_sel


if __name__ == "__main__":

    test = test_simes()

    E_sel = test[1]

    E_sel_5 = E_sel[E_sel<500]

    sel_5 = E_sel_5.shape[0]

    power_5 = 0.

    for i in range(sel_5):
        np.random.seed(E_sel_5[i])
        sample = instance(n=350, p=1000, s=5, sigma=1., rho=0)
        np.random.seed(E_sel_5[i])
        X, y, beta, nonzero, sigma = sample.generate_response()
        power_5 += random_lasso(X,
                              y,
                              beta,
                              sigma,
                              s=5)

    E_sel_4 = E_sel[(E_sel >= 500) & (E_sel < 1000)]

    sel_4 = E_sel_4.shape[0]

    power_4 = 0.

    for i in range(sel_4):
        np.random.seed(E_sel_4[i])
        sample = instance(n=350, p=1000, s=4, sigma=1., rho=0)
        np.random.seed(E_sel_4[i])
        X, y, beta, nonzero, sigma = sample.generate_response()
        power_4 += random_lasso(X,
                                y,
                                beta,
                                sigma,
                                s=4)


    E_sel_3 = E_sel[(E_sel >= 1000) & (E_sel < 1800)]

    sel_3 = E_sel_3.shape[0]

    power_3 = 0.

    for i in range(sel_3):
        np.random.seed(E_sel_3[i])
        sample = instance(n=350, p=1000, s=3, sigma=1., rho=0)
        np.random.seed(E_sel_3[i])
        X, y, beta, nonzero, sigma = sample.generate_response()
        power_3 += random_lasso(X,
                                y,
                                beta,
                                sigma,
                                s=3)


    E_sel_2 = E_sel[(E_sel >= 1800) & (E_sel < 3000)]

    sel_2 = E_sel_2.shape[0]

    power_2 = 0.

    for i in range(sel_2):
        np.random.seed(E_sel_2[i])
        sample = instance(n=350, p=1000, s=2, sigma=1., rho=0)
        np.random.seed(E_sel_2[i])
        X, y, beta, nonzero, sigma = sample.generate_response()
        power_2 += random_lasso(X,
                                y,
                                beta,
                                sigma,
                                s=2)


    E_sel_1 = E_sel[(E_sel >= 3000) & (E_sel < 4500)]

    sel_1 = E_sel_1.shape[0]

    power_1 = 0.

    for i in range(sel_1):
        np.random.seed(E_sel_1[i])
        sample = instance(n=350, p=1000, s=1, sigma=1., rho=0)
        np.random.seed(E_sel_1[i])
        X, y, beta, nonzero, sigma = sample.generate_response()
        power_1 += random_lasso(X,
                                y,
                                beta,
                                sigma,
                                s=1)

    print("regime 5", power_5 / (5. * sel_5))
    print("regime 4", power_4 / (4. * sel_4))
    print("regime 3", power_3 / (3. * sel_3))
    print("regime 2", power_2 / (2. * sel_2))
    print("regime 1", power_1 / (1. * sel_1))


# def selection_information(simes_level= 0.065):
#
#     X, y, true_beta, nonzero, noise_variance = gaussian_instance(n=350, p=700, s=50, sigma=1, rho=0, snr=5.)
#
#     simes = simes_selection_egenes(X, y)
#
#     return simes.post_BH_selection(simes_level)

#print(selection_information())