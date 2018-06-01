from __future__ import print_function

import sys
import os

import numpy as np
from selection.bayesian.initial_soln import selection, instance

from selection.reduced_optimization.lasso_reduced import nonnegative_softmax_scaled, neg_log_cube_probability, selection_probability_lasso, \
    sel_prob_gradient_map_lasso, selective_inf_lasso
from selection.tests.instance import logistic_instance, gaussian_instance
#from selection.reduced_optimization.generative_model import generate_data

class generate_data():

    def __init__(self, n, p, sigma=1., rho=0., scale =True, center=True):
         (self.n, self.p, self.sigma, self.rho) = (n, p, sigma, rho)

         self.X = (np.sqrt(1 - self.rho) * np.random.standard_normal((self.n, self.p)) +
                   np.sqrt(self.rho) * np.random.standard_normal(self.n)[:, None])
         if center:
             self.X -= self.X.mean(0)[None, :]
         if scale:
             self.X /= (self.X.std(0)[None, :] * np.sqrt(self.n))

         beta_true = np.zeros(p)
         u = np.random.uniform(0.,1.,p)
         for i in range(p):
             if u[i]<= 0.95:
                 beta_true[i] = np.random.laplace(loc=0., scale= 0.05)
             else:
                 beta_true[i] = np.random.laplace(loc=0., scale= 0.5)

         self.beta = beta_true
         #print(max(abs(self.beta)), min(abs(self.beta)))
         #print(self.beta[np.where(self.beta>3.)].shape)

    def generate_response(self):

        Y = (self.X.dot(self.beta) + np.random.standard_normal(self.n)) * self.sigma

        return self.X, Y, self.beta * self.sigma, self.sigma

def randomized_lasso_trial(X,
                           y,
                           beta,
                           sigma):

    from selection.api import randomization

    n, p = X.shape

    random_Z = np.random.standard_normal(p)
    sel = selection(X, y, random_Z)
    lam, epsilon, active, betaE, cube, initial_soln = sel

    if sel is not None:

        lagrange = lam * np.ones(p)
        active_sign = np.sign(betaE)
        nactive = active.sum()
        print("number of selected variables by Lasso", nactive)

        feasible_point = np.fabs(betaE)

        noise_variance = sigma ** 2

        randomizer = randomization.isotropic_gaussian((p,), 1.)

        generative_X = X[:, active]
        prior_variance = 1000.

        grad_map = sel_prob_gradient_map_lasso(X,
                                               feasible_point,
                                               active,
                                               active_sign,
                                               lagrange,
                                               generative_X,
                                               noise_variance,
                                               randomizer,
                                               epsilon)

        inf = selective_inf_lasso(y, grad_map, prior_variance)

        samples = inf.posterior_samples()

        adjusted_intervals = np.vstack([np.percentile(samples, 5, axis=0), np.percentile(samples, 95, axis=0)])

        selective_mean = np.mean(samples, axis=0)

        projection_active = X[:, active].dot(np.linalg.inv(X[:, active].T.dot(X[:, active])))
        M_1 = prior_variance * (X.dot(X.T)) + noise_variance * np.identity(n)
        M_2 = prior_variance * ((X.dot(X.T)).dot(projection_active))
        M_3 = prior_variance * (projection_active.T.dot(X.dot(X.T)).dot(projection_active))
        post_mean = M_2.T.dot(np.linalg.inv(M_1)).dot(y)

        print("observed data", post_mean)

        post_var = M_3 - M_2.T.dot(np.linalg.inv(M_1)).dot(M_2)

        unadjusted_intervals = np.vstack([post_mean - 1.65 * (np.sqrt(post_var.diagonal())),
                                          post_mean + 1.65 * (np.sqrt(post_var.diagonal()))])

        coverage_ad = np.zeros(nactive)
        coverage_unad = np.zeros(nactive)
        ad_length = np.zeros(nactive)
        unad_length = np.zeros(nactive)

        true_val = projection_active.T.dot(X.dot(beta))

        for l in range(nactive):
            if (adjusted_intervals[0, l] <= true_val[l]) and (true_val[l] <= adjusted_intervals[1, l]):
                coverage_ad[l] += 1
            ad_length[l] = adjusted_intervals[1, l] - adjusted_intervals[0, l]
            if (unadjusted_intervals[0, l] <= true_val[l]) and (true_val[l] <= unadjusted_intervals[1, l]):
                coverage_unad[l] += 1
            unad_length[l] = unadjusted_intervals[1, l] - unadjusted_intervals[0, l]


        sel_cov = coverage_ad.sum() / nactive
        naive_cov = coverage_unad.sum() / nactive
        ad_len = ad_length.sum() / nactive
        unad_len = unad_length.sum() / nactive
        bayes_risk_ad = np.power(selective_mean-true_val, 2.).sum()/nactive
        bayes_risk_unad = np.power(post_mean-true_val, 2.).sum()/nactive

        return np.vstack([sel_cov, naive_cov, ad_len, unad_len, bayes_risk_ad, bayes_risk_unad])

    else:
        return None


# if __name__ == "__main__":
#     ### set parameters
#     n = 200
#     p = 1000
#     s = 0
#     snr = 5.
#
#     ### GENERATE X
#     np.random.seed(0)  # ensures same X
#
#     sample = generate_data(n, p)
#
#     niter = 5
#
#     ad_cov = 0.
#     unad_cov = 0.
#     ad_len = 0.
#     unad_len = 0.
#
#     for i in range(niter):
#
#          ### GENERATE Y BASED ON SEED
#          np.random.seed(i+1)  # ensures different y
#          X, y, beta, sigma = sample.generate_response()
#
#          print("true value of underlying parameter", beta)
#
#          ### RUN LASSO AND TEST
#          lasso = randomized_lasso_trial(X,
#                                         y,
#                                         beta,
#                                         sigma)
#
#          if lasso is not None:
#              ad_cov += lasso[0,0]
#              unad_cov += lasso[1,0]
#              ad_len += lasso[2, 0]
#              unad_len += lasso[3, 0]
#              print("\n")
#              print("iteration completed", i)
#              print("\n")
#              print("adjusted and unadjusted coverage", ad_cov, unad_cov)
#              print("adjusted and unadjusted lengths", ad_len, unad_len)


# if __name__ == "__main__":
# # read from command line
#     seedn=int(sys.argv[1])
#     outdir=sys.argv[2]
#
#     outfile = os.path.join(outdir, "list_result_" + str(seedn) + ".txt")
#
# ### set parameters
#     n = 200
#     p = 1000
#     s = 0
#     snr = 5.
#
# ### GENERATE X
#     np.random.seed(0)  # ensures same X
#
#     sample = generate_data(n, p)
#
# ### GENERATE Y BASED ON SEED
#     np.random.seed(seedn) # ensures different y
#     X, y, beta, sigma = sample.generate_response()
#
#     lasso = randomized_lasso_trial(X,
#                                    y,
#                                    beta,
#                                    sigma)
#
#     np.savetxt(outfile, lasso)

from rpy2 import robjects
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

def glmnet_lasso(X, y, lambda_val):
    robjects.r('''
                library('glmnet')
                glmnet_LASSO = function(X,y,lambda){
                y = as.matrix(y)
                X = as.matrix(X)
                lam = as.matrix(lambda)[1,1]
                n = nrow(X)
                fit = glmnet(X, y, standardize=FALSE, intercept=FALSE, thresh=1.e-10)
                estimate = coef(fit, s=lam, exact=TRUE, x=X, y=y)[-1]
                return(list(estimate = estimate))
                }''')

    lambda_R = robjects.globalenv['glmnet_LASSO']
    n, p = X.shape
    r_X = robjects.r.matrix(X, nrow=n, ncol=p)
    r_y = robjects.r.matrix(y, nrow=n, ncol=1)
    r_lam = robjects.r.matrix(lambda_val, nrow=1, ncol=1)
    estimate = np.array(lambda_R(r_X, r_y, r_lam).rx2('estimate'))
    return estimate

def selInf_R(X, y, beta, lam, sigma, Type, alpha=0.1):
    robjects.r('''
               library("selectiveInference")
               selInf = function(X, y, beta, lam, sigma, Type, alpha= 0.1){
               y = as.matrix(y)
               X = as.matrix(X)
               beta = as.matrix(beta)
               lam = as.matrix(lam)[1,1]
               sigma = as.matrix(sigma)[1,1]
               Type = as.matrix(Type)[1,1]
               if(Type == 1){
                   type = "full"} else{
                   type = "partial"}
               inf = fixedLassoInf(x = X, y = y, beta = beta, lambda=lam, family = "gaussian",
                                   intercept=FALSE, sigma=sigma, alpha=alpha, type=type)
               return(list(ci = inf$ci, pvalue = inf$pv))}
               ''')

    inf_R = robjects.globalenv['selInf']
    n, p = X.shape
    r_X = robjects.r.matrix(X, nrow=n, ncol=p)
    r_y = robjects.r.matrix(y, nrow=n, ncol=1)
    r_beta = robjects.r.matrix(beta, nrow=p, ncol=1)
    r_lam = robjects.r.matrix(lam, nrow=1, ncol=1)
    r_sigma = robjects.r.matrix(sigma, nrow=1, ncol=1)
    r_Type = robjects.r.matrix(Type, nrow=1, ncol=1)
    output = inf_R(r_X, r_y, r_beta, r_lam, r_sigma, r_Type)
    ci = np.array(output.rx2('ci'))
    pvalue = np.array(output.rx2('pvalue'))
    return ci, pvalue

if __name__ == "__main__":
    ### set parameters
    n = 200
    p = 1000
    s = 0
    snr = 5.

    ### GENERATE X
    np.random.seed(0)  # ensures same X

    sample = generate_data(n, p)

    niter = 50

    ad_cov = 0.
    unad_cov = 0.
    ad_len = 0.
    unad_len = 0.
    prop_infty = 0.
    nosel = 0.

    for i in range(niter):

         while True:
             ### GENERATE Y BASED ON SEED
             #np.random.seed(i + 1)  # ensures different y
             #X, y, beta, sigma = sample.generate_response()
             X, y, beta, nonzero, sigma = gaussian_instance(n=n, p=p, s=s, sigma=1., rho=0, snr=snr)

             ### RUN GLMNET LASSO AND CALL SEL INFERENCE FROM R PACKAGE
             lam = 1. * sigma * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 10000)))).max(0))
             glm_LASSO = glmnet_lasso(X, y, lam / float(n))
             active_LASSO = (glm_LASSO != 0)
             nactive_LASSO = active_LASSO.sum()
             print("active variables sel by LASSO ", nactive_LASSO)

             if nactive_LASSO > 0:
                 Lee_intervals, Lee_pval = selInf_R(X, y, glm_LASSO, lam, sigma, Type=0, alpha=0.1)
                 beta_target = np.linalg.pinv(X[:, active_LASSO]).dot(X.dot(beta))
                 if (Lee_pval.shape[0] == beta_target.shape[0]):
                     Lee_coverage = np.mean((beta_target > Lee_intervals[:, 0]) * (beta_target < Lee_intervals[:, 1]))
                     inf_entries = np.isinf(Lee_intervals[:, 1] - Lee_intervals[:, 0])
                     if inf_entries.sum() == nactive_LASSO:
                         length_Lee = 0.
                     else:
                         length_Lee = np.mean((Lee_intervals[:, 1] - Lee_intervals[:, 0])[~inf_entries])

                     ad_cov += Lee_coverage
                     ad_len += length_Lee
                     prop_infty += inf_entries.sum() / float(nactive_LASSO)
                     print("\n")
                     print("iteration completed", i)
                     print("\n")
                     print("adjusted coverage", ad_cov)
                     print("adjusted  lengths", ad_len, prop_infty)
                     break

    print("empty draws so far ", nosel)
    print("adjusted coverage so far", ad_cov/(50.-nosel))
    print("adjusted lengths so far", ad_len / (50. - nosel))
    print("prop of intervals that are infty ", prop_infty/(50. - nosel))

