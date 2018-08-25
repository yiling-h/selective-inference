import os, numpy as np, pandas, statsmodels.api as sm
import time
import regreg.api as rr

import matplotlib.pyplot as plt
from selection.randomized.api import randomization
from selection.reduced_optimization.par_carved_reduced import selection_probability_carved, sel_inf_carved
from selection.randomized.M_estimator import M_estimator_split
from selection.randomized.glm import bootstrap_cov

class M_estimator_approx_carved(M_estimator_split):

    def __init__(self, loss, epsilon, subsample_size, penalty, estimation):

        M_estimator_split.__init__(self,loss, epsilon, subsample_size, penalty, solve_args={'min_its':50, 'tol':1.e-10})
        self.estimation = estimation

    def solve_approx(self):

        self.solve()

        self.nactive = self._overall.sum()
        X, _ = self.loss.data
        n, p = X.shape
        self.p = p
        self.target_observed = self.observed_score_state[:self.nactive]

        self.feasible_point = np.concatenate([self.observed_score_state, np.fabs(self.observed_opt_state[:self.nactive]),
                                              self.observed_opt_state[self.nactive:]], axis = 0)

        (_opt_linear_term, _opt_affine_term) = self.opt_transform
        self._opt_linear_term = np.concatenate(
            (_opt_linear_term[self._overall, :], _opt_linear_term[~self._overall, :]), 0)

        self._opt_affine_term = np.concatenate((_opt_affine_term[self._overall], _opt_affine_term[~self._overall]), 0)
        self.opt_transform = (self._opt_linear_term, self._opt_affine_term)

        (_score_linear_term, _) = self.score_transform
        self._score_linear_term = np.concatenate(
            (_score_linear_term[self._overall, :], _score_linear_term[~self._overall, :]), 0)

        self.score_transform = (self._score_linear_term, np.zeros(self._score_linear_term.shape[0]))

        lagrange = []
        for key, value in self.penalty.weights.iteritems():
            lagrange.append(value)
        lagrange = np.asarray(lagrange)

        self.inactive_lagrange = lagrange[~self._overall]

        self.bootstrap_score, self.randomization_cov = self.setup_sampler()

        if self.estimation == 'parametric':
            score_cov = np.zeros((p,p))
            inv_X_active = np.linalg.inv(X[:, self._overall].T.dot(X[:, self._overall]))
            projection_X_active = X[:,self._overall].dot(np.linalg.inv(X[:, self._overall].T.dot(X[:, self._overall]))).dot(X[:,self._overall].T)
            score_cov[:self.nactive, :self.nactive] = inv_X_active
            score_cov[self.nactive:, self.nactive:] = X[:,~self._overall].T.dot(np.identity(n)- projection_X_active).dot(X[:,~self._overall])

        elif self.estimation == 'bootstrap':
            score_cov = bootstrap_cov(lambda: np.random.choice(n, size=(n,), replace=True), self.bootstrap_score)

        self.score_cov = score_cov
        self.score_cov_inv = np.linalg.inv(self.score_cov)

if not os.path.exists("NRTI_DATA.txt"):
    NRTI = pandas.read_table("http://hivdb.stanford.edu/pages/published_analysis/genophenoPNAS2006/DATA/NRTI_DATA.txt", na_values="NA")
else:
    NRTI = pandas.read_table("NRTI_DATA.txt")

NRTI_specific = []
NRTI_muts = []
mixtures = np.zeros(NRTI.shape[0])
for i in range(1,241):
    d = NRTI['P%d' % i]
    for mut in np.unique(d):
        if mut not in ['-','.'] and len(mut) == 1:
            test = np.equal(d, mut)
            if test.sum() > 10:
                NRTI_specific.append(np.array(np.equal(d, mut)))
                NRTI_muts.append("P%d%s" % (i,mut))

NRTI_specific = NRTI.from_records(np.array(NRTI_specific).T, columns=NRTI_muts)
print("here")

# Next, standardize the data, keeping only those where Y is not missing

X_NRTI = np.array(NRTI_specific, np.float)
Y = NRTI['3TC'] # shorthand
keep = ~np.isnan(Y).astype(np.bool)
X_NRTI = X_NRTI[np.nonzero(keep)]; Y=Y[keep]
Y = np.array(np.log(Y), np.float); Y -= Y.mean()
X_NRTI -= X_NRTI.mean(0)[None, :]; X_NRTI /= X_NRTI.std(0)[None,:]
X = X_NRTI # shorthand
n, p = X.shape
X /= np.sqrt(n)

ols_fit = sm.OLS(Y, X).fit()
sigma_3TC = np.linalg.norm(ols_fit.resid) / np.sqrt(n-p-1)
OLS_3TC = ols_fit.params
lam = 1. * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 2000)))).max(0)) * sigma_3TC

n, p = X.shape

loss = rr.glm.gaussian(X, Y)
epsilon = 1. / np.sqrt(n)

W = np.ones(p) * lam
penalty = rr.group_lasso(np.arange(p), weights=dict(zip(np.arange(p), W)), lagrange=1.)

total_size = loss.saturated_loss.shape[0]
subsample_size = int(0.50 * total_size)
inference_size = total_size - subsample_size
M_est = M_estimator_approx_carved(loss, epsilon, subsample_size, penalty, 'parametric')

sel_indx = M_est.sel_indx
X_inf = X[~sel_indx, :]
y_inf = Y[~sel_indx]
M_est.solve_approx()

active = M_est._overall
active_set = [t for t in range(p) if active[t]]
active_set_0 = [NRTI_muts[i] for i in range(p) if active[i]]
nactive = M_est.nactive
active_bool = np.zeros(nactive, np.bool)
prior_variance = 1000.
noise_variance = sigma_3TC ** 2
projection_active = X[:, active].dot(np.linalg.inv(X[:, active].T.dot(X[:, active])))
projection_active_split = X_inf[:, active].dot(np.linalg.inv(X_inf[:, active].T.dot(X_inf[:, active])))

M_1 = prior_variance * (X.dot(X.T)) + noise_variance * np.identity(n)
M_2 = prior_variance * ((X.dot(X.T)).dot(projection_active))
M_3 = prior_variance * (projection_active.T.dot(X.dot(X.T)).dot(projection_active))
post_mean = M_2.T.dot(np.linalg.inv(M_1)).dot(Y)
post_var = M_3 - M_2.T.dot(np.linalg.inv(M_1)).dot(M_2)
unadjusted_intervals = np.vstack([post_mean - 1.65 * (np.sqrt(post_var.diagonal())),
                                  post_mean + 1.65 * (np.sqrt(post_var.diagonal()))])
unadjusted_intervals = np.vstack([post_mean, unadjusted_intervals])

M_1_split = prior_variance * (X_inf.dot(X_inf.T)) + noise_variance * np.identity(int(inference_size))
M_2_split = prior_variance * ((X_inf.dot(X_inf.T)).dot(projection_active_split))
M_3_split = prior_variance * (projection_active_split.T.dot(X_inf.dot(X_inf.T)).dot(projection_active_split))
post_mean_split = M_2_split.T.dot(np.linalg.inv(M_1_split)).dot(y_inf)
post_var_split = M_3_split - M_2_split.T.dot(np.linalg.inv(M_1_split)).dot(M_2_split)
adjusted_intervals_split = np.vstack([post_mean_split - 1.65 * (np.sqrt(post_var_split.diagonal())),
                                      post_mean_split + 1.65 * (np.sqrt(post_var_split.diagonal()))])
adjusted_intervals_split = np.vstack([post_mean_split, adjusted_intervals_split])

grad_lasso = sel_inf_carved(M_est, prior_variance)
samples = grad_lasso.posterior_samples()
adjusted_intervals = np.vstack([np.percentile(samples, 5, axis=0), np.percentile(samples, 95, axis=0)])
selective_mean = np.mean(samples, axis=0)
adjusted_intervals = np.vstack([selective_mean, adjusted_intervals])

intervals = np.vstack([unadjusted_intervals, adjusted_intervals_split, adjusted_intervals])
print("active set and intervals shape", nactive, intervals.shape)
print("intervals", unadjusted_intervals, adjusted_intervals_split, adjusted_intervals)
print("average lengths", np.mean(intervals[2,:]-intervals[1,:]), np.mean(intervals[5,:]-intervals[4,:]), np.mean(intervals[8,:]-intervals[7,:]))

ind = np.zeros(len(active_set), np.bool)
index = active_set_0.index('P184V')
ind[index] = 1

active_set_0.pop(index)
active_set.pop(index)

intervals = intervals[:, ~ind]

un_mean = intervals[0,:]
un_lower_error = list(un_mean-intervals[1,:])
un_upper_error = list(intervals[2,:]-un_mean)
unStd = [un_lower_error, un_upper_error]
ad_mean_split = intervals[3,:]
ad_lower_error_split = list(ad_mean_split-intervals[4,:])
ad_upper_error_split = list(intervals[5,:]- ad_mean_split)
adStd_split = [ad_lower_error_split, ad_upper_error_split]
ad_mean = intervals[6,:]
ad_lower_error = list(ad_mean -intervals[7,:])
ad_upper_error = list(intervals[8,:]- ad_mean_split)
adStd = [ad_lower_error, ad_upper_error]

N = len(un_mean)               # number of data entries
ind = 2*np.arange(N)              # the x locations for the groups
width = 0.35                    # bar width
print('here')

fig, ax = plt.subplots()

rects1 = ax.bar(ind, un_mean,                  # data
                width,                          # bar width
                color='red',        # bar colour
                yerr=unStd,  # data for error bars
                error_kw={'ecolor':'maroon',    # error-bars colour
                          'linewidth':2})       # error-bar width

rects2 = ax.bar(ind + width, ad_mean_split,
                width,
                color='royalblue',
                yerr=adStd_split,
                error_kw={'ecolor':'darkblue',
                          'linewidth':2})

rects3 = ax.bar(ind + (2*width), ad_mean,
                width,
                color='seagreen',
                yerr=adStd,
                error_kw={'ecolor':'darkgreen',
                          'linewidth':2})

axes = plt.gca()
axes.set_ylim([-5, 20])             # y-axis bounds

ax.set_ylabel(' ')
ax.set_title('selected variables'.format(active_set))
ax.set_xticks(ind + 1.5* width)

ax.set_xticklabels(active_set_0, rotation=90)

ax.legend((rects1[0], rects2[0], rects3[0]), ('Unadjusted', 'Split', 'Carved'), loc='upper right')

print('here')

plt.savefig('/Users/snigdhapanigrahi/Documents/Research/hiv_selected_1.pdf',
             bbox_inches='tight')