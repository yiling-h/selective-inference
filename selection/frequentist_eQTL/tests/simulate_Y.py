import os, numpy as np, pandas, statsmodels.api as sm
import sys
from scipy.stats.stats import pearsonr

class generate_data():

    def __init__(self, X, nsignals, candidate, sigma=1.):

         self.sigma = sigma

         self.n, self.p = X.shape
         X -= X.mean(0)[None, :]
         X /= (X.std(0)[None, :] * np.sqrt(self.n))

         self.X = X
         beta_true = np.zeros(self.p)

         cl_subsample = np.random.choice(len(candidate), nsignals, replace=False)
         print("sampled cluster", cl_subsample)

         if nsignals > 0:
             for j in range(nsignals):
                 #print("sampled cluster", cl_subsample[j])
                 signal_subsample = np.random.choice(candidate[cl_subsample[j]].shape[0], 1 , replace=False)
                 signal_indx = candidate[cl_subsample[j]][signal_subsample]
                 #print("chosen index", signal_indx)
                 beta_true[signal_indx] = 3.

         self.beta = beta_true

         print("true beta", self.beta[self.beta>0])

    def generate_response(self):

        Y = (self.X.dot(self.beta) + np.random.standard_normal(self.n)) * self.sigma

        return self.X.dot(self.beta), Y, self.beta * self.sigma, self.sigma

if __name__ == "__main__":

    path = '/Users/snigdhapanigrahi/Test_bon_egenes/Egene_data/'

    gene = str("ENSG00000215915.5")
    X = np.load(os.path.join(path + "X_" + gene) + ".npy")

    n, p = X.shape
    dist_X = 1. - np.corrcoef(X.T)

    prototypes = np.loadtxt(
        os.path.join("/Users/snigdhapanigrahi/Test_bon_egenes/Egene_data/protoclust_" + gene) + ".txt",
        delimiter='\t')
    prototypes = prototypes.astype(int)

    representatives = np.unique(prototypes).astype(int)

    candidate = []

    for i in range(representatives.shape[0]):
        cl_indices = prototypes[prototypes == representatives[i]]
        corr_indices = np.argsort(dist_X[representatives[i],:][cl_indices])

        if cl_indices.shape[0] == 1:
            candidate.append(np.array([representatives[i]]))
        elif cl_indices.shape[0] == 2:
            candidate.append(corr_indices[:2])
        elif cl_indices.shape[0] == 3:
            candidate.append(corr_indices[:3])
        elif cl_indices.shape[0] == 4:
            candidate.append(corr_indices[:4])
        elif cl_indices.shape[0] > 4:
            candidate.append(corr_indices[:5])

    signals = [0,1,2,3,5,10]
    decide = np.random.choice(6, 1, replace=False, p=[0.65, 0.17, 0.08, 0.05, 0.03, 0.02])
    print("no of signals", decide)

    np.random.seed(0)
    sample = generate_data(X, signals[decide], candidate)
    true_mean, y, beta, sigma = sample.generate_response()









