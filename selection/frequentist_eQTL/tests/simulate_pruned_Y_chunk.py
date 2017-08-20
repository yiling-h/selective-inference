import os, numpy as np, pandas, statsmodels.api as sm
import sys
from scipy.stats.stats import pearsonr

class generate_data():

    def __init__(self, X, nsignals, sigma=1.):

         self.sigma = sigma

         self.n, self.p = X.shape
         X -= X.mean(0)[None, :]
         X /= (X.std(0)[None, :] * np.sqrt(self.n))

         self.X = X
         beta_true = np.zeros(self.p)
         signal_indices = -1* np.ones(self.n)

         cl_subsample = np.random.choice(self.p, nsignals, replace=False)

         if nsignals > 0:
             for j in range(nsignals):
                 beta_true[cl_subsample[j]] = 3.
                 signal_indices[j] = cl_subsample[j]

         self.beta = beta_true
         self.signal_indices = signal_indices

    def generate_response(self):

        Y = (self.X.dot(self.beta) + np.random.standard_normal(self.n)) * self.sigma

        return self.X.dot(self.beta), Y, self.beta * self.sigma, self.sigma, self.signal_indices

if __name__ == "__main__":

    inpath = sys.argv[1]
    outpath = sys.argv[2]
    protopath = sys.argv[3]

    gene_file = inpath + "Genes.txt"

    with open(gene_file) as g:
        content = g.readlines()

    content = [x.strip() for x in content]
    niter = len(content)
    sys.stderr.write("length" + str(niter) + "\n")

    #regime = np.random.choice(range(6), niter, replace=True, p=[0.65, 0.17, 0.08, 0.05, 0.03, 0.02])
    #regime = np.random.choice(range(6), niter, replace=True, p=[0.50, 0.24, 0.12, 0.07, 0.05, 0.02])
    #regime = np.random.choice(range(6), niter, replace=True, p=[0.20, 0.36, 0.20, 0.12, 0.07, 0.05])
    #regime = np.random.choice(range(6), niter, replace=True, p=[0.30, 0.34, 0.17, 0.10, 0.06, 0.03])
    regime = np.random.choice(range(6), niter, replace=True, p=[0.40, 0.30, 0.15, 0.08, 0.05, 0.02])
    signals = [0, 1, 2, 3, 5, 10]

    for j in range(niter):

        X = np.load(os.path.join(inpath + "X_" + str(content[j])) + ".npy")
        n, p = X.shape

        prototypes = np.loadtxt(os.path.join(protopath + "protoclust_" + str(content[j])) + ".txt", delimiter='\t')
        representatives = np.unique(prototypes).astype(int)
        X = X[:, representatives]
        nsignals = signals[regime[j]]

        sample = generate_data(X, nsignals)
        true_mean, y, beta, sigma, signal_indices = sample.generate_response()

        output = np.zeros((3, n))
        output[0,:] = y
        output[1,:] = true_mean
        output[2,:] = signal_indices

        outfile = os.path.join(outpath, "y_pruned_simulated_" + str(content[j]) + ".txt")

        np.savetxt(outfile, output)
