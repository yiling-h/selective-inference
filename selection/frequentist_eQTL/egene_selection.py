from __future__ import print_function
from scipy.stats import norm as normal
import numpy as np
import os

def BH_selection_egenes(p_simes, level):

    m = p_simes.shape[0]
    p_sorted = np.sort(p_simes)
    indices = np.arange(m)
    indices_order = np.argsort(p_simes)

    #if np.any(p_sorted - np.true_divide(level * (np.arange(m) + 1.), m) <= np.zeros(m)):

    order_sig = np.max(indices[p_sorted - np.true_divide(level * (np.arange(m) + 1.), m) <= 0])
    E_sel = indices_order[:(order_sig+1)]

    return order_sig+1, E_sel


class simes_selection_egenes():

    def __init__(self,
                 X,
                 y,
                 randomizer= 'gaussian',
                 noise_level = 1.,
                 randomization_scale=1.):

        self.X = X
        self.y = y
        self.n, self.p = self.X.shape
        self.sigma = noise_level
        self.T_stats = self.X.T.dot(self.y) / self.sigma

        if randomizer == 'gaussian':
            perturb = np.random.standard_normal(self.p)
            self.randomized_T_stats = self.T_stats + randomization_scale * perturb
            self.p_val_randomized = np.sort(
                2 * (1. - normal.cdf(np.true_divide(np.abs(self.randomized_T_stats),np.sqrt(2.)))))

            self.indices_order = np.argsort(
                2 * (1. - normal.cdf(np.true_divide(np.abs(self.randomized_T_stats),np.sqrt(2.)))))

        elif randomizer == 'none':
            perturb = np.zeros(self.p)
            self.randomized_T_stats = self.T_stats + randomization_scale * perturb

            self.p_val_randomized = np.sort(
                2 * (1. - normal.cdf(np.true_divide(np.abs(self.randomized_T_stats), np.sqrt(1.)))))

            self.indices_order = np.argsort(
                2 * (1. - normal.cdf(np.true_divide(np.abs(self.randomized_T_stats), np.sqrt(1.)))))


    def simes_p_value(self):

        simes_p_randomized = np.min((self.p / (np.arange(self.p) + 1.)) * self.p_val_randomized)

        i_0 = np.argmin((self.p / (np.arange(self.p) + 1.)) * self.p_val_randomized)

        t_0 = self.indices_order[i_0]

        T_stats_active = self.T_stats[i_0]

        u_1 = ((i_0 + 1.) / self.p) * np.min(
            np.delete((self.p / (np.arange(self.p) + 1.)) * self.p_val_randomized, i_0))

        u_2 = self.p_val_randomized[i_0 + 1]

        return simes_p_randomized, i_0, t_0, np.sign(T_stats_active), u_1, u_2


gene_file = r'/Users/snigdhapanigrahi/Results_freq_EQTL/Muscle_Skeletal_mixture4amp0.30/Muscle_Skeletal_chunk001_mtx/Genes.txt'

with open(gene_file) as g:
    content = g.readlines()

content = [x.strip() for x in content]

path = '/Users/snigdhapanigrahi/Results_freq_EQTL/Muscle_Skeletal_mixture4amp0.30/Muscle_Skeletal_chunk001_mtx/'
files = []

if __name__ == "__main__":

    for i in os.listdir(path):
        if os.path.isfile(os.path.join(path, i)) and content[99] in i:
            files.append(i)

    for each_file in files:
        if each_file.startswith('X'):
            print(each_file)
            X = np.asarray(np.load(os.path.join(path + str(each_file))))
            n, p = X.shape
            X -= X.mean(0)[None, :]
            X /= (X.std(0)[None, :] * np.sqrt(n))
        elif each_file.startswith('y'):
            y = np.asarray(np.load(os.path.join(path + str(each_file))))
            y = y.reshape((y.shape[0],))
            print(each_file)
        elif each_file.startswith('b'):
            beta = np.load(os.path.join(path + str(each_file)))
            print(each_file)

    simes = simes_selection_egenes(X, y, randomizer='gaussian')

    print("p-value", simes.simes_p_value())
