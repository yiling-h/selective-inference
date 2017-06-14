import glob
import os, numpy as np, pandas, statsmodels.api as sm

gene_file = r'/Users/snigdhapanigrahi/Results_freq_EQTL/Muscle_Skeletal_mixture4amp0.30/Muscle_Skeletal_chunk001_mtx/Genes.txt'

with open(gene_file) as g:
    content = g.readlines()

content = [x.strip() for x in content]

path = '/Users/snigdhapanigrahi/Results_freq_EQTL/Muscle_Skeletal_mixture4amp0.30/Muscle_Skeletal_chunk001_mtx/'
files = []

for i in os.listdir(path):
    if os.path.isfile(os.path.join(path,i)) and content[120] in i:
        files.append(i)

for each_file in files:
    if each_file.startswith('X'):
        print(each_file)
        X = np.load(os.path.join(path + str(each_file)))
        n, p = X.shape
        print(n,p)
    elif each_file.startswith('y'):
        print(each_file)
    elif each_file.startswith('b'):
        print(each_file)



