import glob
import os, numpy as np, pandas, statsmodels.api as sm

# path='/Users/snigdhapanigrahi/simes_output_Liver/sigma_est_output/'
# outdir = '/Users/snigdhapanigrahi/simes_output_Liver/sigma_est_output/combined'
#
# for i in range(100):
#
#     i = i + 1
#
#     list = []
#     list.append(np.loadtxt(os.path.join(path, "1_simes_output_sigma_estimated_"+ str(format(i, '03')) + ".txt")))
#     list.append(np.loadtxt(os.path.join(path, "2_simes_output_sigma_estimated_"+ str(format(i, '03')) + ".txt")))
#
#     file = np.vstack(list)
#
#     print("file shape", file.shape)
#     outfile = os.path.join(outdir, "simes_output_sigma_estimated_" + str(format(i, '03')) + ".txt")
#     np.savetxt(outfile, file)

path = '/Users/snigdhapanigrahi/Test_simes'

allFiles = glob.glob(path + "/*.txt")
list_ = []
for file_ in allFiles:
    df = np.loadtxt(file_)
    list_.append(df)

simes_output = np.vstack(list_)
print("p", simes_output[:,0])
print("simes output", simes_output[:,1])
