import numpy as np
import glob

inf_path =r'/Users/snigdhapanigrahi/selective-inference/selection/frequentist_eQTL/simulation_prototype/inference/'

allFiles = glob.glob(inf_path + "/*.txt")
metrics = np.zeros(11)
count = 0
for file_ in allFiles:
    df = np.loadtxt(file_)
    gene = file_.strip().split('/')[-1].split(".txt")[0].split("_")[1]
    print("gene", df.shape, gene, np.mean(df, axis=0), df.ndim)

    if df.ndim > 1:
        metrics += np.mean(df, axis=0)
    elif df.ndim == 1:
        metrics += df


    count +=1

print(count, metrics/float(count))

