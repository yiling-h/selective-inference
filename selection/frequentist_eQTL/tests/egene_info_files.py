from __future__ import print_function
import glob
import numpy as np, os, sys

# if __name__ == "__main__":
#
#     path = sys.argv[1]
#     result = sys.argv[5]
#     infile = os.path.join(path, "egenes_" + str(result) + ".txt")
#     with open(infile) as g:
#         content = g.readlines()
#
#     content = [x.strip() for x in content]
#     sys.stderr.write("length" + str(len(content)) + "\n")
#
#     outdir = sys.argv[6]
#     X_path = sys.argv[2]
#     Y_path = sys.argv[3]
#     bon_path = sys.argv[4]
#
#     for i in range(len(content)):
#
#         X = np.load(os.path.join(X_path + "X_" + str(content[i])) + ".npy")
#         out_X = os.path.join(outdir + "X_" + str(content[i]) + ".npy")
#         np.save(out_X, X)
#
#         y = np.loadtxt(os.path.join(Y_path + "y_pruned_simulated_" + str(content[i])) + ".txt")
#         out_Y = os.path.join(outdir + "y_pruned_simulated_" + str(content[i]) + ".txt")
#         np.savetxt(out_Y, y)
#
#         prototypes = np.loadtxt(os.path.join(Y_path + "protoclust_" + str(content[i])) + ".txt", delimiter='\t')
#         outfile = os.path.join(outdir + "protoclust_" + str(content[i]) + ".txt")
#         np.savetxt(outfile, prototypes, delimiter='\t')
#
#         simes_output = np.loadtxt(os.path.join(bon_path + "simes_" + str(content[i])) + ".txt")
#         simesfile = os.path.join(outdir + "simes_" + str(content[i]) + ".txt")
#         np.savetxt(outfile, simesfile)
#
#         sys.stderr.write("egene completed" + str(i) + "\n")



path='/Users/snigdhapanigrahi/bon_output_liver/randomized_egene_names_05/'
allFiles = glob.glob(path + "/*.txt")
list_egenes = []
for infile in sorted(allFiles):
    with open(infile) as g:
        content = g.readlines()

    content = [x.strip() for x in content]
    for x in content:
        list_egenes.append(x)
    #print("list so far", list_egenes)

outfile='/Users/snigdhapanigrahi/bon_output_liver/eGenes_05.txt'
with open(outfile, 'w') as fo:
    for x in list_egenes:
        fo.write(str(x) + '\n')


path='/Users/snigdhapanigrahi/bon_output_liver/eGenes_05.txt'
with open(path) as g:
    content = g.readlines()
content = [x.strip() for x in content]
print("length egenes", len(content))


#E = np.load('/Users/snigdhapanigrahi/fwd_bwd_inference/s_ENSG00000160072.15.npy')
#E = E.reshape((E.shape[0],))
#print(E, E.shape)




