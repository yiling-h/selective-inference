import glob
import os
import numpy as np

path = r'/Users/snigdhapanigrahi/inference_liver/eGenes.txt'
fwdpath = r'/Users/snigdhapanigrahi/fwd_bwd_inference_10/eGenes_10.txt'
outpath = r'/Users/snigdhapanigrahi/inference_liver/'

#path = r'/Users/snigdhapanigrahi/nonrandomized_bon_output_liver/eGenes.txt'
#fwdpath = r'/Users/snigdhapanigrahi/nonrandomized_bon_output_liver.05/eGenes.txt'
with open(path) as g:
    eGenes = g.readlines()
eGenes_selinf = [x.strip() for x in eGenes]
print("length egenes", len(eGenes_selinf))

with open(fwdpath) as h:
    eGenes = h.readlines()
eGenes_fwd = [x.strip() for x in eGenes]
print("length egenes", len(eGenes_fwd))

print("intersection", len(list(set(eGenes_selinf).intersection(set(eGenes_fwd)))))

# inpath=r'/Users/snigdhapanigrahi/inference_liver/Lee_inf/'
# outpath=r'/Users/snigdhapanigrahi/inference_liver/Lee_inf_0.05/'
#
# for i in range(len(eGenes_fwd)):
#     data = np.loadtxt(os.path.join(inpath, "Leeoutput_" + str(eGenes_fwd[i]) + ".txt"))
#     outfile = os.path.join(outpath + "Leeoutput_" + str(eGenes_fwd[i]) + ".txt")
#     np.savetxt(outfile, data)


# outfile = os.path.join(outpath + "common_egenes_10.txt")
# common_egenes = list(set(eGenes_selinf).intersection(set(eGenes_fwd)))
# with open(outfile, 'w') as fo:
#     for gene_name in common_egenes:
#        fo.write(str(gene_name) + '\n')

# path = r'/Users/snigdhapanigrahi/inference_liver/inference_05'
# outpath =  r'/Users/snigdhapanigrahi/inference_liver/'
# outfile = os.path.join(outpath + "egenes_completed_05.txt")
#
# allFiles = glob.glob(path + "/*.txt")
# gene_names = []
# for file_ in allFiles:
#     file_name = os.path.basename(file_)
#     file_name =  os.path.splitext(file_name)[0]
#
#     index_of_underscore = file_name.index('_')
#     gene_name = file_name[index_of_underscore+1:]
#     gene_names.append(gene_name)
#
# with open(outfile, 'w') as fo:
#     for gene_name in gene_names:
#         fo.write(str(gene_name) + '\n')
#
# path='/Users/snigdhapanigrahi/inference_liver/eGenes_05.txt'
# with open(path) as g:
#     eGenes = g.readlines()
# eGenes = [x.strip() for x in eGenes]
# print("length egenes", len(eGenes))
#
# path='/Users/snigdhapanigrahi/inference_liver/egenes_completed_05.txt'
# with open(path) as g:
#     cGenes = g.readlines()
# cGenes = [x.strip() for x in cGenes]
# print("length completed egenes", len(cGenes))
#
# print("difference", len(list(set(eGenes) - set(cGenes))))
#
# # with open(outfile, 'w') as fo:
# #     for gene_name in gene_names:
# #         fo.write(str(gene_name) + '\n')
#
# outfile = os.path.join(outpath + "extreme_egenes_05.txt")
# extreme_egenes = list(set(eGenes) - set(cGenes))
# with open(outfile, 'w') as fo:
#     for gene_name in extreme_egenes:
#        fo.write(str(gene_name) + '\n')