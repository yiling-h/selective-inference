import glob
import os

path = r'/Users/snigdhapanigrahi/sim_inference_liver/inference'
outpath =  r'/Users/snigdhapanigrahi/sim_inference_liver/'
outfile = os.path.join(outpath + "egenes_completed.txt")

allFiles = glob.glob(path + "/*.txt")
gene_names = []
for file_ in allFiles:
    file_name = os.path.basename(file_)
    file_name =  os.path.splitext(file_name)[0]

    index_of_underscore = file_name.index('_')
    gene_name = file_name[index_of_underscore+1:]
    gene_names.append(gene_name)

with open(outfile, 'w') as fo:
    for gene_name in gene_names:
        fo.write(str(gene_name) + '\n')

path='/Users/snigdhapanigrahi/sim_inference_liver/eGenes.txt'
with open(path) as g:
    eGenes = g.readlines()
eGenes = [x.strip() for x in eGenes]
eGenes = eGenes[:200]
print("length egenes", len(eGenes))

path='/Users/snigdhapanigrahi/sim_inference_liver/egenes_completed.txt'
with open(path) as g:
    cGenes = g.readlines()
cGenes = [x.strip() for x in cGenes]
print("length completed egenes", len(cGenes))

print("difference", list(set(eGenes) - set(cGenes)))
with open(outfile, 'w') as fo:
    for gene_name in gene_names:
        fo.write(str(gene_name) + '\n')

outfile = os.path.join(outpath + "extreme_egenes.txt")
extreme_egenes = list(set(eGenes) - set(cGenes))
with open(outfile, 'w') as fo:
    for gene_name in extreme_egenes:
        fo.write(str(gene_name) + '\n')