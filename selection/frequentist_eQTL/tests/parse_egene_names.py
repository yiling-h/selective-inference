
import os


path = '/Users/snigdhapanigrahi/sim_fwd_bwd_inference_10/names'
list_egenes = []
for i in range(100):

    gene_file = os.path.join(path + str(format(i+1, '03')) + "/eGenes.txt")

    with open(gene_file) as g:
        content = g.readlines()

    content = [x.strip() for x in content]
    for x in content:
        list_egenes.append(x)

outfile='/Users/snigdhapanigrahi/sim_fwd_bwd_inference_10/eGenes_10.txt'
with open(outfile, 'w') as fo:
    for x in list_egenes:
        fo.write(str(x) + '\n')

path='/Users/snigdhapanigrahi/sim_fwd_bwd_inference_10/eGenes_10.txt'
with open(path) as g:
    content = g.readlines()
content = [x.strip() for x in content]
print("length egenes", len(content))