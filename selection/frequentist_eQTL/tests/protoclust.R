library("protoclust")
library("RcppCNPy")


pcluster = function()
{
  args = commandArgs(trailingOnly = TRUE)
  
  ind = args[3]
  
  indir = toString(args[1])
  infile = file.path(indir, paste(sep="","egenes_", toString(ind), ".txt"))
  
  outdir = toString(args[2])
  
  #strip names of egenes from infile
  conn <- file(infile, open="r")
  lines <- readLines(conn)
  
  datadir = toString(args[4])

  #get X with the egene_name from directory
  for(i in 1:length(lines))
  { 
    print(i)
    X = npyLoad(file.path(datadir, paste(sep="","X_",toString(lines[i]), ".npy")))
    p = ncol(X)
    height = function(k, hc)
    {
      return(hc$height[p - k])
    }
    
    Corr_X = cor(X)
    dist_X = 1. - abs(Corr_X)
    
    hc <- protoclust(dist_X)
    
    k.seq = seq(from= 100, to = 500, by = 10)
    h.seq = sapply(k.seq, height, hc =hc)
    k.ch = k.seq[which(abs(h.seq- 0.50) ==min(abs(h.seq- 0.50)))]
    
    cut <- protocut(hc, k=k.ch)
    pr <- cut$protos[cut$cl]
    #to match indexing in python: starts from 0
    pr <- pr -1
    
    outfile = file.path(outdir, paste(sep="","protoclust_",toString(lines[i]), ".txt"))
    write.table(pr, outfile, sep="\t", col.names = F, row.names = F)
  }
}