library("protoclust")
library("RcppCNPy")


pcluster = function()
{
  args = commandArgs(trailingOnly = TRUE)

  ind = args[3]

  indir = toString(args[1])
  infile = file.path(indir, paste(sep="","Genes.txt"))

  outdir = toString(args[2])

  #strip names of egenes from infile
  conn <- file(infile, open="r")
  lines <- readLines(conn)

  datadir = toString(args[4])
  print(paste("datadir", datadir, indir))

  #get X with the egene_name from directory
  for(i in 1:length(lines))
  {
    X = npyLoad(file.path(indir, paste(sep="","X_",toString(lines[i]), ".npy")))
    p = ncol(X)
    print(paste("p", p))
    height = function(k, hc)
    {
      return(hc$height[p - k])
    }

    Corr_X = cor(X)
    dist_X = 1. - abs(Corr_X)

    hc <- protoclust(dist_X)

    if(p>700){

    k.max = 700
    k.seq = seq(from= 1, to = k.max, by = 10)
    h.seq = sapply(k.seq, height, hc =hc)
    #print(paste("h.seq", h.seq))
    k.ch = k.seq[which(abs(h.seq- 0.50) ==min(abs(h.seq- 0.50)))]
    print(paste("K chosen", k.ch))

    cut <- protocut(hc, k=k.ch)
    pr <- cut$protos[cut$cl]
    #to match indexing in python: starts from 0
    pr <- pr -1

    outfile = file.path(outdir, paste(sep="","protoclust_",toString(lines[i]), ".txt"))
    write.table(pr, outfile, sep="\t", col.names = F, row.names = F)} else if(p>300 & p<=700){

    k.max = 100
    k.seq = seq(from= 1, to = k.max, by = 2)
    h.seq = sapply(k.seq, height, hc =hc)
    #print(paste("h.seq", h.seq))
    k.ch = k.seq[which(abs(h.seq- 0.50) ==min(abs(h.seq- 0.50)))]
    print(paste("K chosen", k.ch))

    cut <- protocut(hc, k=k.ch)
    pr <- cut$protos[cut$cl]

    pr <- pr-1
    outfile = file.path(outdir, paste(sep="","protoclust_",toString(lines[i]), ".txt"))
    write.table(pr, outfile, sep="\t", col.names = F, row.names = F)} else{
    pr <- 1:p
    pr <- pr-1
    outfile = file.path(outdir, paste(sep="","protoclust_",toString(lines[i]), ".txt"))
    write.table(pr, outfile, sep="\t", col.names = F, row.names = F)}
  }
}

pcluster()