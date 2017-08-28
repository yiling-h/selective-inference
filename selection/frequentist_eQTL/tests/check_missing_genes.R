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

  #get X with the egene_name from directory
  count = 0
  for(i in 1:length(lines))
  {
    outfile = file.path(outdir, paste(sep="","protoclust_",toString(lines[i]), ".txt"))
    if(!file.exists(outfile)){
    # print(paste("Missing gene", toString(lines[i])))
    count= count + 1}
  }
  if(count>0){
  print(paste("missing datadir", indir))
  print(paste("total missing genes", count))} 
}

pcluster()