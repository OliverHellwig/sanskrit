# some descriptive statistics for the paper
library(XML)
path = '../data/vc-KGS.xml'
outpath = '../paper/tab-corpus.tex'
ps = xmlParse(file=path, encoding='UTF-8')



## Table: corpus
c = getNodeSet(ps, "//citations")
u = unlist(lapply(c,function(k){
  return(length(getNodeSet(k,'citation')))
}))
lns = c(
  sprintf('Mantras & %s', 
          prettyNum(getNodeSet(ps,"count(//line)"),big.mark=",")),
  sprintf('Occurrences & %s', 
          prettyNum(getNodeSet(ps, "count(//citation)"),big.mark=',')),
  sprintf('Mantras with more than one occurrence & %s',prettyNum(length(which(u>1)),big.mark=",")),
  sprintf('Prat{\\={\\i}}kas &  %s', 
          prettyNum(getNodeSet(ps, "count(//citation[@type='pratika'])"),big.mark=',')),
  sprintf('Minor variants & %s', 
          prettyNum(getNodeSet(ps, "count(//citation[@type='variant'])"),big.mark=',')),
  sprintf('Cross-references & %s', 
          prettyNum(getNodeSet(ps, "count(//targetNumber[@type='parent'])"),big.mark=','))
  ### ... and much more! added by us, which texts, ...
)

txt = paste(lns, collapse=' \\\\ \n')
cat(iconv(txt,to='UTF-8'),file=outpath)