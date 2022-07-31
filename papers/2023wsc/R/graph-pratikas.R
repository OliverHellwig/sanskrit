# do younger sutras tend to cite pratikas?

library(XML)
library(ggplot2)
library(igraph)
library(stringi)
library(data.table)
library(DescTools)

path = '../data/vc.xml'
ps = xmlParse(file=path, encoding='UTF-8')
info = fread('../data/Dates - Primary sources.tsv', sep='\t', data.table = FALSE,
             encoding='UTF-8')
ss = '\u015AS'
shrauta = info$`Abbreviation VC`[stri_cmp_eq(info$Group, ss) & nchar(info$`Abbreviation VC`)>0]
#grhya = info$`Abbreviation VC`[stri_cmp_eq(info$Group, 'GS') & nchar(info$`Abbreviation VC`)>0]

citations = getNodeSet(ps,"//citations")
a = lapply(citations, function(n){
  vars = getNodeSet(n, 'citation')
  res = NULL
  if(length(vars)>1){
    # for debugging
    #ln = xmlValue(getNodeSet(n,'parent::line/number/text()'))
    
    srcs = c()
    tars = c()
    types = c()
    for(v in vars){
      txt = xmlValue(getNodeSet(v, 'text/text()')[[1]])
      if((txt %in% shrauta)){
        tars = c(tars, txt)
        srcs = c(srcs,
                 xmlValue(getNodeSet(n, 'parent::line/citations/citation/text/text()')[1][[1]]))
        types= c(types, xmlGetAttr(v,'type'))
      }
    }
    if(length(srcs)>0){
      res = data.frame(cited=srcs,citing=tars,type=types,stringsAsFactors = FALSE)
    }
  }
  return(res)
})
b = do.call('rbind',a) # create one huge data frame
b = b[b$cited!=b$citing,]
# todo: Tables for the paper!
school.cited = info$School[match(b$cited,info$`Abbreviation VC`)]
school.citing= info$School[match(b$citing,info$`Abbreviation VC`)]
ta = table(school.citing,school.cited)
ta = ta/rowSums(ta)
b = b[school.citing==school.cited,]

b$type[b$type=='variant'] = 'default'
w = which(b$type %in% c('default','pratika'))
ta = table(b$citing[w],b$type[w])
df = data.frame(text=rownames(ta), sakala=as.integer(ta[,1]), 
                pratika=as.integer(ta[,2]), type=rep('',nrow(ta)),
                ratio = 100 * ta[,2]/rowSums(ta) )
df$type = info$Group[match(df$text,info$`Abbreviation VC`)]
#df$ratio = 100*df$pratika/(df$sakala+df$pratika)
# sort by increasing ratios
df = df[order(df$ratio,decreasing = FALSE),]
df = df[which(df$type==ss & (df$sakala+df$pratika)>100),] # Shrautasutras with sufficient support
n = nrow(df)
ms = matrix(0,nrow=n,ncol=n)
mp = matrix(0,nrow=n,ncol=n)
earlier = c()
later = c()
wts = c()
pvals = c()
# perform pairwise G-tests
signif = 0.01
for(i in 1:(n-1)){
  for(j in (i+1):n){
    m = matrix(c(df$sakala[i],df$pratika[i],
                 df$sakala[j],df$pratika[j]),nrow=2, byrow=TRUE)
    tst=GTest(m)
    ms[i,j] = tst$statistic[1]
    mp[i,j] = tst$p.value
    if(tst$p.value < signif){
      earlier = c(earlier,df$text[i])
      later = c(later, df$text[j])
      wts = c(wts, tst$statistic[1])
    }
  }
}
dfg = data.frame(from=earlier,to=later,weight=wts)
g = graph_from_data_frame(dfg)
plot(g)
top = names(topo_sort(g,mode='out'))
s = '';group = ''
for(i in 1:(length(top)-1)){
  ti = top[i]
  tj = top[i+1]
  w = which(dfg$from==ti & dfg$to==tj)
  if(length(w)==0){
    print(sprintf('no difference: %s > %s',ti,tj))
    
  }else{
    if(group!=''){
      s = sprintf('%s \\topbox{%s}',s,group)
    }else{
      s = paste(s,ti,collapse=' ')
    }
  }
}

## for Gephi, not used in the paper
nodes = sort(unique(c(earlier,later)))
cat(iconv(
  sprintf('Id Label\n%s',paste(sprintf('%s %s',nodes,nodes), collapse='\n')), to='UTF-8'),
  file='../gephi/pratika-nodes.csv')
lwts = log(wts)
x = round(1+2*(lwts-min(lwts))/(max(lwts)-min(lwts)))
cat(iconv(
  sprintf('Source Target Weight\n%s',paste(sprintf('%s %s %d',earlier,later,x), collapse='\n')), to='UTF-8'),
  file='../gephi/pratika-edges.csv')
