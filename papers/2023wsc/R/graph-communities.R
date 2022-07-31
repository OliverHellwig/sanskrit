# detect communities in the graph
library(XML)
library(ggplot2)
library(igraph)


path = '../data/vc.xml'
ps = xmlParse(file=path, encoding='UTF-8')


citations = getNodeSet(ps,"//citations")
a = lapply(citations, function(n){
  txts = getNodeSet(n, 'citation/text/text()')
  if(length(txts)>1){
    return(t(combn(xmlValue(txts),2))) # all pairwise combinations
    
  }else{
    return(NULL)
  }
})
b = do.call('rbind',a) # create one huge data frame
b = b[b[,1]!=b[,2],]# remove self loops
u = sort(unique(c(b[,1],b[,2])))
A = table(factor(b[,1],levels=u),factor(b[,2],levels=u)) # quadratic adjacency matrix
di = diag(A)
A = A + t(A) # make it symmetric
diag(A) = diag(A)-di # but no double entries on the diagonal (self loops)
nam = colnames(A)

# transform into igraph
w = which(A>0, arr.ind = TRUE)
w = w[w[,1]>w[,2],] # lower triangular
# Take care how the frequencies are calculated
# Raw frequencies produce some kind of unintelligible grouping (at least for me).
# log(freq) seems to produce a clustering by text category, e.g. Upanishads, Dharmasutras, Grhyasutras
# binary: ?
edges = data.frame(from=nam[w[,1]],to=nam[w[,2]],
                   weight=log(A[w]) # rep(1,nrow(w))#A[w]
                   )

# use the Louvain algorithm
g = graph_from_data_frame(edges, directed = FALSE)
lc = cluster_louvain(g)
membership(lc)
communities(lc) 
plot(lc, g)