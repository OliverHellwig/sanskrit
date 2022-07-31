# The Random Kavi model
library(XML)
library(ggplot2)
library(igraph)
library(Rmpfr) # high precision floating numbers
library(CVXR)
library(data.table)
library(hrbrthemes)

bincoeff = function(N,n,use.log){
  if(n==0 | n==N){
    return(ifelse(use.log==TRUE,0,1))
  }
  #browser()
  a = max(n,N-n)
  b = min(n,N-n)
  if(use.log==FALSE){
    u = prod((a+1):N)
    v = factorial(b)
    return(u/v)
  }else{
    u = sum(log((a+1):N))
    if(any((1:b)<0)){
      browser()
    }
    v = sum(log(1:b))
    return(u-v)
  }
}

######## Parameters
outpath = '../output/hypergeom.dat'

######## Step 1
path = '../data/vc.xml'
ps = xmlParse(file=path, encoding='UTF-8')


citations = getNodeSet(ps,"//citations")
a = lapply(citations, function(n){
  return(xmlValue(getNodeSet(n, 'citation/text/text()')))
})
# how many Mantras?
N = length(a)
# how many unique texts?
nam = sort(unique(as.character(unlist(a))))
nTexts = length(nam)
# transaction matrix, M[i,j]=1 -> text j contained in transaction i
M = matrix(0,nrow=N,ncol=nTexts)
for(i in 1:length(a)){
  M[i,match(a[[i]],nam)] = 1
}

# main function

cat(iconv('text.A text.B A B AB p\n',to='UTF-8'),file=outpath,append=FALSE)
for(a in 1:(length(nam)-1))
{
  for(b in (a+1):length(nam)){
    print(sprintf('%d %d',a,b))
    ta = table(M[,a],M[,b])
    A = max(sum(ta[2,]),sum(ta[,2]))
    B = min(sum(ta[2,]),sum(ta[,2]))
    
    x = vector('list',B+1)
    bits = 1024
    su = mpfr(0,precBits =  bits)
    
    Na = bincoeff(N,A,use.log=TRUE)
    Nb = bincoeff(N,B,use.log=TRUE)
    
    
    for(k in 0:B){
      ak = bincoeff(A,k,use.log=TRUE)
      Nabk = bincoeff(N-A,B-k,use.log=TRUE)
      val = exp(mpfr(ak+Nabk-Na-Nb, precBits = bits))
      x[[k+1]] = val
      su=su+val
    }
    X = rep(0,length(x))
    for(k in 1:length(x)){
      X[k] = asNumeric(x[[k]]/su)
    }
    K = ta[2,2]
    plot(X,t='l', main=sprintf('%s [%d] > %s [%d]: %d > %.3f', nam[a],a,nam[b],b,K,cumsum(X)[K]))
    abline(v=K,col='red') # common
    cat(iconv(
      sprintf('%s %s %d %d %d %f\n',nam[a],nam[b],
              sum(ta[2,]),sum(ta[,2]),K,cumsum(X)[K+1]),
      to='UTF-8'), file=outpath, append=TRUE)
  }
}


########### Step 2
data = fread(outpath, sep=' ', header=TRUE,encoding = 'UTF-8',data.table=FALSE)
info = fread('../data/Dates - Primary sources.tsv', header=TRUE,
             sep='\t', encoding='UTF-8', data.table = FALSE)
type1 = info$Group[match(data$text.A,info$`Abbreviation VC`)]
type2 = info$Group[match(data$text.B,info$`Abbreviation VC`)]
data = cbind(data,type1,type2)

# Samhita texts only
#w = which(data$p>0.9 & data$type1=='S' & data$type2=='S')
w = which(data$type1=='S' & data$type2=='S' & (data$p<0.001 | data$p>0.999))
nodes = sort(unique(c(data$text.A[w],data$text.B[w])))
path = '../output/gephi-nodes-hypergeometric.csv'
cat(iconv(
  sprintf('Id Label\n%s',paste(sprintf('%s %s',nodes,nodes),collapse = '\n')),
  to='UTF-8'),file=path)
wts = as.double(data$p[w])
wts = ceiling(1+2*(wts-min(wts))/(max(wts)-min(wts)))
attr = rep('attracting',length(w))
attr[data$p[w]<0.5] = 'repelling'
cat(iconv(
  sprintf('Source Target Weight Attracting\n%s',
          paste(sprintf('%s %s %d %s',data$text.A[w],data$text.B[w],wts,attr),collapse='\n')),
  to='UTF-8'),file='../output/gephi-edges-hypergeometric.csv')



# Samhita only, but attracting and repelling
w = which(data$type1=='S' & data$type2=='S' & (data$p<0.001 | data$p>0.999))
dx = data[w,]
txts = sort(unique(c(data$text.A[w],data$text.B[w])))
Mat = matrix(0,nrow=length(txts),ncol=length(txts))
a = match(data$text.A[w],txts)
b = match(data$text.B[w],txts)
for(i in 1:length(a)){
  Mat[a[i],b[i]] = Mat[b[i],a[i]] = ifelse(dx$p[i]<0.5,-1,+1)
}
coords = runif(2*length(txts),0,1)

fun = function(a){
  n = length(a)
  m = n/2
  x = a[1:m]
  y = a[(m + 1):n]
  u = 0
  for(i in 1:(m-1)){
    for(j in (i+1):m){
      if(Mat[i,j]!=0){
        d = ifelse(Mat[i,j]==1, 0.2, 0.8)
        p = ifelse(Mat[i,j]==1, 1, 0.1)
        u = u + p*abs( sqrt((x[i]-x[j])^2 + (y[i]-y[j])^2) - d )
      }
    }
  }
  #browser()
  return(u)
}

grad = function(a){
  n = length(a)
  m = n/2
  x = a[1:m]
  y = a[(m + 1):n]
  G = rep(0,n)
  for(i in 1:(m-1)){
    for(j in (i+1):m){
      if(Mat[i,j]!=0){
        sgn = Mat[i,j]
        d = ifelse(sgn==1, 0.2, 0.8)
        p = ifelse(sgn==1, 1, 0.1)
        delta = sqrt((x[i]-x[j])^2 + (y[i]-y[j])^2)
        f = p*(delta - d)/(delta*abs(delta-d))
        G[i] = G[i] + (x[i]-x[j])*f
        G[j] = G[j] - (x[i]-x[j])*f
        G[i+m] = G[i+m] + (y[i]-y[j])*f
        G[j+m] = G[j+m] - (y[i]-y[j])*f
      }
    }
  }
  return(G)
}

res = optim(coords,fun,grad,method='BFGS')
coords = matrix(res$par,ncol=2, byrow=FALSE)
# for Gephi
cat(iconv(
  sprintf('Id Label x y\n%s',paste(sprintf('%s %s %f %f',txts,txts,coords[,1],coords[,2]),collapse = '\n')),
  to='UTF-8'),file='../output/gephi-nodes-hypergeometric.csv')
# for the paper
df.coord = data.frame(coords)
df.coord = cbind(df.coord, Text=txts)
colnames(df.coord) = c('x','y','Text')
df.lines = data.frame(x=coords[a,1],y=coords[a,2],
                      xend=coords[b,1],yend=coords[b,2],
                      Type = ifelse(data$p[w]<0.001,'low','high'),
                      width = ifelse(data$p[w]<0.001, 0.5, 0.7))

(plt=ggplot(df.lines, aes(x=x,y=y,xend=xend,yend=yend,colour=Type,size=Type)) +
  geom_segment(alpha=0.4) +
  scale_color_manual(values=c('green','red')) +
  scale_size_manual(values=c(2,1)) +
  geom_text(data=df.coord,aes(x=x,y=y,label=Text),inherit.aes=FALSE,size=4) +
  theme_ipsum() + 
  theme(
    legend.position="none"
  ) + xlab('') + ylab(''))
ggsave(filename = '../paper/img-input/random-kavi-result.png', plot = plt, device = 'png', 
       width = 6, height=6, units = 'in')
