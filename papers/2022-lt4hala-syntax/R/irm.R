# with Gaussians on the classes
library(data.table)
library(Rcpp)
library(stringr)
library(ggplot2)
source('functions.R')
Sys.setenv("PKG_CXXFLAGS"="-std=c++0x -Wall -O3") # compiler flags, needed for std
sourceCpp('cpp/functions.cpp')

# create or load the data
n.sample = 0 # 0 = all data
rds.path = sprintf('../data/data-%d.rds',n.sample)
if(file.exists(rds.path)){
  d = readRDS(rds.path)
}else{
  d = mm.data(n.sample)
  saveRDS(d, file=rds.path)
}
sig.level = 0.01



x = unigramLinks(d$mm.data$dependents,d$lookup$cost,2) # this is very slow + could be optimized
keep = sort(unique(c(x$i,x$j)))
ut = unigramDistributions(d$mm.data$dependents,d$mm.data$time)
# filter 
g.t = dist.g.test(ut)[keep]
n.old=length(keep)
keep = keep[g.t < sig.level]
print(sprintf('number/node: %d > %d',n.old,length(keep)),q=FALSE)
x = x[(x$i %in% keep) & (x$j %in% keep),]

us = unigramDistributions(d$mm.data$dependents,d$mm.data$school)
ur = unigramDistributions(d$mm.data$dependents,d$mm.data$register)
counts = rowSums(ut[keep,])
kl.t = dist.hellinger(ut)[keep]
kl.s = dist.hellinger(us)[keep]
kl.r = dist.hellinger(ur)[keep]
o = kl.s + kl.r - kl.t

di.t = dist.diff(ut)[keep,]
di.s = dist.diff(us)[keep,]
di.r = dist.diff(ur)[keep,]
ut = ut[keep,]
us = us[keep,]
ur = ur[keep,]
g.t = dist.g.test(ut)
g.s = dist.g.test(us)
g.r = dist.g.test(ur)


# initialization and parameters
n = length(keep)
X = match(x$i,keep)
Y = match(x$j,keep)

K = 30
z = sample(K,n,replace=TRUE) # class assignments
alpha = 1 # DP
beta = 0.5
itrs = 100
sigma.t = 0.1
mu.t.0 = 0
sigma.t.0 = 0.1

A = as.integer(table(factor(z,levels=1:K)))

#library(profvis)
#profvis({
for(itr in 1:itrs){
  ixes = sample(n,n,replace=FALSE)
  print(itr)
  for(i in 1:length(ixes)){
    ix = ixes[i]
    
    # decrement
    A[z[ix]] = A[z[ix]]-1
    stopifnot(all(A>=0))
    res = rebuildBC_old(ix,X,Y,z,K)
    stopifnot(all(res$B>=0))
    stopifnot(all(res$C>=0))
    tix = as.integer(ut[ix,])
    
    # aggregate makes problems for singleton classes + is slow
    mu.ml = muMl(ix,di.t,z,K)
    
    N = matrix(rep(A,ncol(di.t)),nrow=K,byrow=FALSE)
    # Bishop, eq. 2.141
    m = sigma.t^2/(N*sigma.t.0^2+sigma.t^2) + (N*sigma.t.0^2)/(N*sigma.t.0^2 + sigma.t^2)*mu.ml
    s = matrix(rep(sqrt(1/(1/sigma.t.0^2 + A/sigma.t^2)),ncol(di.t)),nrow=K,byrow=FALSE)
    xs = matrix(rep(di.t[ix,],K),nrow=K,byrow=TRUE)
    rs = rowSums(dnorm(xs,mean=m,sd=s+sigma.t.0,log=TRUE))
    
    pa = rep(0,K+1)
    pb = rep(0,K+1)
    if(K>0){
      pa[1:K] = log(A)
      pb[1:K] = as.double(colSums(
        lbeta(res$B+rep(res$np,K)+beta,res$C+rep(res$nm,K)+beta) - lbeta(res$B+beta,res$C+beta)
        )) + rs
    }
    
    
    
    # new class
    pa[K+1] = log(alpha)
    pb[K+1] = sum(lbeta(res$np+beta,res$nm+beta) - lbeta(beta,beta)) + 
      sum(dnorm(di.t[ix,],mean=mu.t.0,sd=sqrt(sigma.t^2 + sigma.t.0^2),log=TRUE))
    # Gumbel max trick
    g = -log(-log(runif(K+1,0,1)))
    zold = z[ix]
    znew = which.max(g+pa+pb)
    # assignments and updates
    z[ix] = znew
    if(znew<=K){
      A[znew]=A[znew]+1
    }else{
      A = c(A,1)
      K=K+1
      stopifnot(K==length(A))
    }
    if(zold>-1){
      if(A[zold]==0){
        w = which(z>zold)
        if(length(w)>0){
          z[w] = z[w]-1
        }
        A = A[-zold]
        K=K-1
      }
    }
  }
  if(itr %% 10==0){print(table(z))}
}

# result as text > csv
df = data.frame(constituent=d$lookup$feature[keep],
                group = z, value=o,
                gValue = (1-g.s)*kl.s + (1-g.r)*kl.r - (1-g.t)*kl.t,
                time=kl.t,school=kl.s,register=kl.r,
                gTime=g.t, gSchool=g.s, gRegister=g.r,
                stringsAsFactors = FALSE)
df = cbind(df,di.t)
df = df[order(df$group,df$value),]
w = which(!duplicated(df$group))
groups.r=df$group[w][order(df$value[w])]
df = df[order(match(df$group,groups.r),df$value),]

fwrite(df,'../data/irm-result.csv',sep=',')
