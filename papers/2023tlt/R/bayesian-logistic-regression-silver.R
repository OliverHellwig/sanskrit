# logistic regression with gold data
library(rstan)
rstan_options(auto_write = TRUE)
library(shinystan)
library(data.table)
library(ggplot2)
library(coda)
library(loo)
library(readODS)
source('functions.R')


# Bayesian logistic regression
# run Stan

# collect the data
data.joint = build.data.joint(TRUE)


is.noun = ifelse(data.joint$head.pos %in% c('NOUN','ADJ'),1,0)
is.mantra.level = ifelse(data.joint$time.slot %in% c('1-RV','2-MA'), 1,0)
is.discourse = ifelse(data.joint$label=='discourse',1,0)
time = as.integer(factor(data.joint$time.slot))
head.pos = as.integer(factor(data.joint$head.pos,levels=sort(unique(data.joint$head.pos))))
head.label = as.integer(factor(data.joint$head.label, levels=sort(unique(data.joint$head.label))))
label.lvls = sort(unique(data.joint$label))
labels = as.integer(factor(data.joint$label,levels=label.lvls))
data.joint$position.diff = data.joint$word.position-data.joint$head.position
data.joint$position.diff[data.joint$position.diff < (-3)] = -3
data.joint$position.diff[data.joint$position.diff > 2] = 2
position.diff.lvls = sort(unique(data.joint$position.diff))
data.joint$position.diff = as.integer(factor(data.joint$position.diff,levels=position.diff.lvls))
w.corr.pred = which(data.joint$gold==1 & data.joint$from.vtb==0)


# w = which(data.joint$syntax.status=='gold')
# table(time[w],is.noun[w])
### ML glm
# 

dat = data.joint[data.joint$gold==1,]
dat$time.z = (as.integer(factor(dat$time.slot))-2)/3
mo = glm(correct ~ head.pos + time.slot + head.label + position.diff + from.vtb + head.pos:time.slot + position.diff:time.slot, 
         data=dat, family='binomial')
summary(mo)
pr = predict(mo, type='response')
boxplot(pr~factor(dat$correct))

model = 'blg-gdn-04a-time-or-mantra-with-silver'
Zj = stan(file=sprintf('stan/%s.stan',model),
         data=list(N=nrow(data.joint),
                   P=length(position.diff.lvls),
                   V=length(unique(data.joint$head.pos)),
                   H=length(unique(head.label)),
                   I=length(label.lvls),
                   nCorrPred = length(w.corr.pred),
                   time=time,
                   timeInt = time,
                   y=is.noun,
                   gold=data.joint$gold,
                   correct=data.joint$correct,
                   fromVTB = data.joint$from.vtb,
                   label = labels,
                   headPOS = head.pos,
                   headLabel = head.label,
                   mantra=is.mantra.level,
                   positionDiff = data.joint$position.diff
                   #goldDis=goldDis,silverDis=silverDis
                   ),
         iter = 2000,
         chains=3
)
saveRDS(Zj, file=sprintf('stan-result/%s.z',model))

# use model-comparison.R for the further evaluation!

launch_shinystan(Zj)

## loo
log_lik_3 = extract_log_lik(Zj, merge_chains = FALSE)
r_eff = relative_eff(exp(log_lik_3), cores = 2) 
loo_3 = loo(log_lik_3, r_eff = r_eff, cores = 2)
print(loo_3)


# coefficients
ae = extract(Zj, pars='a', permute=FALSE)
be = extract(Zj, pars='b', permute=FALSE)
ce = extract(Zj, pars='c', permute=FALSE)
c(sprintf('%.2f\\pm %.2f',mean(ae),sd(ae)),
  sprintf('%.2f\\pm %.2f',mean(be),sd(be)),
  sprintf('%.2f\\pm %.2f',mean(ce),sd(ce)))

u = extract(Zj,pars='c',permute=FALSE)
df = NULL
for(i in 1:dim(u)[1]){
  df = rbind(df, data.frame(value=as.double(u[i,,]),chain=1:4,iteration=i))
}
df$chain = factor(df$chain)
ggplot(df,aes(x=iteration,y=(value),fill=chain,colour=chain)) + geom_line()

# plot?
head.is.noun = ifelse(data.joint$head.pos %in% c('NOUN','ADJ'), 'noun', 'other')
w = which(data.joint$syntax.status=='gold')
go = table(head.is.noun[w], data.joint$time.slot[w])
go = go[1,]/colSums(go)
w = which(data.joint$syntax.status=='silver')
si = table(head.is.noun[w], data.joint$time.slot[w])
si = si[1,]/colSums(si)
df = data.frame(Proportion=c(go,si),
                Time=rep(names(go),2),
                Type=rep(c('gold','silver'),each=5))
A = mean(ae); B = mean(be); C = mean(ce)
ggplot(df,aes(x=Time,y=Proportion,fill=Type,colour=Type)) + 
  geom_bar(stat='identity',position='dodge') +
  stat_function(fun=function(x){
    #browser()
    w = A + B*(x-3)/2
    w[x<=2] = NA
    return(1/(1+exp(-w)))
  })


## predictive quality
head.is.noun = ifelse(data.joint$head.pos %in% c('NOUN','ADJ'), 'noun', 'other')
#Zj = readRDS(file=sprintf('stan-result/%s.z',model))
corr.pred = extract(Zj,pars='corr_pred',permute=FALSE)
x = rep(0,length(w.corr.pred))
y = rep(0,length(w.corr.pred))
for(i in 1:dim(corr.pred)[1]){
  x = x + colSums(corr.pred[i,,])
  y = y + apply(corr.pred[i,,],2,function(col){ta=table(col);as.integer(names(ta)[which.max(ta)])})
}
df = data.frame(gold=data.joint$correct[w.corr.pred],
                Aggregated.x=as.integer(x),
                Aggregated.y=as.integer(y))
ggplot(df,aes(y=Aggregated.y, group=gold)) + geom_boxplot()
go = data.joint$correct[w.corr.pred]
si = rep(1,nrow(df))
si[df$Aggregated.y < dim(corr.pred)[1]/2] = 0

(ta=table(go,si))
round(100*sum(diag(ta))/sum(ta),1)
TP = length(which(go==1 & si==1))
FP = length(which(go==0 & si==1))
FN = length(which(go==1 & si==0))
(P = 100*TP/(TP+FP))
(R = 100*TP/(TP+FN))
(F = 2*P*R/(P+R))

