# evaluate the results of gold and silver models
library(rstan)
library(loo)
library(ggplot2)
library(data.table)
source('functions.R')

models = c('blg-gdn-01-time',
           'blg-gdn-02-mantra',
           'blg-gdn-03-time-or-mantra',
           'blg-gdn-04a-time-or-mantra-with-silver'
           #'blg-gdn-05-time-and-mantra-bitone'
)

data = fread('../data/iva-syntax-gold.csv', sep='\t', encoding='UTF-8',data.table=FALSE)
data = data[data$label=='discourse',]
data.joint = build.data.joint(use.true.label = TRUE)
w.joint.gold.syntax = which(data.joint$syntax.status=='gold')
p = extract(readRDS(sprintf('stan-result/%s.z',models[4])),pars='p',permute=FALSE)
silver.chains = dim(p)[2]
joint.gold.slots = matrix(
  rep(as.integer(factor(data.joint$time.slot[w.joint.gold.syntax])),silver.chains),
  nrow=silver.chains, byrow=TRUE)

is.noun = ifelse(data$head.pos %in% c('NOUN','ADJ'),1,0)
ta = table(data$time.slot, is.noun)
noun = ta[,2]
total = rowSums(ta)

### PPC
short.names = c('Time','Reg.','Time/reg.', 'Time/reg., silver')
slots = sort(unique(data.joint$time.slot))
df = NULL
tab = c()
rates = NULL
#rates.models = c()
for(i in 1:length(models)){
  z = readRDS(sprintf('stan-result/%s.z',models[i]))
  
  # loo
  log_lik = extract_log_lik(z, merge_chains = FALSE)
  r_eff = relative_eff(exp(log_lik), cores = 2) 
  loo = loo(log_lik, r_eff = r_eff, cores = 2)
  print(loo)
  
  # for the table
  x = extract(z,pars='y_rep',permute=FALSE)
  y.rep = NULL
  for(j in 1:dim(x)[1]){
    y.rep = rbind(y.rep,x[j,,])
  }
  a = as.double(extract(z,pars='a',permute=FALSE))
  a = sprintf('$%.2f\\pm %.2f$',mean(a),sd(a))
  b = ''
  if(i %in% c(1,3,4)){
    b = as.double(extract(z,pars='b',permute=FALSE))
    b = sprintf('$%.2f\\pm %.2f$',mean(b),sd(b))
  }
  c = ''
  if(i %in% c(2,3,4)){
    c = as.double(extract(z,pars='c',permute=FALSE))
    c = sprintf('$%.2f\\pm %.2f$',mean(c),sd(c))
  }
  p = apply(y.rep,1,function(v){
    m = matrix(c(noun,v),nrow=2, byrow=TRUE)
    return(fisher.test(m)$p.value)
  })
  beta = mean( p < 0.05)
  tab = c(tab,
          paste(c(short.names[i], 
                  #a,b,c,
                  round(loo$estimates[1,],2),round(beta,4)), collapse=' & '))
  df = rbind(df,
             data.frame(x=rep(1:5,2),
                        Noun = c(
                          noun, # /total,
                          apply(y.rep,2,median) #/total #colMeans(y.rep)/total
                          ),
                        Model=short.names[i],
                        Type = rep(c('gold','predicted'),each=5)))
  m = matrix(c(noun,floor(colMeans(y.rep))),nrow=2, byrow=TRUE)
  print(fisher.test(m))
  
  # rates per time slot
  p = extract(z,pars='p',permute=FALSE)
  for(k in 1:dim(p)[1]){
    if(i==4){
      # we have nrow(data.joint) estimates per iteration and chain here.
      # average them per time slot to make the plot comparable with those of
      # the other models.
      
      rates = rbind(rates,
                    data.frame(
                      Estimate = as.double(apply(p[k,,w.joint.gold.syntax], 1, function(row) tapply(row, joint.gold.slots[1,], mean) )),
                      Slot = rep(slots,dim(p)[2]),
                      Model = short.names[i]))
    }else{
      # we have one estimate per time slot here
      rates = rbind(rates,
                    data.frame(Estimate=as.double(p[k,,]),
                      Slot = rep(slots,each=4),
                      Model = short.names[i]))
    }
    #rates.models = c(rates.models, rep(short.names[i],dim(p)[2]))
  }
  
}
cat(paste(tab, collapse='\\\\ \n'),file='../paper/tab-glm-gold.tex',append=FALSE)
df$Model = factor(df$Model, levels=short.names)
ggplot(df,aes(x=x,y=Noun,fill=Type,colour=Type)) + 
  geom_bar(stat='identity',position='dodge') + 
  facet_wrap(.~Model)

is.noun.joint = ifelse(data.joint$head.pos %in% c('NOUN','ADJ'),1,0)
w = which(data.joint$syntax.status=='gold' & data.joint$label=='discourse')
(ta=table(data.joint$time.slot[w],is.noun.joint[w]))
gold = ta[,2]/rowSums(ta)
w = which(data.joint$syntax.status=='silver' & data.joint$label=='discourse')
(ta=table(data.joint$time.slot[w],is.noun.joint[w]))
silver = ta[,2]/rowSums(ta)


gold.silver = data.frame(Estimate=c(gold,silver),Slot=rep(rownames(ta),2),Source=rep(c('Gold','Silver'),each=5))
rates$Slot = factor(rates$Slot)
rates$Model = factor(rates$Model, levels=short.names)
(plt=ggplot(rates,aes(y=Estimate,x=Slot,fill=Model)) + geom_boxplot() +
    geom_point(data=gold.silver,aes(x=Slot,y=Estimate,shape=Source),inherit.aes = FALSE,size=2,alpha=0.6, show.legend=FALSE) +
    geom_line(data=gold.silver,aes(x=Slot,y=Estimate,group=Source,linetype=Source),inherit.aes = FALSE, alpha=0.6, show.legend=FALSE) +
  theme(legend.position = 'bottom'))
ggsave(filename = '../paper/predicted-rates.png', plot = plt, device = 'png',
       width = 5, height=4, units = 'in')
