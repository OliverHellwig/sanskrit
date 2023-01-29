library(readODS)



m.std = function(z,var,last.itr=FALSE)
{
  q = extract(z,pars=var,permuted=FALSE)
  if(last.itr==TRUE){
    final.vals = q[dim(q)[1],,]
    mu=as.double(apply(final.vals,2,mean))
    s=as.double(apply(final.vals,2,sd))
  }else{
    K = dim(q)[3]
    mu = rep(0,K)
    s = rep(0,K)
    for(k in 1:K){
      mu[k] = mean(q[,,k])
      s[k] = sd(q[,,k])
    }
  }
  return(list('mu'=mu,'sd'=s))
}

build.data.joint = function(use.true.label = FALSE)
{
  gold = fread('../data/iva-syntax-gold.csv', sep='\t', encoding='UTF-8',data.table=FALSE)
  silver = read_ods('../data/iva-syntax-silver-random-2.ods')
  #browser()
  # if(use.true.label==FALSE){
  #   silver.anno = silver$decision
  #   silver.anno[is.na(silver.anno)] = ''
  #   decision = c(rep('y',nrow(gold)),silver.anno)
  # }else{
  silver.dec = rep('',nrow(silver))
  silver.dec[silver$true.label=='discourse'] = 'y'
  silver.dec[!is.na(silver$true.label) & nchar(silver$true.label)>0 & silver$true.label!='discourse'] = 'n'
  gold.dec = ifelse(gold$label=='discourse','y','n')
  decision = c(gold.dec,silver.dec)
  #}
  
  
  data.joint = data.frame(
    head.pos = c(gold$head.pos,silver$head.pos),
    head.label = c(gold$head.label, silver$head.label),
    time.slot = c(gold$time.slot,silver$time.slot),
    label = c(gold$label,silver$label),
    syntax.status = c(gold$syntax.status,silver$syntax.status),
    word.position = c(gold$word.position,silver$word.position),
    head.position = c(gold$head.position,silver$head.position)
  )
  
  gold = ifelse(data.joint$syntax.status=='gold',1,0) # gold and correct are identical!
  gold[which(decision %in% c('y','n'))] = 1
  data.joint = cbind(data.joint,
                     gold=gold,
                     correct = ifelse(decision=='y',1,0),
                     from.vtb = ifelse(data.joint$syntax.status=='gold',1,0))
  if(use.true.label==FALSE){
    data.joint = data.joint[data.joint$label=='discourse',]
  }
  return(data.joint)
}