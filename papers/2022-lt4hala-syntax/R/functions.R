# general purpose functions
library(DescTools)

build.features = function(word)
{
  u.default = apply(expand.grid(2:1,2:1,2:1,4:1),1,prod)
  u.root    = apply(expand.grid(2:1,2:1,2:1,2:1),1,prod)
  pos = c(word$pos,'_')
  morphosyntax = c(word$cas,'_')
  lab = c(word$label,'_')
  if(word$label=='root'){
    placement = c('root','_')
    u = u.root
  }else{
    di = word$head.position-word$wrd.position
    if(abs(di)>4){
      di = sign(di) * 4;
    }
    placement = c(di,sprintf('a%d',abs(di)),ifelse(di>0,'+','-'),'_')
    u = u.default
  }
  e = expand.grid(pos,morphosyntax,lab,placement)
  return(list('f'=apply(e,1,function(r)paste(r,collapse='/')), 'u'=u))
}

remove.mantras = function(sen){
  #browser()
  # find consecutive ranges in the head positions.
  spl = split(sen$head.position, cumsum(c(1, diff(sen$head.position) != 1)))
  # get the respective word positions
  word.positions = split(sen$wrd.position,
                         f=rep(1:length(spl),unlist(lapply(spl,length)) ))
  mantra.pos = c('flat','ccomp','orphan','mark','parataxis','cc')
  if(length(spl)==1){
    # this may be one sentence consisting of a single mantra citation, so include the root.
    mantra.pos = c(mantra.pos,'root')
  }
  keep = unlist(lapply(word.positions,function(ixes){
    if(any(sen$pos[ixes]=='MANTRA') & all(sen$label[ixes] %in% mantra.pos ) ){
      return(rep(FALSE,length(ixes)))
    }
    return(rep(TRUE,length(ixes)))
  }))
  sen.rem = sen[keep,]
  
  
  len = -1
  while(nrow(sen.rem)!=len)
  {
    w = which(sen.rem$head.position!=0 & 
                (!(sen.rem$head.position %in% sen.rem$wrd.position) |
                   (sen.rem$pos=='MANTRA'))
    )
    if(length(w)>0){
      #browser()
      sen.rem = sen.rem[-w,]
    }
    len = nrow(sen.rem)
  }
  
  return(sen.rem)
}

# removes mantras, and splits the data into sentences
preprocess.vtb.data = function(data)
{
  print(sprintf('%d words **before** removing mantras', nrow(data)),q=FALSE)
  # remove all sentences in which the root is a mantra. Mostly obscure elliptic constructions.
  u = unique(data$sen.id[which(data$pos=='MANTRA' & data$label=='root')])
  if(length(u)!=0){
    print(sprintf('Removing %d sentences with root mantras',length(u)),q=FALSE)
    data = data[which(!(data$sen.id %in% u)),]
  }
  print(sprintf('%d words **after** removing root mantras', nrow(data)),q=FALSE)
  sens = split(data, f=data$sen.id)
  
  sens = lapply(sens,function(s)remove.mantras(s))
  n = sum(unlist(lapply(sens,nrow)))
  print(sprintf('%d words **after** removing mantras', n),q=FALSE)
  return(sens)
}

mm.data = function(n.smp, feature.threshold=20, filter.fun=NULL)
{
  d = fread('../data/conllu.dat', header=TRUE, sep='\t', encoding='UTF-8', data.table=FALSE)
  info = fread('../data/Dates - Primary sources.tsv', sep='\t', data.table = FALSE,
               encoding='UTF-8')
  # texts not found, check the abbreviations in the DCS
  sort(unique(d$text[which(!(d$text %in% info$Abbreviation))]))
  # only schools and groups that are relevant for our data (else too many categories)
  ma = match(unique(d$text),info$Abbreviation)
  schools = sort(unique(info$School[ma]))
  ## sort the groups in a somehow chronological order
  ta=table(info$Group[match(d$text,info$Abbreviation)],d$layer)
  srt=apply(ta,1,function(row)sum(((1:5)*row)/sum(row)))
  groups = as.character(names(srt)[order(as.double(srt),decreasing=FALSE)])
  annotators = sort(unique(d$annotator))
  
  d$cas[d$pos=='MANTRA'] = 'mantra'
  d$cas[d$pos %in% c('ADV','CCONJ','PART', 'SCONJ')] = 'ind'
  # split into sentences
  sens = preprocess.vtb.data(d)
  
  if(n.smp<=0){
    smp = 1:length(sens)
  }else{
    smp = sort(sample(length(sens),n.smp,replace=FALSE))
  }
  ## first round: get all data in order to apply a frequency threshold
  lookup = data.frame(feature=c('root'),count=c(0),cost=c(1))
  for(k in smp)
  {
    if(k %% 1000==0){print(k,q=FALSE)}
    sen = sens[[k]]
    if(any(sen$head.position!=0))
    {
      w.txt = which(info$Abbreviation==sen$text[1])
      if(length(w.txt)!=1){
        #browser()
        print(sprintf('Text not found: %s', sen$text[1]))
      }else{
        ixes = rep('', nrow(sen))
        for(i in 1:nrow(sen)){
          accept=FALSE
          if(is.null(filter.fun)){
            accept=TRUE
          }else if(filter.fun(sen[i,])==TRUE){
            accept=TRUE}
          if(accept){
            feats = build.features(sen[i,])
            f = feats$f
            u = feats$u
            ma = match(f,lookup$feature)
            w = which(is.na(ma))
            if(length(w)>0){
              xx = data.frame(feature=f[w],count=rep(0,length(w)),cost=u[w])
              xx = xx[!duplicated(xx$feature),]
              lookup = rbind(lookup,xx)
              ma = match(f,lookup$feature)
            }
            lookup$count[ma] = lookup$count[ma]+1
          }
        }
      }
    }
  }
  # restrict the number of analyses
  lookup = lookup[lookup$count>feature.threshold | lookup$feature=='root',]
  
  if(is.null(filter.fun)){
    useless = c('_/_/_/_', '_/_/root/root')
    w = which(lookup$feature %in% useless)
    #stopifnot(length(w)==length(useless))
    if(length(w)>0){
      lookup = lookup[-w,]
    }
  }
  stopifnot(any(duplicated(lookup))==FALSE)
  print(sprintf('%d features', nrow(lookup)),q=FALSE)
  
  ## second round: it's getting serious
  outpath.data = sprintf('../data/sampler-%d.input',n.smp)
  outpath.data.comp = sprintf('../data/sampler-all-%d.input',n.smp)
  append = FALSE
  Parents = c(); Parents.full = c()
  Dependents = c(); Dependents.full = c()
  Dependent.positions = c()
  Texts = c();Schools = c(); Groups = c(); Times = c(); Registers = c()
  Sentences = c()
  for(k in smp)
  {
    if(k %% 1000==0){print(k,q=FALSE)}
    sen = sens[[k]]
    if(any(sen$head.position!=0) & length(which(sen$head.position==0))==1)
    {
      w.txt = which(info$Abbreviation==sen$text[1])
      if(length(w.txt)==1){
        w.school = which(schools==info$School[w.txt])
        w.group  = which(groups ==info$Group[w.txt])
        w.annotators = match(sen$annotator,annotators)
        stopifnot(any(is.na(w.annotators))==FALSE)
        layer    = sen$layer[1]
        stopifnot(layer %in% 1:5)
        ## determine the register
        register = ifelse(layer %in% c(1,2), 1, 2)
        # metrical Upanisads, tested
        if(sen$text[1]=='\u015AvetU'){
          register = 1
        }else if(sen$text[1]=='JUB'){
          if(length(grep('^4_18_.+',sen$chapter[1]))>0 | 
             length(grep('^4_19_.+',sen$chapter[1]))>0 |
             length(grep('^4_20_.+',sen$chapter[1]))>0 | 
             length(grep('^4_21_.+',sen$chapter[1]))>0){
            register = 1
          }
        }
        
        ixes = rep('', nrow(sen))
        feats= rep('', nrow(sen)) # for compositional models
        orig = rep('', nrow(sen)) # original full feature
        dep.positions = 1:nrow(sen)
        for(i in 1:nrow(sen)){
          accept=FALSE
          if(is.null(filter.fun)){
            accept=TRUE
          }else if(filter.fun(sen[i,])==TRUE){
            accept=TRUE}
          if(accept){
            f = build.features(sen[i,])$f
            orig[i] = f[1]
            ma = match(f,lookup$feature)
            ma = ma[!is.na(ma)]
            if(length(ma)==0){
              browser()
            }
            stopifnot(any(is.na(ma))==FALSE)
            ixes[i] = paste(ma,collapse=' ')
            feats[i]= paste(lookup$feature[ma],collapse=' ')
          }
        }
        
        parents = rep('',nrow(sen))
        parents.comp = rep('',nrow(sen))
        parents.full = rep('',nrow(sen))
        parents[sen$label=='root'] = '1' # root label
        parents.full[sen$label=='root'] = 'root'
        parents.comp[sen$label=='root'] = 'root/root/root/root'
        ma = match(sen$head.position,sen$wrd.position)
        w = which(!is.na(ma))
        parents[w] = ixes[ma[w]]
        parents.comp[w] = feats[ma[w]]
        parents.full[w] = orig[ma[w]]
        
        #lns = sprintf('%d$%d$%d$%d$%d$%s$%s',w.txt,w.school,w.group,sen$layer,register,parents,ixes)
        #cat(sprintf('%s\n',paste(lns,collapse='\n')),file=outpath.data,append=append)
        #lns = sprintf('%d$%d$%d$%d$%d$%s$%s$%d',w.txt,w.school,w.group,sen$layer,register,parents.comp,feats,w.annotators)
        #cat(sprintf('%s\n',paste(lns,collapse='\n')),file=outpath.data.comp,append=append)
        w = which(feats!='')
        if(length(w)!=length(feats)){
          if(is.null(filter.fun)){
            browser() # this should not happen
          }else{
            parents = parents[w]
            ixes = ixes[w]
            parents.full = parents.full[w]
            orig = orig[w]
            dep.positions = dep.positions[w]
          }
        }
        Parents = c(Parents,parents)
        Dependents = c(Dependents,ixes)
        Parents.full = c(Parents.full,parents.full)
        Dependents.full = c(Dependents.full,orig)
        Dependent.positions = c(Dependent.positions,dep.positions)
        if(length(Parents)!=length(Dependent.positions)){
          browser()
        }
        nn = length(parents)
        Schools = c(Schools, rep(info$School[w.txt],nn))
        Groups  = c(Groups, rep(info$Group[w.txt],nn))
        Times   = c(Times, rep(layer,nn))
        Texts   = c(Texts, rep(w.txt,nn))
        Sentences = c(Sentences, rep(k,nn))
        Registers = c(Registers, rep(register,nn))
      }else{
        print('text not found',q=FALSE)
      }
    }
  }
  Schools = match(Schools,schools)
  Groups  = match(Groups,groups)
  df.data = data.frame(parents=Parents,dependents=Dependents,
                       parent=Parents.full,dependent=Dependents.full,
                       dependent.position=Dependent.positions,
                       school=Schools,group=Groups,time=Times,text=Texts,
                       sentence=Sentences,register=Registers,
                       stringsAsFactors = FALSE)
  return(list(
    'src.data' = d,
    'sample' = smp,
    'lookup' = lookup,
    'sentences' = sens,  
    'schools' = schools,
    'groups'  = groups,
    'annotators' = annotators,
    'mm.data' = df.data,
    'info' = info
  ))
}


# KL divergences between an observed variable and the corpus distribution
dist.kl = function(mat){
  S = colSums(mat)
  N = sum(mat)
  r = rep(0,nrow(mat))
  for(i in 1:nrow(mat)){
    n = sum(mat[i,])
    o = mat[i,]/n
    e = (S - mat[i,])/(N-n)
    w = which(o>0 & e>0)
    r[i] = sum(o[w]*log(o[w]/e[w]))
    if(is.na(r[i])){
      browser()
    }
  }
  return(r)
}

# Hellinger between an observed variable and the corpus distribution
dist.hellinger = function(mat){
  S = colSums(mat)
  N = sum(mat)
  r = rep(0,nrow(mat))
  for(i in 1:nrow(mat)){
    n = sum(mat[i,])
    o = mat[i,]/n
    e = (S - mat[i,])/(N-n)
    r[i] = 1/sqrt(2) * sqrt(sum( (sqrt(e)-sqrt(o))^2 ))
    if(i>1 & is.na(r[i])){ # i==1 > root
      browser()
    }
  }
  return(r)
}

# does a distribution differ significantly from the corpus distribution?
dist.g.test = function(mat, sig.level=0.01){
  S = colSums(mat)
  r = rep(1,nrow(mat))
  for(i in 2:nrow(mat)){ # the first row is the root. Skip it.
    tst = GTest(matrix(c(mat[i,],S-mat[i,]),nrow=2,byrow=TRUE))
    r[i] = tst$p.value
  }
  return(r)
}

dist.diff = function(mat){
  S = colSums(mat)
  N = sum(mat)
  r = matrix(0,nrow=nrow(mat),ncol=ncol(mat))
  for(i in 1:nrow(mat)){
    n = sum(mat[i,])
    o = mat[i,]/n
    e = (S - mat[i,])/(N-n)
    r[i,] = o-e
  }
  return(r)
}

