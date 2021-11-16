# functions for evaluating citation models
library(data.table)
source('constants.R')

load.mm.result = function(model,affix)
{
  path = sprintf('../data/output/result-%s-%s.dat',model,affix)
  if(!file.exists(path)){
    print(sprintf('file %s not found!',path),quote=FALSE)
    return(NULL)
  }
  res = fread(file=path, sep=' ', header = TRUE, data.table = FALSE)
  # if(length(colNames)==0){
  #   res = fread(file=path, sep=' ', header = TRUE, data.table = FALSE)
  # }else{
  #   res = fread(file  = path, sep=' ', data.table = FALSE)
  #   names(res) = colNames
  # }
  return(res)
}

# Generates plots for Fig 1 and 2 of the paper
citation.mm.plots = function(res,conn,info,periodization)
{
  #browser()
  true.dates = info[,4]
  c = conn
  diag(c)=0
  citablePerText = as.integer(rowSums(c))/nrow(info)
  own = table(factor(res$Doc[res$Doc==res$Cit], levels=1:nrow(info)))
  citing = table(factor(res$Doc[res$Doc!=res$Cit], levels=1:nrow(info)))
  rCitingOwn = as.double((citing+1)/(citing+own+1))
  
  # some intermediate plots
  o = par(mfrow=c(2,2))
  plot(rCitingOwn~true.dates)
  moLm = lm(rCitingOwn~citablePerText)
  plot(rCitingOwn~citablePerText);abline(coef(moLm)[1],coef(moLm)[2])
  mo = nls(rCitingOwn ~ a* exp(b*citablePerText), start = list (a=1, b =1))
  a = coef(mo)[1];b=coef(mo)[2]
  curve(a*exp(b*x), from=0,to=1,add = TRUE)
  p = as.double(predict(moLm))
  plot(p~rCitingOwn);abline(0,1)
  residuals = rCitingOwn-p
  plot(residuals~true.dates)
  par(o)
  
  ## Figure for the paper
  tns = as.character(info[,1])
  q = as.double(quantile(residuals, probs=seq(0,1, 0.1)))
  tns[residuals>=q[2] & residuals<=q[length(q)-1]] = ''
  df = data.frame(Time=true.dates,Residual=residuals, Text=tns, stringsAsFactors = FALSE)
  plt1 = ggplot(df,aes(x=Time,y=Residual)) + 
    geom_hline(yintercept = 0, linetype='dashed', colour='grey') +
    geom_point() + geom_smooth() + geom_text_repel(aes(label=Text)) + theme_bw()
  
  ####
  ### Citations: 2d plot
  # map each text to an Adamik period
  ixes=cut(true.dates, breaks=c(periodization$Start, periodization[nrow(periodization),]$End), labels = FALSE)
  w = which(res$Doc != res$Cit)
  ta = table( periodization[ixes[res$Doc[w]],]$Name, # citing documents
              periodization[ixes[res$Cit[w]],]$Name  # cited
  )
  lo = periodization$Start; hi = periodization$End; m = (lo+hi)/2
  # correct order of rows and columns
  maRow = match(rownames(ta), periodization$Name)
  maCol = match(colnames(ta), periodization$Name)
  ta = ta[order(maRow,decreasing=FALSE),]
  ta = ta[,order(maCol,decreasing=FALSE)]
  # remove spurious citations introduced by the dynamic re-calculation of times
  if(nrow(ta)==ncol(ta)){
    ta[upper.tri(ta)] = 0
  }else{
    # aaahhh ...
    for(i in 1:nrow(ta)){
      wc  = which(colnames(ta)==rownames(ta)[i])[1]
      if(wc < ncol(ta)){
        for(j in (wc+1):ncol(ta)){
          ta[i,j] = 0
        }
      }
    }
  }
  # divide each row by the total number of words in the epoch
  tc = table(periodization[ixes[res$Doc],]$Name)
  tc = tc[order(match(names(tc), periodization$Name),decreasing=FALSE)]#[2:length(tc)]
  # which values in tc are actually set in ta? Ex.: some models do not record citations 
  # between texts of the old period
  wz = which(is.na(match(names(tc),rownames(ta))))
  if(length(wz)>0){
    tc = tc[-wz]
  }
  stopifnot(all(names(tc)==rownames(ta)))
  ta = ta/as.integer(tc)
  w = which(ta>0, arr.ind = TRUE)
  xSrc = m[sort(maCol)[w[,2]]] # translates the indices into real dates
  xDst = m[sort(maRow)[w[,1]]]
  df = data.frame(xstart=xSrc,xend=xDst,ystart=rep(0,nrow(w)),yend=rep(1,nrow(w)), Weight=as.double(ta[w]))
  dfTxtSrc = data.frame(Text=colnames(ta), x=m[sort(maCol)], y=rep(-0.05,ncol(ta)), stringsAsFactors = FALSE)
  dfTxtDst = data.frame(Text=rownames(ta), x=m[sort(maRow)], y=rep( 1.05,nrow(ta)), stringsAsFactors = FALSE)
  plt2 = ggplot(df, aes(x=xstart, xend=xend, y=ystart, yend=yend, 
                        #colour=Weight, 
                        size=Weight)) +
    geom_segment() + 
    geom_text(data=dfTxtSrc, mapping=aes(x=x,y=y,label=Text), inherit.aes = FALSE) +
    geom_text(data=dfTxtDst, mapping=aes(x=x,y=y,label=Text), inherit.aes = FALSE) +
    theme_void() + theme(legend.position="none")
  return(list('plt1'=plt1,'plt2'=plt2))
}

# for Tab. 1: which authors are cited in each period, and which are the top words/bigrams?
cited.authors.per.period = function(res,periodization,true.dates,info)
{
  ixes=cut(true.dates, breaks=c(periodization$Start, periodization[nrow(periodization),]$End), labels = FALSE)
  ta = table(factor(res$Cit[res$Doc != res$Cit], levels=1:nrow(info)))
  df = data.frame(text=info[,1], text.id=1:nrow(info), cnt=as.integer(ta), period=ixes)
  
  s = split(df,f = factor(df$period))
  top.txt = 3
  top.wrd = 3
  authors = c()
  tab = data.frame(matrix(0,nrow=0,ncol=6))
  table.strings = c('\\hline\\hline')
  author.counts = c()
  for(i in 1:length(s))
  {
    su = s[[i]][order(s[[i]]$cnt,decreasing = TRUE),] # most frequently cited authors in this period
    for(j in 1:min(top.txt,nrow(su)))
    {
      w = which(res$Doc!=res$Cit & res$Cit==su[j,]$text.id)
      sta = table(factor(ixes[res$Doc[w]],levels=1:nrow(periodization)))
      if(sum(sta)>1000){
        authors = c(authors, as.character(su[j,]$text))
        author.counts = c(author.counts, length(which(res$Doc==su[j,]$text.id)))
        tab = rbind(tab, as.integer(sta))
      }
      # top uni-/bigrams for this text
      dfb = NULL; dfu = NULL
      if('Big' %in% colnames(res)){
        wu = which(res$Doc!=res$Cit & res$Cit==su[j,]$text.id & res$Big==1)
        ta = table(finfo$name[match(res$Fea[wu],finfo$id)])
        ds = data.frame(feature = as.character(names(ta)), cnt=as.integer(ta), stringsAsFactors = FALSE)
        dfu = ds[order(ds$cnt,decreasing=TRUE),]
        wb = which(res$Doc!=res$Cit & res$Cit==su[j,]$text.id & res$Big==2)
        ta = table(sprintf('%s %s', finfo$name[match(res$Fea[wb],finfo$id)],finfo$name[match(res$Fea[wb+1],finfo$id)]))
        dfb = data.frame(feature = as.character(names(ta)), cnt=as.integer(ta), stringsAsFactors = FALSE)
        ds = rbind(ds,dfb)
        dfb = dfb[order(dfb$cnt,decreasing = TRUE),]
        print(as.character(su[j,]$text))
        print(head(dfb,10))
      }else{
        wu = which(res$Doc!=res$Cit & res$Cit==su[j,]$text.id)
        ta = table(finfo$name[match(res$Fea[wu],finfo$id)])
        ds = data.frame(feature = as.character(names(ta)), cnt=as.integer(ta), stringsAsFactors = FALSE)
      }
      ds = ds[order(ds$cnt,decreasing = TRUE),]
      if(length(w)>1000){
        if(is.null(dfb)){
          ss = paste(ds[1:top.wrd,]$feature, collapse = ', ')
        }else{
          ss = sprintf('%s; %s', 
                       paste(dfu[1:top.wrd,]$feature, collapse = ', '),
                       paste(dfb[1:top.wrd,]$feature, collapse = ', '))
        }
        if(j==1){ # first text of this period
          table.strings = c(table.strings, '\\hline')
        }
        table.strings = c(table.strings,
                          sprintf('%s & %s \\\\', su[j,]$text, ss))
      }
    }
  }
  colnames(tab) = periodization$Name
  rownames(tab) = authors
  tab = tab[,3:ncol(tab)]
  return(list('tex.table' = table.strings, 'cnt.table' = tab, 
              'n.per.period' = as.integer(table(ixes[res$Doc])[2:5]),
              'n.per.author' = as.integer(author.counts) ))
}



build.citation.model.data.simple = function(params)
{
  
  info = read.delim(file = '../data/info.csv', header = FALSE, row.names = NULL, sep='\t', encoding = 'UTF-8')
  if(params$textIdTest!=0){
    ### testing temporal predictions:
    # set the date range of the tested text to that of the containing period
    periodization = read.delim(file = '../data/permanent/periodization-Adamik.csv', header = TRUE, row.names = NULL, sep='\t', encoding = 'UTF-8', stringsAsFactors = FALSE)
    mt = info[params$textIdTest,4]
    period = which(periodization$Start<=mt & periodization$End>=mt)
    if(length(period)>1){
      print('more than one period!')
      period = period[2]
    }
    info[params$textIdTest,5] = periodization$Start[period]
    info[params$textIdTest,6] = periodization$End[period]
    info[params$textIdTest,4] = mean(as.integer(info[params$textIdTest,c(5:6)]))
  }else if(params$useExactDates==FALSE){
    # each text is set to its Adamik period
    periodization = read.delim(file = '../data/permanent/periodization-Adamik.csv', 
                               header = TRUE, row.names = NULL, sep='\t', encoding = 'UTF-8', stringsAsFactors = FALSE)
    for(i in 1:nrow(info)){
      mt = info[i,4]
      period = which(periodization$Start<=mt & periodization$End>=mt)
      if(length(period)>1){
        print('more than one period!')
        period = period[2]
      }
      info[i,5] = periodization$Start[period]
      info[i,6] = periodization$End[period]
      info[i,4] = mean(as.integer(info[i,c(5:6)]))
    }
  }
  
  dat = fread(file  = sprintf('../data/input/bmm-input-%s-%d.dat',
                              params$pathAffix, params$nPerAuthor), 
              sep=' ', encoding = 'UTF-8', data.table = FALSE)
  
  texts = dat[,4]
  ut = unique(texts)
  stopifnot(length(ut)==nrow(info))
  
  if(params$model %in% c('ToCN')){
    # only lexical features, but in their original order
    Fea = as.integer(dat[,1])
    Doc = texts
    # do not remove rows with feat(lex)==0
    LexFea = as.integer(dat[,1])
    w = which(LexFea==0)
    if(length(w)>0){
      print(sprintf('removing %d records', length(w)), quote=FALSE)
      Fea = Fea[-w]
      Doc = texts[-w]
      LexFea = LexFea[-w]
    }
    ftab = table( Doc,LexFea)
    featNames = as.integer(colnames(ftab))
  }else{
    # nyi
  }
  
  
  # create a matrix of possible sources of citations
  # Note: This **must** be an int matrix, else the cpp doesn't work.
  dates = info[sort(unique(Doc)),4]
  dm = outer(dates,dates,'-')
  Conn = dm
  Conn[dm>=0] = 1 # if in test mode, this includes all texts from the same period
  Conn[dm<0] = 0
  diag(Conn) = 50 # this becomes the prior for self-citations during inference
  # sanity
  
  # initial citation assignments
  # TODO the code is super-slow. speed it up!
  
  Cit = rep(0, length(Doc))
  nn=1
  fea2col = rep(0, max(LexFea))
  uuf = as.integer(unique(LexFea))
  for(uf in uuf){
    fea2col[uf] = which(uf==featNames)
  }
  for(u in ut)
  {
    w = which(Doc==u)
    stopifnot(length(w)>0)
    # indices of possible sources
    # Attention: Not every source is valid. Have a look at the features.
    # A citation is only allowed if the lexical feature at position i is shared by
    # source and target texts.
    wx = which(Conn[u,]>0) 
    if(length(wx)==1){
      Cit[w] = wx[1]
    }else{
      for(ix in w){
        if(nn %% 100000==0){print(sprintf('%d|%d',nn,length(Doc)),quote=FALSE)}
        nn=nn+1
        # TODO replace the which(LexFea...) with one precalculated match
        wxx = as.integer(which(Conn[u,]>0 & ftab[,fea2col[LexFea[ix]] ]>0))
        if(length(wxx)==1){
          Cit[ix] = wxx[1]
        }else{
          p = Conn[u,wxx]
          Cit[ix] = sample(wxx,size=1, prob=p/sum(p))
        }
      }
    }
  }
  stopifnot(all(Cit>0))
  
  
  # initial time assignments
  Tim = rep(0, length(Doc))
  L = (info[,5]-DATE_LOW)/DATE_RANGE
  U = (info[,6]-DATE_LOW)/DATE_RANGE
  # uncertainty about the dates: prior = [mean-200,mean+200]
  time.bins = c(1:params$I)/params$I
  for(u in ut){
    w = which(Cit==u)
    stopifnot(length(w)>0)
    ll = L[u];uu = U[u]
    if(ll==uu){
      ll=ll-0.02;uu=uu+0.02
    }
    r = list('lo'=ll,'hi'=uu)
    p = 1e-3 + as.double(dunif(time.bins, min=r$lo, max=r$hi))
    Tim[w] = sample(x=params$I, size = length(w), replace=TRUE, prob = p/sum(p))
  }
  
  ### Dirichlet priors for the time
  time.bins = c(0:params$I)/params$I
  
  n = length(time.bins)
  tau = matrix(0, length(ut),params$I)
  for(i in 1:length(ut)){
    ll = L[i];uu = U[i]
    if(ll==uu){
      ll=ll-0.02;uu=uu+0.02
    }
    r = list('lo'=ll,'hi'=uu)
    a = punif(time.bins[2:n], r$lo,r$hi )
    a[n-1] = a[n-2]
    b = punif(time.bins[1:(n-1)], r$lo,r$hi )
    ab = as.double( a - b )
    if(sum(ab)==0){
      # can occur when too few bins are available
      ix = min(params$I, ceiling(0.5*(L[i]+U[i]) * params$I) )
      ab = rep(0,params$I)
      ab[ix] = 1
    }
    # This is important.
    # scale the probs so that the largest one is x
    # If the prior values are too small, the sampled values tend to form spikes,
    # i.e. concentrate too early on the corners of the simplex, see e.g. Neapolitan
    # usual values of x: 0.5, 1
    tau[i,] = ab * (1/max(ab))
    if(any(is.na(tau[i,]))){
      browser()
    }
  }
  
  ### Additional information
  if(params$model=='ToCN'){
    # uni- or bigrams
    Big = c(sample(x = c(1,2), size = length(Doc)-1, replace = TRUE, prob = c(0.95,0.05)),1)
    # make sure that bigrams do not cross text boundaries
    bi1 = which(Big==2)
    bi2 = bi1+1
    w = which(Doc[bi1]!=Doc[bi2])
    if(length(w)>0){
      Big[bi1[w]] = 1
    }
    return(list('Doc'=Doc, 'Fea'=Fea, 'Cit'=Cit, 'Tim'=Tim, 'Big'=Big, 'Conn'=Conn, 'tau'=tau))
  }
}

