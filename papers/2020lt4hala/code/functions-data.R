library(RMySQL)
source('constants.R')

set.dates = function(ti,w,lo,hi,lvl){
  ti[w,]$date.l = lo
  ti[w,]$date.u = hi
  ti[w,]$mean.time = 0.5*(lo+hi)
  ti[w,]$layer.zehnder = lvl
  return(ti)
}

# outermost data loading wrapper; call this function from a script.
load.all.data = function(types, win.size, plot.text.id, rem.text.id=0)
{
  dat = load.data.and.text.info(types, win.size, top.n.pos2=200, top.n.pos3=300)
  txt.info = dat$text.info
  
  ds = dat$ds
  feat.cols = dat$feat.cols
  
  # all covered?
  # w1 = which(nchar(txt.info$layer.zehnder)>0)
  # w2 = which(txt.info$date.l <= (-500))
  # t1 = unique(txt.info[w1,]$text.id)
  # t2 = unique(txt.info[w2,]$text.id)
  # setdiff(t2,t1)
  
  
  w.rem = c()
  
  if(TRUE){
    # add the oldest layer of the Ramayana
    #browser()
    # w = which(txt.info$text.id==143 & txt.info$date.l==(-500))
    # if(length(w)>0){
    #   txt.info[w,]$layer.zehnder = '5-SU'
    # }
    w.rem = which(nchar(txt.info$layer.zehnder)==0)
	# 6 levels
    #dl  = c(-1500,  -1200,  -1000,  -900,   -700,   -600)
    #du  = c(-1200,  -1000,  -800,   -700,   -400,   -300)
	# 5 levels
	  dl  = DATES_LOWER_5
    du  = DATES_UPPER_5
    #stopifnot(length(unique(txt.info[-w.rem,]$layer.zehnder))==length(dl))
    xx = data.frame(cbind.data.frame(level=levels.zehnder, lo=dl, hi=du), stringsAsFactors = FALSE)
    xx$level = as.character(xx$level)
    for(i in 1:nrow(xx)){
      w = which(txt.info$layer.zehnder==xx[i,]$level)
      txt.info = set.dates(txt.info, w, xx[i,]$lo, xx[i,]$hi, xx[i,]$level)
    }
    # set the dates for plot.text.id to an uninformed prior
    if(plot.text.id==450){ # RV
      txt.info = set.dates(txt.info, which(txt.info$text.id==plot.text.id), dl[1], du[2], '1-RV')
    }else if(plot.text.id==464){
      txt.info = set.dates(txt.info, which(txt.info$text.id==plot.text.id), dl[2], du[3], '2-MA')
    }else if(plot.text.id==471){
      txt.info = set.dates(txt.info, which(txt.info$text.id==plot.text.id), -900,-700, '4-PO')
    }
    # if required, remove a single text from the data
    if(rem.text.id>0){
      w.rem = unique( c(w.rem, which(txt.info$text.id==rem.text.id)) )
    }
  }else{
    # w.rem = which(txt.info$date.l==0 & txt.info$date.u==0 | txt.info$text.id %in% c(143,154))
    # if(length(w.rem)>0){
    #   ds = ds[-w.rem,]
    #   txt.info = txt.info[-w.rem,]
    # }
    # w.rem = which(txt.info$date.l > (-500))
  }
  if(length(w.rem)>0){
    ds = ds[-w.rem,]
    txt.info = txt.info[-w.rem,]
    print(sprintf('Removed %d records', length(w.rem)), quote = FALSE)
  }
  # zero columns ... don't do it here!
  # w0 = which(colSums(ds)==0)
  # if(length(w0)>0){
  #   ds = ds[,-w0]
  #   print(sprintf('Removed %d zero-columns', length(w0)), quote = FALSE)
  # }
  
  # observed data = linguistic features
  X = as.matrix(ds[,feat.cols] )
  X = X[,-1] # this feature is meaningless
  return(list('X' = X, 'txt.info' = txt.info))
}



# creates compound length features
compound.lens = function(d,F)
{
  cpd = d$CAS
  cpd[cpd!='cpd'] = ''
  r = rle(cpd)
  cs = cumsum(r$lengths)
  cpd.num = rep(0, nrow(d))
  wc = which(r$values=='cpd')
  for(w in wc){
    n = r$lengths[w]+1
    if(n>5 & n <= 10){
      n = 6
    }else if(n>10){
      n = 7
    }
    cpd.num[cs[w] ] = n
  }
  wp = which(cpd.num>0)
  cpd.num[wp] = sprintf('cpd%d', cpd.num[wp])
  cpd.num[-wp] = ''
  tmp = table(F, cpd.num)
  tmp = tmp[,-which(colnames(tmp)=='')]
  return(tmp)
}

# Creates POS bigrams.
# @param F The grouping factor for the text windows.
pos.2.features = function(d, F, top.n = 0, max.freq=100, ignore.sentence.breaks=TRUE)
{
  n = nrow(d)
  #browser()
  f1 = c();f2=c(); x1=c();x21=c();x22=c()
  if(ignore.sentence.breaks==TRUE)
  {
    L1 = c(1, d[1:(n-1),]$LINEID)
    L2 = d$LINEID
    p1 = c('_', d[1:(n-1),]$POS)
    p2 = d$POS
    w  = which(L1==L2 & F>(-1) & p1!='_') # safe cases
    x1 = sprintf("%s_%s", p1[w], p2[w])
    f1 = F[w]
  }else{
    L1 = c(1, d[1:(n-1),]$LINEID)
    L2 = d$LINEID
    p1 = c('BOS', d[1:(n-1),]$POS)
    p2 = d$POS
    
    df = data.frame(cbind(p1,p2,L1,L2,F), stringsAsFactors = FALSE)
    w  = which(df$L1==df$L2 & df$F>(-1) ) # safe cases
    x1 = sprintf("%s_%s", df[w,]$p1, df[w,]$p2)
    f1 = df[w,]$F
    w  = which(df$L1!=df$L2 & df$F>(-1) ) # with BOS/EOS markers
    x21 = sprintf("%s_EOS", df[w,]$p1)
    x22 = sprintf("BOS_%s", df[w,]$p2)
    f2 = df[w,]$F
  }
  ta = table(c(f1,f2,f2), c(x1,x21,x22))
  cs = colSums(ta)
  df = data.frame(cbind(feat=names(cs), n=as.integer(cs)), stringsAsFactors = FALSE)
  df$n = as.integer(df$n)
  df = df[order(df$n, decreasing = TRUE),]
  if(top.n>0){
    sel = 1:top.n
  }else{
    sel = which(df$n>=max.freq)
  }
  w = which(colnames(ta) %in% df[sel,]$feat)
  return(ta[,w])
}

# Creates POS trigrams from the input data d
# @param F The grouping factor for the text windows.
pos.3.features = function(d, F, top.n = 0, max.freq=100, add.case.info = FALSE, use.ngram.eos = FALSE, ignore.sentence.breaks=TRUE)
{
  #browser()
  n = nrow(d)
  x21 = c(); x22 = c(); x31=c();x32=c();x33=c(); f2 =c();f3=c()
  # for a detailed inspection of the results
  ngrams = rep('_', nrow(d)); words = rep('_', nrow(d)); info = rep('_', nrow(d))
  
  if(ignore.sentence.breaks==TRUE){
    # This branch takes each chapter as one continuous text.
    # Currently (Nov. 2019) used.
    id1 = F[[1]]
    L1 = c(id1,id1, F[1:(n-2)])
    L2 = c(id1, F[1:(n-1)] )
    L3 = F
    # collect the features
    features = d$POS
    if(add.case.info==TRUE){
      w = which(d$CAS!='_' & d$POS %in% c('NC', 'JJ', 'PPR', 'PRQ', 'PRD', 'PRI', 'JQ'))
      features[w] = sprintf("%s.%s", d[w,]$POS, d[w,]$CAS)
    }
    p1 = c('_', '_', features[1:(n-2) ])
    p2 = c('_', features[1:(n-1) ])
    p3 = features
    wrds = d$LEMMA
    w1 = c('_', '_', wrds[1:(n-2)])
    w2 = c('_', wrds[1:(n-1)])
    w3 = wrds
    df = data.frame(cbind(p1,p2,p3, w1,w2,w3, L1,L2,L3, F), stringsAsFactors = FALSE)
    w  = which(df$L1==df$L2 & df$L1==df$L3 & df$F>(-1) & df$p1!='_' & df$p2!='_' ) # safe cases
    f1 = df[w,]$F
    x1 = sprintf("%s_%s_%s", df[w,]$p1, df[w,]$p2, df[w,]$p3)
    ngrams[w] = x1
    words[w]  = sprintf("%s_%s_%s", df[w,]$w1, df[w,]$w2, df[w,]$w3)
    info = c(0,0, d$LINEID[1:(n-2)])
  }
  else
  {
    # this branch uses dandas and sentence breaks as delimiters of sequences
    # we insert a new sentence mark for new text lines, and if a manual sentence 
    # break occurs. To synchronize both, the sentence breaks need to be shifted
    # by one position.
    # nice, but currently not used.
    L = d$LINEID
    P = d$PUNC # sentence breaks
    P[P==1] = 0 # comma -> nil
    w = which(P==2)
    ws = w+1
    wx = which(ws>length(P))
    if(length(wx)>0){ ws = ws[-wx]}
    P[wx] = 2
    P[w]  = 0
    p1 = P[[1]]
    id1 = d[1,]$LINEID
    L1 = paste( c(id1,id1, L[1:(n-2)]), c(p1,p1, P[1:(n-2)]), sep = '+' )
    L2 = paste( c(id1, L[1:(n-1)] ), c(p1, P[1:(n-1)]), sep = '+')
    L3 = paste( L, P, sep='+')
    #browser()
    # collect the features
    features = d$POS
    if(add.case.info==TRUE){
      w = which(d$CAS!='_' & d$POS %in% c('NC', 'JJ', 'PPR', 'PRQ', 'PRD', 'PRI', 'JQ'))
      features[w] = sprintf("%s.%s", d[w,]$POS, d[w,]$CAS)
    }
    p1 = c('BOS', 'BOS', features[1:(n-2) ])
    p2 = c('BOS', features[1:(n-1) ])
    p3 = features
    df = data.frame(cbind(p1,p2,p3, L1,L2,L3, F), stringsAsFactors = FALSE)
    w  = which(df$L1==df$L2 & df$L1==df$L3 & df$F>(-1) ) # safe cases
    x1 = sprintf("%s_%s_%s", df[w,]$p1, df[w,]$p2, df[w,]$p3)
    f1 = df[w,]$F
    ng1[w] = x1
    
    if(use.ngram.eos==TRUE)
    {
      w  = which( df$L1!=df$L2 & df$F>(-1) ) # with BOS/EOS markers
      x21 = sprintf("%s_EOS_EOS", df[w,]$p2)
      x22 = sprintf("BOS_BOS_%s", df[w,]$p3)
      f2 = df[w,]$F
      ng1[w] = x21; ng2[w] = x22
      w = which( df$L1==df$L2 & df$L2!=df$L3 & df$F>(-1) )
      x31 = sprintf("%s_%s_EOS", df[w,]$p1, df[w,]$p2)
      x32 = sprintf("%s_EOS_EOS", df[w,]$p2)
      x33 = sprintf('BOS_BOS_%s', df[w,]$p3)
      f3 = df[w,]$F
      ng1[w] = x31;ng2[w]=x32;ng3[w]=x33
    }
  }
  # save in a file, for evaluation
  write.utf8.data.frame.transposed(data.frame(cbind.data.frame(ngrams,words,info),stringsAsFactors = FALSE),
                        sprintf('../%s/ngrams.dat', data.dir.name), '\t')
  #browser()
  ta = table(c(f1,f2,f2,f3,f3,f3), c(x1,x21,x22,x31,x32,x33))
  cs = colSums(ta)
  df = data.frame(cbind(feat=names(cs), n=as.integer(cs)), stringsAsFactors = FALSE)
  df$n = as.integer(df$n)
  df = df[order(df$n, decreasing = TRUE),]
  if(top.n>0){
    sel = 1:top.n
  }else{
    sel = which(df$n>=max.freq)
  }
  w = which(colnames(ta) %in% df[sel,]$feat)
  return(ta[,w])
}

write.utf8.data.frame = function(df, path, sep='\t')
{
  cat( iconv( sprintf('%s\n', paste(colnames(df), collapse = sep)), to='UTF-8')  , file = path, append = FALSE)
  m=apply(df,1,function(row) cat( iconv( sprintf('%s\n', paste(row, collapse = sep)), to='UTF-8')  , file = path, append = TRUE)  )
}

# faster for huge data frames.
write.utf8.data.frame.transposed = function(df, path, sep='\t')
{
  #cat( iconv( sprintf('%s\n', paste(colnames(df), collapse = sep)), to='UTF-8')  , file = path, append = FALSE)
  #apply(df,1,function(row) cat( iconv( sprintf('%s\n', paste(row, collapse = sep)), to='UTF-8')  , file = path, append = TRUE)  )
  cat( iconv( sprintf('%s\n', paste(df[,1], collapse = sep)), to='UTF-8')  , file = path, append = FALSE)
  for(i in 2:ncol(df)){
    cat( iconv( sprintf('%s\n', paste(df[,i], collapse = sep)), to='UTF-8')  , file = path, append = TRUE)
  }
}

tbl = function(sec,fea)
{
  return(table(factor(sec, levels=sort(unique(sec))), fea))
}

# main function for building all types of linguistic feature distributions.
build.data = function(types, win.size, top.n.pos2 = 0, max.pos2=100, top.n.pos3 = 0, max.pos3=100, use.ngram.eos = FALSE, ignore.sentence.breaks=TRUE)
{
  
  s = paste(types, collapse='-')
  path = sprintf('../%s/%s-%d.dat', data.dir.name, s, win.size)
  if(file.exists(path)){
    ds = read.delim(file = path, header = TRUE, row.names = NULL, sep=' ')
    return(ds)
  }
  # these data are built with python/build_data.py
  d = read.delim(file = sprintf('../%s/all-data.dat', data.dir.name), header = TRUE, row.names=NULL,
                 sep = ' ', quote = '', stringsAsFactors = FALSE, encoding = 'UTF-8',
                 nrows = -1)
  
  # We are only interested in the Vedic subcorpus ->
  # Remove later texts.
  lyrs = load.layer.info()
  d = d[which(d$CHID %in% lyrs$ch.id),]
  
  # assign readable names to the cases
  cases = c('cpd', "nom", "voc", "acc", "ins", "dat", "abl", "gen", "loc")
  cas = rep('_', nrow(d))
  cas.cpy = d$CAS
  for(i in 1:length(cases)){
    w = which(d$TMF==0 & 
                (d$TMINF==0 | d$TYPEINF=='ppp' | d$TYPEINF=='ger' | d$TYPEINF=='par' | d$TYPEINF=='ppa') & 
                d$POS!='CADP' & # post-annotation in python
                d$CAS==(i-1) & d$NUM!=254 & d$GEN!=254 & d$GEN>0)
    cas[w] = cases[[i]]
  }
  d$CAS = cas
  
  X = split(d, d$TEXTID)
  
  # Create the text windows, whose ids are stored in F
  # Sections with less than win.size words get F = -1
  F = c() 
  offset = 0
  for(x in X){
    n = ceiling(nrow(x)/win.size)
    f = rep((1+offset):(offset+n), each=win.size)
    F = c(F, f[1:nrow(x)])
    offset = offset+n
  }
  ta = table(F)
  w = which(ta < win.size)
  if(length(w)>0){
    F[which(F %in% as.integer(names(ta)[w]) )] = -1
  }
  # invalid case annotations, skip these sections completely
  ta = table(F[cas.cpy==254 | d$NUM==254 | d$GEN==254])
  groups = as.integer(names(ta)[which(ta>1)])
  if(length(w)>0){
    F[which(F %in% groups)] = -1
  }
  #browser()
  Feats = list()
  Feat.types = list()
  W = which(F>(-1))
  if('compounds' %in% types){
    Feats[[length(Feats)+1]] = compound.lens(d[W,],F[W])
    Feat.types[length(Feat.types)+1] = 'compounds'
  }
  if('deriv' %in% types){
    # derivational morphology
    Feats[[length(Feats)+1]] = table(F[W], d$DERIV[W])
    Feat.types[length(Feat.types)+1] = 'deriv'
  }
  if('case' %in% types){
    # get: inflected nouns, adjectives and verbal participles
    Feats[[length(Feats)+1]] = table(F[W], d$CAS[W])
    Feat.types[length(Feat.types)+1] = 'case'
  }
  if('tense-mode' %in% types){
    Feats[[length(Feats)+1]] = table(F[W], sprintf("%s_%s", d$TENSEF[W], d$MODEF[W]))
    Feat.types[length(Feat.types)+1] = 'tensemood'
  }
  if('tense' %in% types){
    Feats[[length(Feats)+1]] = table(F[W], d$TENSEF[W])
    Feat.types[length(Feat.types)+1] = 'tense'
  }
  if('infinite' %in% types){
    # suffixes
    Feats[[length(Feats)+1]] = tbl(F[W], d$FORMINF[W])
    Feat.types[length(Feat.types)+1] = 'inf-suffix'
    # participles
    x = sprintf('inf-%s-%s',d[W,]$TENSEINF, d[W,]$TYPEINF)
    ta = tbl(F[W], x[W])
    nc = ncol(ta)
    ta = ta[,-which(colnames(ta) %in% c('inf-_-_', 'inf-_-abs', 'inf-_-ger', 'inf-_-inf', 'inf-_-ppa', 'inf-_-ppp') )]
    stopifnot(ncol(ta)==(nc-6))
    Feats[[length(Feats)+1]] = ta
    Feat.types[length(Feat.types)+1] = 'inf-participle'
  }
  if('etym' %in% types){
    #browser()
    Feats[[length(Feats)+1]] = table(F[W], d$ETYM[W])
    Feat.types[length(Feat.types)+1] = 'etym'
  }
  if('vclass' %in% types){
    # distinguishable present stem formations
    Feats[[length(Feats)+1]] = table(F[W], d$VCLASS[W])
    Feat.types[length(Feat.types)+1] = 'vclass'
  }
  if('pos' %in% types){
    Feats[[length(Feats)+1]] = table(F[W], d$POS[W])
    Feat.types[length(Feat.types)+1] = 'pos'
  }
  if('pos2' %in% types){
    Feats[[length(Feats)+1]] = pos.2.features(d, F, top.n.pos2, max.pos2, ignore.sentence.breaks)
    Feat.types[length(Feat.types)+1] = 'pos2'
  }
  if('pos3' %in% types){
    Feats[[length(Feats)+1]] = pos.3.features(d,F, top.n.pos3, max.pos3, add.case.info = TRUE, use.ngram.eos,ignore.sentence.breaks)
    Feat.types[length(Feat.types)+1] = 'pos3'
  }
  if('lexicon' %in% types){
    # con = dbConnect(MySQL(), user="dcs", password="dcs", dbname='dcs', host="localhost")
    # dbSendQuery(con, 'SET NAMES UTF8;')
    # rs = dbSendQuery(con, 'SELECT lexicon.word, word_references.pos_tag, word_references.lexicon_id, COUNT(word_references.id) AS cnt
    #   	FROM word_references INNER JOIN lexicon ON word_references.lexicon_id=lexicon.id 
    #                   INNER JOIN text_lines ON word_references.sentence_id=text_lines.id
    #                   INNER JOIN chapters ON text_lines.chapter_id=chapters.id
    #                   WHERE chapters.date_lower < (-300) AND word_references.pos_tag IN ("CEM", "CCM", "CNG", "CCD", "CSB", "CAD", "CX")
    #                   GROUP BY word_references.lexicon_id
    #                   HAVING cnt>=100
    #                   ORDER BY cnt')
    # res = fetch(rs,n = -1)
    # dbClearResult(rs)
    # dbDisconnect(con)
    # wids = c(); lemmata = c()
    # for(i in 1:nrow(res)){
    #   wids = c(wids, res[i,3])
    #   lemmata = c(lemmata, res[i,1])
    # }
    # #browser()
    # wrds = d$LEMMAID
    # w = which(wrds %in% wids)
    # wrds[-w] = '_'
    # m = match(wrds[w], table = wids)
    # wrds[w] = sprintf('%d_%s', wids[m], lemmata[m])
    # Feats[[length(Feats)+1]] = table(F[W], wrds[W])
    # Feat.types[length(Feat.types)+1] = 'lex'
    #n.words = 1000
    max.freq = 10000
    min.freq = 30
    ta = table(d$LEMMAID)
    df = data.frame(cbind(id=as.integer(names(ta)), cnt=as.integer(ta)))
    df = df[order(df$cnt,decreasing = TRUE),]
    w = which(df$id==0)
    if(length(w)>0){
      df = df[-w,]
    }
    ta = table(F[W], d[W,]$LEMMAID)
    sel = which(df$cnt<=max.freq & df$cnt>=min.freq)
    #Feats[[length(Feats)+1]] = ta[,which(as.integer(colnames(ta)) %in% df[1:n.words,]$id )]
    Feats[[length(Feats)+1]] = ta[,which(as.integer(colnames(ta)) %in% df[sel,]$id )]
    Feat.types[length(Feat.types)+1] = 'lex'
  }
  # some sanity checks
  # plus: bring all records in the correct order
  secs = sort(unique(F[F>(-1)]))
  nr = c()
  ta = 0
  feat.types = c()
  all.col.names = c()
  #browser()
  for(i in 1:length(Feats)){
    nr = c(nr, nrow(Feats[[i]]))
    w = which(colnames(Feats[[i]])=='_' | colnames(Feats[[i]])=='___')
    if(length(w)>0){
      Feats[[i]] = Feats[[i]][,-w]
      print(sprintf('Removed %d columns for feature %s', length(w), types[[i]]), q=F)
    }
    rn = as.integer(rownames(Feats[[i]]))
    ma = match(secs, table=rn)
    Feats[[i]] = Feats[[i]][ma,]
    if(i==1){
      ta = Feats[[i]]
    }else{
      stopifnot(all(rownames(ta)==rownames(Feats[[i]])))
      ta = cbind(ta, Feats[[i]])
    }
    all.col.names = c(all.col.names, colnames(Feats[[i]]))
    feat.types = c(feat.types, rep(Feat.types[i], ncol(Feats[[i]])))
  }
  stopifnot(length(unique(nr))==1)
  #browser()
  
  ta = data.frame(cbind(ta), stringsAsFactors = FALSE)
  # note: An 'f.' MUST be prefixed, otherwise the column selection does not work.
  colnames(ta) = sprintf("f.%s.%s", feat.types, all.col.names)
  s = which(!duplicated(F) & F>(-1))
  stopifnot(length(s)==nrow(ta))
  
  if(any(F[s]!=as.integer(row.names(ta))) ){
    print('warning: Reordering the data matrix!')
    m = match(x = as.integer(row.names(ta)), table = F[s])
    ta[m,] = ta
    # for some reason, the row names are not reordered here
  }
  t.ids = d[s,]$TEXTID
  c.ids = d[s,]$CHID
  c.ids.end = d[s+win.size-1,]$CHID
  l = d[s,]$DATEL
  u = d[s,]$DATEU
  # some data types include _ as a non-applicable type (e.g. tenses)
  # wn = which(colnames(ta)=='_' | colnames(ta)=='___')
  # if(!(type %in% c('tense-mode-pos2', 'case-tense-mode-pos2', 'case-tense-mode-pos3'))){
  #   if(length(wn)==1){
  #     colnames(ta)[wn]='rest'
  #   }else if(length(wn)==0){
  #     ta = cbind(ta, rest=win.size-rowSums(ta))
  #   }
  # }
  ds = data.frame(cbind( ta, text.id=t.ids, ch.id=c.ids, ch.id.end = c.ids.end, date.l=l, date.u=u), stringsAsFactors = FALSE)
  
  #names(ds)[1:ncol(ta)] = sprintf('f.%s', colnames(ta) )
  write.table(x = ds, file = path, quote = FALSE, sep = ' ', row.names = FALSE, col.names = TRUE)
  return(ds)
}

# loads pre-generated data: Which chapters belong to the Vedic subcorpus?
load.layer.info = function()
{
  path = '../data/vedic-layers.csv'
  lyrs = read.delim(file = path, header = TRUE, row.names=NULL,sep = '\t', quote = '', stringsAsFactors = FALSE, encoding = 'UTF-8')
  lyrs$ch.id = as.integer(lyrs$ch.id)
  return(lyrs)
}

# main wrapper function for accessing the data.
load.data.and.text.info = function(type, win.size=250, top.n.pos2 = 100, top.n.pos3 = 100)
{
  ds = build.data(type, win.size, top.n.pos2, top.n.pos3)
  
  cn = c('text.id', 'ch.id', 'ch.id.end', 'date.l', 'date.u')
  text.info = ds[,colnames(ds) %in% cn]
  text.info = cbind.data.frame(text.info, 
                               mean.time = 0.5*(ds$date.l+ds$date.u), 
                               layer = rep('', nrow(text.info)),
                               layer.zehnder = rep('', nrow(text.info)),
                               text.name = rep('', nrow(text.info)),
                               chapter.name.start = rep('', nrow(text.info)),
                               chapter.name.end = rep('', nrow(text.info)),
                               book = rep(0, nrow(text.info)) )
  text.info$layer = as.character(text.info$layer)
  text.info$text.name = as.character(text.info$text.name)
  text.info$layer.zehnder = as.character(text.info$layer.zehnder)
  text.info$chapter.name.start = as.character(text.info$chapter.name.start)
  text.info$chapter.name.end = as.character(text.info$chapter.name.end)
  wf = grep(names(ds), pattern = '^f\\..+$')
  
  
  # 
  lyrs = load.layer.info()
  # some information
  con = dbConnect(MySQL(), user="dcs", password="dcs", dbname='dcs', host="localhost")
  dbSendQuery(con, 'SET NAMES UTF8;')
  #################
  # create the stratifications
  #################
  for(i in 1:nrow(text.info)){
    w = which(lyrs$ch.id==text.info[i,]$ch.id)
    if(length(w)==1){
      text.info[i,]$layer = lyrs[w,]$layer.internal
      text.info[i,]$layer.zehnder = lyrs[w,]$layer.zehnder
      text.info[i,]$book = lyrs[w,]$book
    }else if(length(w)>1){
      browser()
    }
  }
  
  
  # rest: take the name of the text as the layer
  tids = unique(text.info$text.id)
  sql = sprintf("SELECT ID, (CASE WHEN Kuerzel='' OR Kuerzel IS NULL THEN Textname ELSE Kuerzel END) AS name FROM texts WHERE ID IN (%s)", 
                paste(tids, collapse=','))
  rs   = dbSendQuery(con, sql)
  res = fetch(rs, n=-1)
  dbClearResult(rs)
  rs = dbSendQuery(con, 'SELECT id,name FROM chapters')
  res.ch = fetch(rs,n=-1)
  dbClearResult(rs)
  ml = match(text.info$ch.id, table=res.ch[,1])
  text.info$chapter.name.start = res.ch[ml,2]
  ml = match(text.info$ch.id.end, table = res.ch[,1])
  text.info$chapter.name.end = res.ch[ml,2]
  for(i in 1:nrow(res)){
    w = which(text.info$text.id==res[i,1])
    text.info[w,]$text.name = res[i,2]
  }
  
  dbDisconnect(con)
  return(list('ds' = ds, 'text.info' = text.info, 'feat.cols' = wf))
}


# Initialization of the hidden time variable
assign.random.times = function(dp,sd.param,X,N,I,method, add.rnd=c(1e-5,1e-3))
{
  T = matrix(0, nrow=N, ncol=I)
  if(method=='kmeans'){
    #initial topic assignments with k-means
    sa.i = as.integer( kmeans(x=X, centers=I)$cluster)
    for(n in 1:N){
      T[n,sa.i[[n]] ] = 1
    }
  }else if(method=='normal'){
    m = 0.5*(dp$L+dp$U) 
    s =  (dp$U-m)/sd.param
    sa.i = rep(0,N)
    for(n in 1:N){
      r = max(min(rnorm(1, mean=m[[n]], sd=s[[n]] ), max(dp$U)),0) / max(dp$U)
      r = as.integer(ceiling(r*I))
      T[n, r ] = 1
      sa.i[[n]] = r
    }
  }else if(method=='unif'){
    s = runif(n = N,min = dp$L, max = dp$U)/max(dp$U)
    sa.i = rep(0,N)
    for(n in 1:N){
      r = as.integer(ceiling(s[[n]]*I))
      T[n, r ] = 1
      sa.i[[n]] = r
    }
  }else if(method=='random'){
    # fully random
    sa.i = as.integer( sample(I,size = K, replace=TRUE))
    for(n in 1:N){
      T[n,sa.i[[n]] ] = 1
    }
  }else if(method=='mean'){
    # just set to the mean of L and U
    sa.i = as.integer(ceiling(0.5*(dp$L+dp$U)/max(dp$U) * I))
    for(n in 1:N){
      T[n,sa.i[n]] = 1
    }
  }else{
    stopifnot(FALSE)
  }
  if(add.rnd[[1]]>0 & add.rnd[[2]]>add.rnd[[1]]){
    for(n in 1:N){
      T[n,] = T[n,] + runif(I,min=add.rnd[[1]], max=add.rnd[[2]])
      T[n,] = T[n,]/sum(T[n,])
    }
  }
  return(list('T'=T, 'sa.i'=sa.i))
}


# @param I number of time slots
get.date.priors = function(t.info, I, sd.parm, method='gaussian')
{
  L = (t.info$date.l+1500)/3500
  U = (t.info$date.u+1500)/3500
  N = nrow(t.info)
  # log prior probabilities of the dates
  p.T = matrix(0, nrow=N, ncol=I) 
  time.bins = max(U) * (c(0:(I-1))/I)
  eps.T = 1e-5
  m = 0.5*(L+U) 
  S =  (U-m)/sd.parm
  if(method=='gaussian'){
    for(n in 1:N){
      r = eps.T + dnorm(x = time.bins, mean=m[n], sd=S[n] )
      p.T[n,] = log( r/sum(r) )
    }
  }else if(method=='unif'){
    l = ceiling(L/max(U) * I)
    l[l==0] = 1
    u = ceiling(U/max(U) * I)
    ll = ifelse(l==1, l, l-1)
    uu = ifelse(u==I, u, u+1)
    for(n in 1:N){
      r = rep(eps.T, I)
      r[ll[n]:uu[n]] = 1
      r[l[n]:u[n]] = 5
      p.T[n,] = log( r/sum(r) )
    }
  }
  return(list('L'=L, 'U'=U, 'p.T' = p.T, 'time.bins'=time.bins, 'm' = m, 's' = S))
}

get.dirichlet.date.priors = function(t.info, I, sd.parm, group)
{
  groups = rep(0, nrow(t.info))
  if(group=='layer'){
    G = length(levels.zehnder)
    for(lvl in 1:G){
      w = which(t.info$layer.zehnder==levels.zehnder[lvl])
      groups[w] = lvl
    }
  }else if(group=='text'){
    u = unique(t.info$text.id)
    G = length(u)
    for(lvl in 1:G){
      w = which(t.info$text.id==u[lvl])
      groups[w] = lvl
    }
  }else if(group=='section'){
    u = c(1:nrow(t.info))
    G = length(u)
    for(lvl in 1:G){
      groups[lvl] = lvl
    }
  }
  L = (t.info$date.l - DATE_LOW)/DATE_RANGE; U = (t.info$date.u - DATE_LOW)/DATE_RANGE
  #browser()
  m = 0.5*(L+U) 
  s =  (U-m)/sd.parm
  time.bins = c(0:I)/I
  n = length(time.bins)
  # prior probabilities of the dates, precalculated, remain constant
  tau = matrix(0, G,I)
  tau.unif = matrix(0.001, G,I)
  #eps.T = 0
  for(lvl in 1:G){
    w = which(groups==lvl)
    tau[lvl,] = as.double( pnorm(time.bins[2:n], mean=mean(m[w]), sd = mean(s[w])) - 
                             pnorm(time.bins[1:(n-1)], mean=mean(m[w]), sd = mean(s[w]) ) )
  }
  
  return(list('L'=L, 'U'=U, 'G'=G, 'groups'=groups, 
              'tau' = tau, 'tau.unif' = tau.unif, 'time.bins' = time.bins,
              'mean' = m, 'sd' = s))
}

# set some constant (e.g. result of running a script) that is later included in a tex file
set.tex.constant = function(key, value)
{
  path = sprintf('%s/constants.dat', con.dir)
  comment = readLines(path, encoding='UTF-8')[1] # comment!
  lns = read.delim(file = path, header = FALSE, row.names = NULL, sep='\t', comment.char = '%', fileEncoding = 'UTF-8')
  lns[,1] = as.character(lns[,1])
  lns[,2] = as.character(lns[,2])
  w = which(lns[,1]==key)
  success = TRUE
  if(length(w)==1){
    lns[w,2] = value
  }else if(length(w)==0){
    lns = rbind(lns, c(key,value))
  }else{
    print(sprintf('Error while loading the constants file: multiple entries for key %s', key))
    success = FALSE
  }
  y = apply(lns, 1, function(x)paste(x, collapse='\t'))
  writeLines(c(comment,y), path)
  y = apply(lns, 1, function(x)sprintf('\\newcommand{\\%s}{%s}', x[1],x[2]))
  writeLines(y, sprintf('%s/constants.tex', con.dir) )
}

# param x X element produced by the function @load.all.data
build.gibbs.data = function(x,tau,txt.i, transform = 'none')
{
  # handle one-row data = text consists of only one section
  if(is.null(dim(x))){
    if(length(x)>0){
      x = matrix(x, nrow=1, byrow=TRUE)
    }
  }
  if(is.null(dim(tau)) & length(tau)>0){
    tau = matrix(tau, nrow=1, byrow=TRUE)
  }
  stopifnot(nrow(x)==nrow(txt.i))
  text.and.layer = sprintf("%d-%s", txt.i$text.id, txt.i$layer.zehnder)
  N = nrow(x)
  K = ncol(x)
  # unfolded features
  F = c()
  Gr = c() # groups
  # hidden assignments
  T = c()
  S = c()
  # initial assignments
  for(n in 1:N){
    r = as.integer(x[n,])
    if(transform=='sqrt'){
      r = as.integer(ceiling(sqrt(r)))
    }
    rs = sum(r) # how many features in this document? Number may vary
    F = c(F, rep(1:K, r) )
    S = c(S, sample(x = params$J, size = rs, replace=TRUE) )
    T = c(T, sample(x = params$I, size = rs, replace = TRUE, prob = tau[n,]/sum(tau[n,])))
    Gr = c(Gr, rep(n, rs))
    #Gr.orig = c(Gr.orig, rep(gr.orig[n], rs) )
    #browser()
    gp = rep(n-1,rs)
    if(n==1){
      gp = rep(n,rs)
    }else if(text.and.layer[n]!=text.and.layer[n-1]){
      # todo use || instead?
      gp = rep(n,rs)
    }
    gn = rep(n+1,rs)
    if(n==N){
      gn = rep(n,rs)
    }else if(text.and.layer[n]!=text.and.layer[n+1]){
      gn = rep(n,rs)
    }
  }
  return(list('T'=T, 'S' = S, 'F' = F, 'Gr' = Gr)) #, 'Gr.orig' = Gr.orig))
}