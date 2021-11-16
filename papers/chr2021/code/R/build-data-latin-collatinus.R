library(data.table)
library(stringr)
source('constants.R')

rem.single.text = function(xx, textids){
  ta = table(xx,textids)
  ta[ta>=1] = 1 # binarize the table
  w1 = which(rowSums(ta)==1)
  if(length(w1)>0){
    xx[xx %in% as.integer(rownames(ta)[w1])]=0
  }
  return(xx)
}

removeStopwords = TRUE
useMaxN = FALSE
pathAffix = ifelse(removeStopwords, 'noStop','withStop')
xAffix = ifelse(useMaxN, 'noFrequent', 'withFrequent')
citationSetting = 'withCitations' # 'noCitations'
pathAffix = sprintf('%s-%s-%s',pathAffix,xAffix,citationSetting)
nPerAuthor = 0
info = read.delim(file = '../data/info.csv', header = FALSE, row.names = NULL, sep='\t', encoding = 'UTF-8')

if(nPerAuthor==5000){
  nums = 1:4
}else{
  nums = 1:5
}

# read the individual collatinus files
sources = c()
lemmata = c()
for(num in nums){
  print(num, quote=FALSE)
  path = sprintf('../%s/coll-%d-%d.csv', data.dir.name, nPerAuthor, num)
  s = str_split(readLines(path), "\t", simplify = TRUE)
  l = sub('^([^ ]+)(, .+)*$', '\\1', s[,6])
  l[l=='unknown'] = '_'
  lemmata = c(lemmata,l)
  sources = c(sources,s[,4])
}

x = table(sources[lemmata=='_'])
df = data.frame(word=names(x), lemma=names(x),count=as.integer(x),stringsAsFactors = FALSE)
df = df[df$count>5,]
df = df[order(df$word,df$count,decreasing = TRUE),]
fwrite(df, file = '../data/collatinus-not-analyzed.csv', sep='\t', quote = FALSE, row.names = FALSE)

# the following is hacky.
text.ids = rep(0,length(sources))
text.ids[which(sources=='__TEXT___')] = 1
file.ids = rep(0, length(sources))
file.ids[which(sources=='__FILE___')] = 1
idt = idf = 0
for(i in 1:length(text.ids)){
  if(text.ids[i]==1){
    idt=idt+1
  }
  text.ids[i] = idt
  if(file.ids[i]==1){
    idf=idf+1
  }
  file.ids[i] = idf
}
stopifnot(max(text.ids)==nrow(info))



ta = table(lemmata)
minval = 5
if(nPerAuthor==0){
  minval = 30
}
if(useMaxN==FALSE){
  excludedWords = (names(ta)[which(ta<=minval)]) # keep high-frequency words
}else{
  wt = which(ta<=minval | ta>10000)
  excludedWords = (names(ta)[wt])
}


ids = as.integer(as.factor(lemmata))
ids[lemmata=='_'] = 0
ids[which((lemmata %in% excludedWords) | nchar(lemmata)<2)] = 0


dfnn = data.frame(lex.id=ids, lemma=lemmata, word=sources, text.id=text.ids, file.id=file.ids,stringsAsFactors = FALSE)
x = which(!(c(1:nrow(dfnn) %in% grep('^[a-zA-Z]+$', dfnn$lemma) )) & dfnn$lemma!='_' )
if(length(x)>0){
  dfnn = dfnn[-x,]
}

## !! refactor the lexical ids, in order to allow for smaller matrices in c++
x = sort(unique(dfnn$lex.id))
r = data.frame(old = x, new = 0:(length(x)-1))
dfnn$lex.id = r$new[match(dfnn$lex.id, r$old)]

# feature information
dup = duplicated(dfnn$lex.id)
L = dfnn$lemma[!dup]
I = dfnn$lex.id[!dup]
w0 = which(I==0)
if(length(w0)>0){ 
  I = I[-w0]
  L = L[-w0]
}
fInfo = data.frame(id=I, name=L, type=rep('lemma', length(I) ), stringsAsFactors = FALSE)
fInfo = fInfo[order(fInfo$id,decreasing = FALSE),]
write.table(x = fInfo, file = sprintf('../data/input/feature-bmm-%d-%s.info', nPerAuthor, pathAffix),
            quote = FALSE, sep = ' ', row.names = FALSE, col.names = TRUE)
w = which(fInfo$type=='lemma' & (fInfo$name %in% latinStopwords))
write.table(x = fInfo[w,]$id, file = sprintf('../data/input/stopword-%d.ids', nPerAuthor),
            quote = FALSE, sep = ' ', row.names = FALSE, col.names = FALSE)

write.table(x = dfnn, file = sprintf('../data/input/bmm-input-%s-%d.dat', pathAffix,nPerAuthor),
            quote = FALSE, sep = ' ', row.names = FALSE, col.names = FALSE)

