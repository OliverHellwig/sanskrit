
## second step of the data preparation pipeline.
source('constants.R')
source('citation-model-functions.R')
library(data.table)


params = list(
  'I' = 150,
  'frequency.transformation' = '',
  'useExactDates' = TRUE,
  'model' = 'ToCN',
  # this text gets uninformative priors
  textIdTest = 0,#TextIdFredegar
  # which data?
  pathAffix = 'noStop-noFrequent-withCitations',
  specialAffix = '',
  nPerAuthor = 0
)

cdata = build.citation.model.data.simple(params)

path.gold = '../data/input/gold-time-slots.dat'
info = read.delim(file = '../data/info.csv', header = FALSE, row.names = NULL, sep='\t', encoding = 'UTF-8')
write.table(matrix(date2bin(info[,4],params$I),ncol=1), file = path.gold, quote = FALSE, row.names = FALSE, col.names = FALSE)


pathAffix = ifelse(params$useExactDates, sprintf('%s-exactDates',params$pathAffix), sprintf('%s-dateRanges',params$pathAffix))
pathAffix = sprintf('%s-%d-%d',pathAffix, params$nPerAuthor, params$I)
testAffix = ''
if(params$textIdTest>0){
  testAffix = sprintf('-%d',params$textIdTest)
}

path.assignments = sprintf('../data/input/assignments-%s-%s%s.dat',
                           params$model, pathAffix, testAffix)

#### ToCN
if(params$model %in% c('ToCN'))
{
  if(params$specialAffix==''){
    path.tau = sprintf('../data/input/tau-ToCN-%s%s.dat',
                       pathAffix, testAffix)
    path.assignments = sprintf('../data/input/assignments-ToCN-%s%s.dat',
                               pathAffix, testAffix)
  }else{
    p = sprintf('../data/input/%s', params$specialAffix)
    path.tau = sprintf(p,'tau')
    path.assignments = sprintf(p, 'assignments')
  }
  write.table(x=cdata$tau, 
              file=path.tau,
              quote = FALSE, sep = ' ', row.names = FALSE, col.names = FALSE)
  # we still need this for evaluating the citations
  write.table(x=cdata$Conn,
              file=sprintf('../data/input/conn-ToCN-%s%s.dat',
                           pathAffix, testAffix),
              quote = FALSE, sep = ' ', row.names = FALSE, col.names = FALSE)
  

  cat( sprintf('%s\n', paste(cdata$Doc, collapse=' ')), file = path.assignments, append=FALSE)
  cat( sprintf('%s\n', paste(cdata$Fea, collapse=' ')), file = path.assignments, append=TRUE)
  cat( sprintf('%s\n', paste(cdata$Cit, collapse=' ')), file = path.assignments, append=TRUE)
  cat( sprintf('%s\n', paste(cdata$Tim, collapse=' ')), file = path.assignments, append=TRUE)
  cat( sprintf('%s\n', paste(cdata$Big, collapse=' ')), file = path.assignments, append=TRUE)
}
