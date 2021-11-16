library(data.table)
library(ggplot2)
library(ggrepel)
source('constants.R')
source('citation-model-functions.R')

nPerAuthor = 0
I = 150 # no of time slots
pathAffixRes = sprintf('noStop-noFrequent-withCitations-exactDates-%d-%d', nPerAuthor,I)
pathAffixInfo= 'noStop-noFrequent-withCitations'


info = read.delim(file = '../data/info.csv', header = FALSE, row.names = NULL, sep='\t', encoding = 'UTF-8')
true.dates = info[,4]
finfo = read.delim(file = sprintf('../data/input/feature-bmm-%d-%s.info',
                                  nPerAuthor,pathAffixInfo), header = TRUE, 
                   row.names = NULL, sep=' ', encoding = 'UTF-8', stringsAsFactors = FALSE)
periodization = read.delim(file = '../data/permanent/periodization-Adamik.csv', 
                           header = TRUE, row.names = NULL, sep='\t', encoding = 'UTF-8', stringsAsFactors = FALSE)

conns = list()
pltsScatter = list()
pltsConn = list()
table.strings = c()
top.cited = 5
top.wrds  = 5
pathTab2ToCN= '../data/output/tab2-ToCN.tex'
model = 'ToCN'
conn = as.matrix(read.delim(file = sprintf('../data/input/conn-%s-%s.dat',
                                           model,pathAffixRes), header = FALSE, row.names = NULL, sep=' '))

# this loads the results of the sampler
res = load.mm.result(model,pathAffixRes)
# generate the plots for the paper; in citation-model-functions.R
plts = citation.mm.plots(res,conn,info,periodization)
pltsScatter[[model]] = plts$plt1
pltsConn[[model]] = plts$plt2


##############################
# this is for tab1 in the paper
r = cited.authors.per.period(res,periodization,true.dates,info)
table.strings = c(table.strings,r$tex.table)

(p = pltsScatter$ToCN)
ggsave(filename = '../data/output/fig1.png', plot = pltsScatter$ToCN, device = 'png', width = 4.5, height=3.5, units = 'in')

# The connection plots
pltsConn$ToCN
ggsave(filename = '../data/output/fig2.png', plot = pltsConn$ToCN, device = 'png', width = 4.5, height=3.5, units = 'in')


# input for table 3 (tab1; Cited authors per literary period)
pathTab1 = '../data/output/tab1.tex'
cat('Author & Features \\\\ \n', file = pathTab1, append=FALSE)
for(ss in table.strings){
  cat(sprintf('%s \n',ss),file=pathTab1,append=TRUE)
}

