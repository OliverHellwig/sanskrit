# logistic regression with gold data
# which model best explains the use of POS tags when iva is labeled as discourse?
library(rstan)
rstan_options(auto_write = TRUE)
library(data.table)
library(ggplot2)
library(coda)
library(loo)
source('functions.R')


# Bayesian logistic regression
# run Stan
data = fread('../data/iva-syntax-gold.csv', sep='\t', encoding='UTF-8',data.table=FALSE)
data = data[data$label=='discourse',]

is.noun = ifelse(data$head.pos %in% c('NOUN','ADJ'),1,0)
ta = table(data$time.slot, is.noun)
noun = ta[,2]
total = rowSums(ta)



models = c('blg-gdn-01-time',
           'blg-gdn-02-mantra',
           'blg-gdn-03-time-or-mantra'
           )

for(model in models){
  z = stan(file=sprintf('stan/%s.stan',model),
           data=list(noun = noun, total=total),
           iter = 5000,
           chains=4,
           verbose=FALSE
  )
  ## save the result
  saveRDS(z, file=sprintf('stan-result/%s.z',model))
  
}

# use model-comparison.R for the further evaluation!
