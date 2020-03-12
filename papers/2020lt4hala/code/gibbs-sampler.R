# wrapper for training various models
library(Rcpp)
Sys.setenv("PKG_CXXFLAGS"="-std=c++0x -Wall -O3") # compiler flags, needed for std
sourceCpp('cpp/gibbs-time-or-background.cpp')
sourceCpp('cpp/gibbs-time.cpp')

source('functions-data.R')
source('constants.R')

types = DEFAULT_LING_FEATURES #NONLEX_LING_FEATURES# NONETYM_LING_FEATURES#
win.size = SMALL_WIN_SIZE#DEFAULT_WIN_SIZE

# If this variable is >0, it is the id of one text from the DCS (texts.id).
# No special priors are applied for this text.
plt.id = 0
# If >0, the text with this id is not included in the training data.
rem.txt.id = 0#464 
d = load.all.data(types, win.size, plt.id, rem.text.id=rem.txt.id)
X = d$X
txt.info = d$txt.info

# parameters
params = list(
  'I' = DEFAULT_NUM_TIMESTEPS,
  'J' = DEFAULT_NUM_BACKGROUND,
  'K' = ncol(X),
  'sd.param' = 1, # ~ 70%
  'plot.text.id' = plt.id,
  'iterations' = 4000,
  'model' = 'ToB', #'T', # 
  'frequency.transform' = 'sqrt',
  'win.size' = win.size
)

null.vec = c(0)
source('functions-data.R')
time.info = get.dirichlet.date.priors(txt.info, params$I, params$sd.param, group = 'section')

gdata = build.gibbs.data(X, time.info$tau, txt.info, transform=params$frequency.transform)
# how many data points per section?
#mean(table(gdata$Gr))
# train the model (Rcpp)
if(params$model=='ToB'){
  res = gibbs_time_or_background(gdata$T,gdata$S,gdata$F,gdata$Gr, time.info$tau, params,
                            C_in = matrix(0,0,0), D_in = matrix(0,0,0), E_in = matrix(0,0,0),
                            beta_in = null.vec, gamma_in = null.vec, delta_in = null.vec, epsilon_in = null.vec)
}else if(params$model=='T'){
  res = gibbs_time(gdata$T, gdata$F, gdata$Gr, time.info$tau, params, D_in=matrix(0,0,0), delta_in=null.vec)
}

res[['params']] = params
res[['types']] = types
res[['win.size']] = win.size
res[['plt.id']] = plt.id
res[['txt.info']] = txt.info
parms = c(params$model, 
          params$plot.text.id,
          paste(substr(sort(types),1,3), collapse='_'))
if(rem.txt.id>0){
  parms = c(parms, sprintf('rem%d', rem.txt.id))
}
path = sprintf('../%s/gibbs-%s.rds', data.dir.name, paste(parms, collapse='-'))
# see the notes on the function @load.result
# tag: @load-store-rds
saveRDS(res, file=path)
