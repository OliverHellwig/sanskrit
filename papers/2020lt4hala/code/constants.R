# constant values
#levels.zehnder = c('1-RV', '2-MA', '3-PS', '4-PO', '5-PL', '6-SU')
levels.zehnder = c('1-RV', '2-MA', '3-PO', '4-PL', '5-SU')
# output dimensions in inches
time.plot.dim = c(10,5)
time.plot.dim.low = c(10,3.5)
square.plot.dim = c(5,5)


# work in progress or for some paper?
data.dir.name = 'data' # 

# which features should be used as default?
DEFAULT_LING_FEATURES = c('vclass', 'deriv', 'compounds','case', 'etym', 'tense-mode', 'infinite', 'pos', 'pos2', 'pos3','lexicon')
NONLEX_LING_FEATURES = c('vclass', 'deriv', 'compounds','case', 'etym', 'tense-mode', 'infinite', 'pos', 'pos2', 'pos3' )
NONETYM_LING_FEATURES = c('vclass', 'deriv', 'compounds','case', 'tense-mode', 'infinite', 'pos', 'pos2', 'pos3','lexicon')

# which window size?
DEFAULT_WIN_SIZE = 400
SMALL_WIN_SIZE = 200
TINY_WIN_SIZE = 100
# How many bg distributions? Use acl-experiments-n-background.R
DEFAULT_NUM_BACKGROUND = 3
DEFAULT_NUM_TIMESTEPS = 30

DATE_LOW = -1300
DATE_UP  = -300
DATE_RANGE = DATE_UP - DATE_LOW

DATES_LOWER_5 = c(DATE_LOW,  -1100,  -900,  -700,   -600)
DATES_UPPER_5 = c(-1000,   -900,  -700,   -400,   DATE_UP)

