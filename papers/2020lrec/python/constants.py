# must have the same order as in build_data DepWord::features
# plus: also used in R::evaluate-ablation.R
LEMMA_IX = 0
POS_IX = 1
CAS_IX = 2
NUM_IX = 3
GEN_IX = 4
VPER_IX = 5
#VNUM_IX = 6
VTEN_IX = 6
VPAS_IX = 7
VINF_IX = 8
CAS_AGR_IX = 9
NUM_AGR_IX = 10
GEN_AGR_IX = 11
ALL_AGR_IX = 12
LR_IX = 13

# size of embeddings for individual non-lexical features, e.g. number, case
NONLEX_EMB_SIZE = 5 # larger/smaller values don't really help

MAX_SEN_LEN = 32 #64

LEX_ADAPT_SIZE = 50