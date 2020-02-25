import constants
data_input_directory = 'data/input/'
data_output_directory = 'data/output/'
# 
npz_path = data_input_directory + 'split-data.npz'
# how many train/test splits?
num_folds = 10

feature2ix = {
    'word' : constants.LEMMA_IX,
    'pos' : constants.POS_IX,
    'cas' : constants.CAS_IX,
    'num' : constants.NUM_IX,
    'gen' : constants.GEN_IX,
    'vper': constants.VPER_IX,
    'vten': constants.VTEN_IX,
    'vpas': constants.VPAS_IX,
    'vinf': constants.VINF_IX,
    'cas-agr' : constants.CAS_AGR_IX,
    'num-agr' : constants.NUM_AGR_IX,
    'gen-agr' : constants.GEN_AGR_IX,
    'all-agr' : constants.ALL_AGR_IX,
    'lr'  : constants.LR_IX
    }


''' which bigram features should be constructed? '''
bigrams_seq = [
    #['word','word',True],
    ['pos', 'pos', True],
    ['pos', 'cas', True],
    ['pos', 'num', True],
    ['cas', 'pos', True],
    ['cas', 'cas', True],
    #['cas', 'vinf',True],
    ['num', 'pos', True],
    ['num', 'num', True],
    # same word
    ['cas', 'num', False]
    #['cas', 'vinf',False]
    ]

trigrams_seq = [
    #['word','word',True],
    ['pos', 'pos', 'pos', True],
    ['cas', 'cas', 'cas', True],
    # same word
    ['cas', 'num', 'gen', False]
    ]