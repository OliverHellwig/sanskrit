config = {
	'max_sequence_length_sen' : 128,
	'max_n_load' : 0, # how many text lines should be loaded. 0 = all
	'n_hidden' : 200, # number of hidden cells in the rnn
	'cell_type' : 'lstm', #'lstm-block',
	'emb_size' : 128, # size of character embeddings
	'filter_sizes' : [3,5,7], # for the convolutional filters
	'num_filters' : 100, # how many conv. filters?
	'max_epochs' : 10, # number of training epochs
	'learning_rate' : 0.005,
	'batch_size' : 80,
	'dropout' : 0.8,# == keep prob!
	'gradient_clipping' : 5.0,
	'valid_batch_size' : 2000, # must not be too large (RAM)!
	'display_step' : 2000,
    'remove_duplicates' : False,
    'optimizer' : 'adam', # see Model::penult_and_classification for other values
    'has_lr_schedule' : 2,
    'use_split_cnts' : 1, # 1 = use split counts, 0 = don't
    'model_directory' : '../data/models' # for storing trained models
}