#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''

@author: ohell
'''
import numpy as np
import h5py,collections,sys,time,json
import defines,de_enc,configuration,split_counter


class DataPreprocessor(split_counter.SplitCounter):
    '''
    Builds data for a given configuration, and stores them in a hdf5 file.
    '''
    def __init__(self, net_config):
        self.reset()
        self.remove_duplicates = net_config['remove_duplicates']
        
        
    def reset(self):
        self.inputs = []
        self.outputs = []
        self.deenc_input = de_enc.DeEncoder()
        self.deenc_output= de_enc.DeEncoder()
        # special output symbols
        self.deenc_input.get_index(defines.SYM_PAD)
        self.deenc_input.get_index(defines.SYM_SPACE)
        self.deenc_output.get_index(defines.SYM_PAD)
        self.deenc_output.get_index(defines.SYM_IDENT)
        self.deenc_output.get_index(defines.SYM_SPLIT)
        self.sandhi_ngrams_right = {}
        self.sandhi_ngrams_left  = {}
        self.sandhi_ngrams_centered = {}
        
        self.position_maxes = []
        self.splitcnts_ngram2cnt_right = collections.defaultdict(int)
        self.splitcnts_ngram2cnt_left  = collections.defaultdict(int)
        self.splitcnts_ngram_minlen = 2
        self.splitcnts_ngram_maxlen = 6
        self.splitcnts_ngram_min_maxes = []
        self.valid_ixes = None
    def pad_left(self, s):
        s.insert(0, defines.SYM_BOL)
        return s
    def get_split_cnts(self, x, lens, verbose = True):
        '''
        Get split counts for sequences.
        The data for this function are built with build_split_cnts
        @todo: move this function into a separate class, along with loading/storing the split count data
        '''
        print('Getting split counts for all records ...')
        nsc = self.splitcnts_ngram_maxlen - self.splitcnts_ngram_minlen + 1
        sc = np.zeros(shape=[x.shape[0], x.shape[1], 2 * nsc], dtype=np.float32)
        for l in range(self.splitcnts_ngram_minlen, self.splitcnts_ngram_maxlen):
            for row in range(x.shape[0]):
                if verbose==True and row % 100==0:
                    sys.stdout.write(' len={0}, record={1}\r'.format(l,row));sys.stdout.flush()
                for col in range(lens[row]-l):
                    ngram = self.join_nums(x[row,col:(col+l)])
                    if ngram in self.splitcnts_ngram2cnt_right:
                        sc[row,col, (l-self.splitcnts_ngram_minlen)]+=self.splitcnts_ngram2cnt_right[ngram]
                    if ngram in self.splitcnts_ngram2cnt_left and (col+l)<=lens[row]:
                        sc[row,col+l, nsc+l-self.splitcnts_ngram_minlen]+=self.splitcnts_ngram2cnt_left[ngram]
        print('')
        ''' 
        normalization 
        '''
#        for x in range(sc.shape[2]):
#            sc[:,:,x]/=(np.max(sc[:,:,x]) + 1e-6)
        sc/=self.splitcnts_ngram_min_maxes
        return sc
    def build_split_cnts_minmax(self):
        print(' Building split-count mins/maxes ...')
        nsc = self.splitcnts_ngram_maxlen - self.splitcnts_ngram_minlen + 1
        self.splitcnts_ngram_min_maxes = np.zeros(shape=[2 * nsc], dtype=np.float32)
        for ngram, cnt in self.splitcnts_ngram2cnt_right.items():
            l = len(ngram.split(' '))
            self.splitcnts_ngram_min_maxes[l-self.splitcnts_ngram_minlen] = max(cnt, self.splitcnts_ngram_min_maxes[l-self.splitcnts_ngram_minlen])
        for ngram, cnt in self.splitcnts_ngram2cnt_left.items():
            l = len(ngram.split(' '))
            self.splitcnts_ngram_min_maxes[nsc + l-self.splitcnts_ngram_minlen] = max(cnt, self.splitcnts_ngram_min_maxes[nsc + l-self.splitcnts_ngram_minlen])
        self.splitcnts_ngram_min_maxes = np.reshape(self.splitcnts_ngram_min_maxes, [1,1,2*nsc]) + 1e-06
        print(' Done!')
        print(self.splitcnts_ngram_min_maxes)
    def build_split_cnts(self):
        '''
        from training data only
        '''
        print(' Building split counts')
        
        batch_size = 1000
        start = 0
        ident_ix = self.deenc_output.get_index(defines.SYM_IDENT, freeze = True)
        #ngram2cnt_right = collections.defaultdict(int)
        #ngram2cnt_left  = collections.defaultdict(int)
        pad_ix = self.deenc_input.get_index(defines.SYM_PAD)
        start_time = time.time()
        while start < len(self.inputs):
            end = min(start+batch_size, len(self.inputs)-1)
            if end<=start:
                break
            n = end-start+1
            X = np.full(shape=[n, self.max_sequence_length], fill_value=pad_ix, dtype=np.int32)
            Y = np.full(shape=[n, self.max_sequence_length], fill_value=pad_ix, dtype=np.int32)
            
            for row,ix in enumerate(range(start,end)):
                l = len(self.inputs[ix])
                for col in range(l):
                    X[row,col] = self.inputs[ix][col]
                    Y[row,col] = self.outputs[ix][col]
            ''' extract the n-grams '''
            w = np.where(Y!=ident_ix)
            rows = w[0]
            cols = w[1]
            x_flat = X.ravel()
            for i in range(self.splitcnts_ngram_minlen, self.splitcnts_ngram_maxlen):
                sys.stdout.write('  {0} => {1}\r'.format(start, i));sys.stdout.flush()
                ''' right contexts '''
                w_i = np.where( (cols + i) <= self.max_sequence_length )[0]
                if w_i.shape[0] > 0:
                    r = rows[w_i]
                    c = cols[w_i]
                    ''' indices into the flattened array (ravel) '''
                    x_ = r*self.max_sequence_length + c
                    slices = np.vstack(np.arange(x, x + i) for x in x_)
                    for s in x_flat[slices]:
                        ''' concatenate the ids into a string '''
                        if not pad_ix in s:
                            self.splitcnts_ngram2cnt_right[self.join_nums(s)]+=1
                ''' left contexts '''
                w_i  = np.where( (cols-i)>=0)[0]
                if w_i.shape[0] > 0:
                    r = rows[w_i]
                    c = cols[w_i]
                    ''' indices into the flattened array (ravel) '''
                    x_ = r*self.max_sequence_length + c
                    slices = np.vstack(np.arange(x-i, x) for x in x_)
                    for s in x_flat[slices]:
                        ''' concatenate the ids into a string '''
                        if not pad_ix in s:
                            self.splitcnts_ngram2cnt_left[self.join_nums(s)]+=1
            # next batch
            start=end
        print('')
        
        print('  duration: {0}'.format(time.time() - start_time ) )
        print('  got {0} left and {1} right n-grams'.format(len(self.splitcnts_ngram2cnt_left), len(self.splitcnts_ngram2cnt_right) ))
    def read_data_internal(self, data_path, max_n_load = 0, freeze = False):
        '''
        Loads data from a train/test/validation split file.
        @param freeze: If true, don't add new symbols to the input and output encoders. 
        '''
        num_truncated = 0
        _inputs = []
        _outputs = []
        len2cnt = dict()
        all_seqs = set()
        n_duplicates = 0
        if data_path:
            '''
            @todo: smarter selection between Sanskrit and German data.
            '''
            with open(data_path, "r", errors='replace') as datfile: # for Sanskrit
            #with codecs.open(data_path, 'r', 'UTF-8') as datfile: # German compounds
                input_ = []
                ''' standard output = target phonetic rule '''
                output = []
                full_seq = ''
                for line in datfile:
                    if max_n_load>0 and len(_inputs) > max_n_load: 
                        break
                    line = line.strip()
                    if not line or line.startswith('###'): 
                        continue
                    if line.startswith('$-'): # 100 letters before
                        pass
                    elif line.startswith('# TEXT '): # name of the text
                        pass
                    elif line.startswith('# TOPIC '): # topic category of the text
                        pass
                    elif line.startswith('# SEN'):
                        if self.remove_duplicates==True and full_seq in all_seqs:
                            n_duplicates+=1
                        elif len(input_)>0 and len(input_)==len(output):
                            all_seqs.add(full_seq)
                            if len(all_seqs) % 50==0:
                                sys.stdout.write(' read {0} sequences ...\r'.format(len(all_seqs)));sys.stdout.flush()
                            L = len(input_)
                            if L in len2cnt:
                                len2cnt[L]+=1
                            else:
                                len2cnt[L] = 1
                            max_len = (self.max_sequence_length-1)
                            if len(input_)>max_len:
                                input_ = input_[:max_len]
                                output = output[:max_len]
                                num_truncated+=1
                            if len(input_)<=max_len:# -1: we need a bol feature
                                ''' This function prepends a begin-of-line symbol '''
                                input_ = self.pad_left(input_)
                                output= self.pad_left(output)
                                '''
                                input_ is prefixed with a bol symbol. 
                                Must be translated to the same one in the output.
                                '''
                                output[0] = defines.SYM_IDENT
                                try:
                                    '''
                                    @attention: DON'T use
                                    _inputs.append([self.deenc_input.get_index(x, freeze) for x in input_])
                                    If get_index raises an error, the coordination between the information stored in the various 
                                    arrays gets lost.
                                    '''
                                    _in = [self.deenc_input.get_index(x, freeze) for x in input_] # can raise a value error!
                                    _out = [self.deenc_output.get_index(x, freeze) for x in output] # ... same ...
                                    ''' now it's safe! '''
                                    _inputs.append(_in)
                                    _outputs.append(_out)
                                except ValueError as err:
                                    print(err)
                        input_ = []
                        output = []
                        full_seq = ''
                    else:
                        ''' input-output pair '''
                        items = line.split(' ')
                        '''
                        We expect at least src, tar, POS and lemma id for each character position. 
                        Other information may come on top.
                        '''
                        if len(items)>=5 and items[0] and items[1]:
                            full_seq+=items[0]
                            input_.append(items[0])
                            target = items[1]
                            if target=="-BOW-" or target==items[0]:
                                target = defines.SYM_IDENT
                            elif target.endswith("-") and len(target)==2 and target[0]==items[0][0]:
                                # a => a-, t => t-
                                target = defines.SYM_SPLIT
                            output.append(target)
                        else:
                            print("Warning: invalid line = {0}".format(line))
            print('')
        print('read {0} sequences'.format(len(_inputs)))
        if num_truncated>0:
            print("!!NOTE: Truncated {0} of {1} sequences = {2} perc.".format(num_truncated, len(all_seqs), 100.0*float(num_truncated)/float(len(all_seqs))))
        if n_duplicates > 0:
            print("!!NOTE: Found {0} duplicated sequences".format(n_duplicates))

        return _inputs, _outputs
    
    def read_test_valid_set(self, is_test, data_path, max_n_load = 0):
        inputs, outputs = self.read_data_internal(data_path, max_n_load, freeze=True)
        n = len(inputs)
        if n > 0:
            print(' got {0} records for {1}'.format(n, 'test' if is_test else 'validation'))
            if is_test:
                self.test_ixes = np.arange(len(self.inputs), len(self.inputs) + n)
            else:
                self.valid_ixes = np.arange(len(self.inputs), len(self.inputs) + n)
            self.inputs.extend(inputs)
            self.outputs.extend(outputs)
    def transform_data(self, data_path_train, data_path_test, data_path_validation, net_config, data_directory):
        '''
        Call this function once before training a split model.
        
        Split data are prepared with the script
        helpers/train_test_split.py
        @param data_path_train_xt: content of 'data_path_train', but in Nagari style. Create these data with preprocessing/extend_training_set.py 
        '''
        self.reset()
        '''
        '''
        self.max_sequence_length = net_config['max_sequence_length_sen']
        self.inputs, self.outputs = self.read_data_internal(data_path_train, net_config['max_n_load'])
        self.train_ixes = np.arange(0, len(self.inputs), dtype=np.int32)
        # split information from training only
        self.build_split_cnts()
        self.build_split_cnts_minmax()
        # these calls automatically extend the arrays 'self.inputs' and 'self.outputs'
        self.read_test_valid_set(True, data_path_test, net_config['max_n_load'])
        self.read_test_valid_set(False, data_path_validation, net_config['max_n_load'])
        
        lens = np.asarray([len(x) for x in self.inputs], np.int32)
        
        inputs = np.full(shape=[lens.shape[0], self.max_sequence_length], 
                               fill_value=self.deenc_input.get_index(defines.SYM_PAD), dtype=np.int32)
        outputs = np.full(shape=[lens.shape[0], self.max_sequence_length], 
                               fill_value=self.deenc_input.get_index(defines.SYM_PAD), dtype=np.int32)
        for i in range(lens.shape[0]):
            L = lens[i]
            assert(L==len(self.outputs[i]))
            inputs[i,0:L] = self.inputs[i]
            outputs[i,0:L] = self.outputs[i]
        # split cnt information for all data
        split_cnt_matrix = self.get_split_cnts(inputs, lens)
        
        
        # store the data
        path_store = '{0}/data-{1}-{2}.hdf5'.format(data_directory, net_config['max_n_load'], net_config['max_sequence_length_sen'])
        with h5py.File(path_store, "w") as f:
            f.create_dataset(defines.HDF5_KEY_SEN_LENS, data = lens)
            f.create_dataset(defines.HDF5_KEY_TRAIN_IXES, data = self.train_ixes)
            f.create_dataset(defines.HDF5_KEY_TEST_IXES, data = self.test_ixes)
            f.create_dataset(defines.HDF5_KEY_VALID_IXES, data = self.valid_ixes)
            f.create_dataset(defines.HDF5_KEY_SPLIT_CNTS, data = split_cnt_matrix)
            f.create_dataset(defines.HDF5_KEY_INPUT, data = inputs)
            f.create_dataset(defines.HDF5_KEY_OUTPUT, data = outputs)
            #f.create_dataset(defines.HDF5_KEY_DE_ENC_INPUT, data = self.deenc_input.idx2sym)
            #f.create_dataset(defines.HDF5_KEY_DE_ENC_OUTPUT, data = self.deenc_output.idx2sym)
        minmaxes = np.squeeze(self.splitcnts_ngram_min_maxes).tolist()
        json_dict = {
            defines.ADD_KEY_DEENC_INPUT: self.deenc_input.idx2sym,
            defines.ADD_KEY_DEENC_OUTPUT:self.deenc_output.idx2sym,
            defines.ADD_KEY_SPLITCNTS_NGRAMS_LEFT : self.splitcnts_ngram2cnt_left,
            defines.ADD_KEY_SPLITCNTS_NGRAMS_RIGHT: self.splitcnts_ngram2cnt_right,
            defines.ADD_KEY_SPLITCNTS_MAXLEN : self.splitcnts_ngram_maxlen,
            defines.ADD_KEY_SPLITCNTS_MINLEN: self.splitcnts_ngram_minlen,
            defines.ADD_KEY_SPLITCNTS_MINMAX : minmaxes
            }
        with open('{0}/additional-data-{1}-{2}.json'.format(data_directory, net_config['max_n_load'], net_config['max_sequence_length_sen']), 'w', encoding = 'UTF-8') as f:
            json.dump(json_dict, f, ensure_ascii=False, indent=4)
        #self.deenc_input.store('{0}/de-enc-input-{1}-{2}.hdf5'.format(data_directory, net_config['max_n_load'], net_config['max_sequence_length_sen']))
        #self.deenc_output.store('{0}/de-enc-output-{1}-{2}.hdf5'.format(data_directory, net_config['max_n_load'], net_config['max_sequence_length_sen']))
        


if __name__ == '__main__':
    language = 'sanskrit' #'german' # 
    
    if language=='sanskrit':
        model_directory = '../data/models' # for storing trained models
        data_directory_input = '../data/input'
        data_file_names = ['sandhi-data-sentences-train.dat', 'sandhi-data-sentences-test.dat', 'sandhi-data-sentences-validation.dat']
        data_directory_result= '../data/result-publish'
        protocol_directory = '../data/protocol' # for the json file
        test_text_path =  data_directory_input + '/trbh.txt'
        
    
    data = DataPreprocessor(configuration.config)
    
    dev_path = data_directory_input + '/' + data_file_names[2] if data_file_names[2] else '' 
    data.transform_data(data_directory_input + '/' + data_file_names[0], 
                         data_directory_input + '/' + data_file_names[1], 
                         dev_path,
                         configuration.config,
                         data_directory_input)