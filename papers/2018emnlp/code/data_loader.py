#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import codecs,os,h5py,json
import defines,de_enc,split_counter


class DataLoader(split_counter.SplitCounter):
    def __init__(self, data_directory, config, load_data_into_ram, load_data = True):
        '''
        Load the data created with preprocess_data.py
        @param load_data: If False, load only the additional data (for applying a trained model) 
        '''
        self.load_data_into_ram = load_data_into_ram
        self.deenc_input = de_enc.DeEncoder()
        self.deenc_output= de_enc.DeEncoder()
        self.data_file = None
        # special output symbols
        self.batch_index = 0
        self.batch_size = 0
        self.batch_x = np.zeros(shape=[1,1])
        self.batch_y = np.zeros(shape=[1,1])
        self.batch_split_cnts = np.zeros(shape=[1,1,1])
        self.max_sequence_length = config['max_sequence_length_sen']
        self.initialize_char_maps()
        #self.deenc_input.load('{0}/de-enc-input-{1}-{2}.hdf5'.format(data_directory, config['max_n_load'], config['max_sequence_length_sen']))
        #self.deenc_output.load('{0}/de-enc-output-{1}-{2}.hdf5'.format(data_directory, config['max_n_load'], config['max_sequence_length_sen']))
        with open('{0}/additional-data-{1}-{2}.json'.format(data_directory, config['max_n_load'], config['max_sequence_length_sen']), 'r') as f:
            dic = json.load(f)
            self.deenc_input.build(dic[defines.ADD_KEY_DEENC_INPUT])
            self.deenc_output.build(dic[defines.ADD_KEY_DEENC_OUTPUT])
            self.splitcnts_ngram2cnt_left = dic[defines.ADD_KEY_SPLITCNTS_NGRAMS_LEFT]
            self.splitcnts_ngram2cnt_right = dic[defines.ADD_KEY_SPLITCNTS_NGRAMS_RIGHT]
            self.splitcnts_ngram_maxlen = dic[defines.ADD_KEY_SPLITCNTS_MAXLEN]
            self.splitcnts_ngram_minlen = dic[defines.ADD_KEY_SPLITCNTS_MINLEN]
            mima = dic[defines.ADD_KEY_SPLITCNTS_MINMAX]
            self.splitcnts_ngram_min_maxes = np.reshape(np.asarray(mima, np.float32), newshape=[1,1,len(mima)])
        if load_data==True:
            data_path = '{0}/data-{1}-{2}.hdf5'.format(data_directory, config['max_n_load'], config['max_sequence_length_sen'])
            if not os.path.exists(data_path):
                raise FileNotFoundError('File not found: {0}'.format(data_path))
            self.data_file = h5py.File(data_path, 'r')
            tst = self.data_file[defines.HDF5_KEY_INPUT]
            if self.max_sequence_length!=tst.shape[1]:
                raise ValueError('Max. seq. length ({0}) does not match the y-dimension of the HDF5 data ({1}). Rebuild the data file!'.format(self.max_sequence_length, tst.shape[1]))
            self.train_ixes = np.asarray(self.data_file[defines.HDF5_KEY_TRAIN_IXES], np.int32)
            self.test_ixes = np.asarray(self.data_file[defines.HDF5_KEY_TEST_IXES], np.int32)
            self.valid_ixes = np.asarray(self.data_file[defines.HDF5_KEY_VALID_IXES], np.int32)
            self.seq_lens = np.asarray(self.data_file[defines.HDF5_KEY_SEN_LENS], np.int32)
            self.n_split_cnts = self.data_file[defines.HDF5_KEY_SPLIT_CNTS].shape[2]
            if self.load_data_into_ram==True:
                self.inputs = np.asarray(self.data_file[defines.HDF5_KEY_INPUT], np.int32)
                self.outputs= np.asarray(self.data_file[defines.HDF5_KEY_OUTPUT], np.int32)
                self.split_cnts = np.asarray(self.data_file[defines.HDF5_KEY_SPLIT_CNTS], np.float32)
        
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_value, traceback):
        if not self.data_file is None:
            self.data_file.close()
            print('HDF5 file closed')
    def initialize_char_maps(self):
        '''
        2 different maps are required, because the order of aspirates is different.
        @todo: check syllabic R's
        '''
        self.unicode2intern = [
            (u'ā',  'A'),
            (u'ī',  'I'),
            (u'ū',  'U'),
            (u'ṛ',  'R'),
            (u'ṝ',  'L'), # ??
            (u'ḷ',  '?'),
            (u'ḹ',  '?'),
            (u'ai', 'E'),
            (u'au', 'O'),
            # gutturals
            (u'kh', 'K'),
            (u'gh', 'G'),
            (u'ṅ',  'F'),
            # palatals
            (u'ch', 'C'),
            (u'jh', 'J'),
            (u'ñ',  'Q'),
            # retroflexes
            (u'ṭh', 'W'),
            (u'ṭ',  'w'),
            (u'ḍh', 'X'),
            (u'ḍ',  'x'),
            (u'ṇ',  'N'),
            # dentals
            (u'th', 'T'),
            (u'dh', 'D'),
            # labials 
            (u'ph', 'P'),
            (u'bh', 'B'),
            # others
            (u'ś',  'S'),
            (u'ṣ',  'z'),
            (u'ṃ',  'M'),
            (u'ḥ',  'H')
            ]
        '''
        @attention: Keep this second list. Order matters! 
        '''
        self.intern2unicode = [
            (u'ā',  'A'),
            (u'ī',  'I'),
            (u'ū',  'U'),
            (u'ṛ',  'R'),
            (u'ṝ',  'L'), # ??
            (u'ḷ',  '?'),
            (u'ḹ',  '?'),
            (u'ai', 'E'),
            (u'au', 'O'),
            # gutturals
            (u'kh', 'K'),
            (u'gh', 'G'),
            (u'ṅ',  'F'),
            # palatals
            (u'ch', 'C'),
            (u'jh', 'J'),
            (u'ñ',  'Q'),
            # retroflexes; ORDER MATTERS
            (u'ṭ',  'w'),
            (u'ṭh', 'W'),
            (u'ḍ',  'x'),
            (u'ḍh', 'X'),
            (u'ṇ',  'N'),
            # dentals
            (u'th', 'T'),
            (u'dh', 'D'),
            # labials 
            (u'ph', 'P'),
            (u'bh', 'B'),
            # others
            (u'ś',  'S'),
            (u'ṣ',  'z'),
            (u'ṃ',  'M'),
            (u'ḥ',  'H')
            ]
    def initialize_batch(self, _batch_size):
        np.random.shuffle(self.train_ixes)
        self.batch_index = 0
        self.batch_size = _batch_size
    def has_more_data(self):
        if self.batch_index >= len(self.train_ixes):
            return False # no more training samples
        return True
    def get_next_batch(self):
        if self.load_data_into_ram==True:
            return self.get_next_batch_ram()
        else:
            return self.get_next_batch_hdf5()
    def get_next_batch_hdf5(self):
        '''
        Get the next training batch from the open hdf5 file
        '''
        if self.has_more_data()==False:
            return False # no more training samples
        
        end = min(self.batch_index + self.batch_size, len(self.train_ixes))
        ixes = self.train_ixes[self.batch_index:end]
        
        # hdf5 requires indices in ascending order
        ixes_asort = np.argsort(ixes)
        
        self.batch_x = self.data_file[defines.HDF5_KEY_INPUT][ixes[ixes_asort],:]
        self.batch_y = self.data_file[defines.HDF5_KEY_OUTPUT][ixes[ixes_asort],:]
        self.batch_seq_lens = self.seq_lens[ixes]
        self.batch_split_cnts = self.data_file[defines.HDF5_KEY_SPLIT_CNTS][ixes[ixes_asort],:,:]
        self.batch_x = self.batch_x[ixes_asort,:]
        self.batch_y = self.batch_y[ixes_asort,:]
        self.batch_split_cnts = self.batch_split_cnts[ixes_asort,:,:]
        
        self.batch_index = end # for the next call
        return True
    def get_next_batch_ram(self):
        '''
        Get the next training batch from the open hdf5 file
        '''
        if self.has_more_data()==False:
            return False # no more training samples
        end = min(self.batch_index + self.batch_size, len(self.train_ixes))
        ixes = self.train_ixes[self.batch_index:end]
        
        self.batch_x = self.inputs[ixes,:]
        self.batch_y = self.outputs[ixes,:]
        self.batch_seq_lens = self.seq_lens[ixes]
        self.batch_split_cnts = self.split_cnts[ixes,:,:]
        
        self.batch_index = end # for the next call
        return True
    def unicode_to_internal_transliteration(self,s):
        '''
        Transforms from IAST to the internal transliteration
        '''
        for src,dst in self.unicode2intern:
            s = s.replace(src,dst)
        return s
        
    def internal_transliteration_to_unicode(self, s):
        for src,dst in self.unicode2intern:
            s = s.replace(dst, src)
        return s
    def get_split_cnts(self, x, lens, verbose = True):
        '''
        Get split counts for sequences.
        The data for this function are built with build_split_cnts
        @todo: move to split_counter
        '''
        if verbose==True:
            print('Getting split counts for all records ...')
        nsc = self.splitcnts_ngram_maxlen - self.splitcnts_ngram_minlen + 1
        sc = np.zeros(shape=[x.shape[0], x.shape[1], 2 * nsc], dtype=np.float32)
        for l in range(self.splitcnts_ngram_minlen, self.splitcnts_ngram_maxlen):
            for row in range(x.shape[0]):
                for col in range(lens[row]-l):
                    ngram = self.join_nums(x[row,col:(col+l)])
                    if ngram in self.splitcnts_ngram2cnt_right:
                        sc[row,col, (l-self.splitcnts_ngram_minlen)]+=self.splitcnts_ngram2cnt_right[ngram]
                    if ngram in self.splitcnts_ngram2cnt_left and (col+l)<=lens[row]:
                        sc[row,col+l, nsc+l-self.splitcnts_ngram_minlen]+=self.splitcnts_ngram2cnt_left[ngram]
        if verbose:print('')
        ''' 
        normalization 
        '''
#        for x in range(sc.shape[2]):
#            sc[:,:,x]/=(np.max(sc[:,:,x]) + 1e-6)
        sc/=self.splitcnts_ngram_min_maxes
        return sc
    def load_external_text(self, path):
        '''
        Opens a file with UTF-8 + IAST text, transforms it into the internal transliteration, 
        and then into a symbolic representation.
        Should work in the same way as DataPreprocessor::read_data_internal()
        '''
        seqs = []
        lines_orig = []
        lens = []
        try:
            with codecs.open(path, 'r', 'UTF-8') as infile:
                pad_ix = self.deenc_input.get_index(defines.SYM_PAD, freeze=True, allow_unk=True)
                bol_ix = self.deenc_input.get_index(defines.SYM_BOL, freeze=True, allow_unk=True)
                for line in infile:
                    line_orig = self.unicode_to_internal_transliteration(line.strip())
                    line = line.strip().replace(u' ', defines.SYM_SPACE)
                    line = self.unicode_to_internal_transliteration(line)
                    if len(line) >= self.max_sequence_length:
                        line = line[:(self.max_sequence_length-1)] # -1: pad_... always prefixes a ^ symbol!
                        line_orig = line_orig[:(self.max_sequence_length-1)]
                    seq = [self.deenc_input.get_index(x, freeze=True, allow_unk=True) for x in line]
                    seq.insert(0, bol_ix)
                    seq_len = len(seq)
                    line_orig = ' ' + line_orig
                    while len(seq) < self.max_sequence_length:
                        seq.append(pad_ix)
                        line_orig+=' '
                    if len(seq)>0:
                        seqs.append(seq)
                        lens.append(seq_len)
                        lines_orig.append(line_orig)
        except IOError as ex:
            print("I/O error({0}): {1}".format(ex.errno, ex.strerror))
            seqs = []
        if len(seqs)>0:
            seqs = np.asarray(seqs, np.int32)
            lens = np.asarray(lens, np.int32)
            splitcnts = self.get_split_cnts(seqs, lens, verbose=False)
            return seqs,lens,splitcnts,lines_orig
        else:
            return None, None,None,None