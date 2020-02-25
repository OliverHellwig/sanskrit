import tensorflow as tf
import numpy as np
import constants,math,settings,helpers

class Model:
    def __init__(self, embs, targets_and_ixes, tar2seq, features_, seq_lens, params, unigrams_, bigrams_, trigrams_, is_training):
        batch_size_ = None 
        self.is_training = is_training
        self.dropout_rate = params['dropout_rate']
        self.max_sen_len = features_.shape[1]
        # pyramid indices
        p_x = helpers.get_pyramid_ranges(self.max_sen_len)
        pix = np.zeros(shape=features_.shape[0], dtype=np.int32)
        for i in range(features_.shape[0]):
            ix = -1
            #s = targets_and_ixes[i,1]; e = targets_and_ixes[i,2]
            s = 0; e = seq_lens[i]
            for j in range(len(p_x)):
                if s>=p_x[j][0] and e<=p_x[j][-1]:
                    ix = j
                    break
            if ix==-1:
                print('invalid pyr index: {0} {1}'.format(s,e))
                print(p_x)
                exit()
            pix[i] = ix
        ''' used for getting the right output node in the pyramid '''
        self.pyramid_ixes = tf.constant(value=pix, dtype=tf.int32, shape=pix.shape, name='pyramid_ixes')
        self.num_pyramid_nodes = len(p_x)
        # lexical features
        if params['train.lex.emb']==False:
            self.lex_embeddings = tf.constant(embs.astype(np.float32), name='lex_embeddings')
        else:
            self.lex_embeddings = tf.get_variable('lex_embeddings', shape=embs.shape, dtype=tf.float32, initializer=tf.constant_initializer(embs) )
        self.lex_size = embs.shape[1]
        
        ''' non-lexical features '''
        # n-gram features
        f = []
        if len(unigrams_)>0:
            f.append(np.copy(features_[:,:,unigrams_]))
        if len(bigrams_) > 0:
            f.append( self.build_bigram_features(np.copy(features_), bigrams_) )
        if len(trigrams_)>0:
            f.append( self.build_trigram_features(np.copy(features_), trigrams_) )
        if len(f)==0:
            print('no features selected!')
            exit()
        features = np.concatenate(f, axis=2)
        '''
        Up to here, the last dimension of @features_ stores indices of individual feature types;
        e.g. [2 1 ...] => value 2 for ft 1, 1 for ft 2 etc.
        These indices need to be mapped to embeddings, so the last dimension is "flattened". 
        Let feature type 1 have 10 different values, and 2 15 different values,
        [2 1 ...] is transformed into [0+2 10+1 15+...] 
        ''' 
        a = features.shape[0]; b = features.shape[1]; c = features.shape[2]
        tmp = np.reshape(np.copy(features), newshape=[a*b,c])
        n_per_feature = np.max(tmp, axis=0) + 1
        n_per_feature[0] = 0 # skip the lexical feature
        self.num_nonlex_features = np.sum(n_per_feature)
        self.num_nonlex_feature_types = c-1
        cs = np.cumsum(n_per_feature, dtype=np.int32)[:-1]
        for i in range(a):
            for j in range(b):
                features[i,j,1:]+=cs # increase the indices of all non-lexical features
        
        self.flex = tf.constant(features[:,:,0], name='f_lex')
        self.fnlex= tf.constant(features[:,:,1:], name='f_nonlex') # B, S, #features-1
        self.fpos = tf.constant(features[:,:,constants.POS_IX], name='f_pos')
        self.fcas = tf.constant(features[:,:,constants.CAS_IX], name='f_cas')
        self.fgen = tf.constant(features[:,:,constants.GEN_IX], name='f_gen')
        self.num_output_classes = np.max(targets_and_ixes[:,0])+1 # 
        
        self.seqlens = tf.constant(seq_lens, shape=seq_lens.shape, name='seqlens')
        self.nonlex_embeddings = tf.get_variable('nonlex_embeddings', shape=[self.num_nonlex_features,constants.NONLEX_EMB_SIZE], dtype=tf.float32,
                                                 initializer = tf.truncated_normal_initializer(0, 0.1))
        
        ''' special bigram features for child -> head '''
        descr_chi_head = [
            ['pos', 'pos'],
            ['cas', 'cas'],
            ['num', 'num'],
            ['cas', 'pos'],
            ['pos', 'cas'],
            ['cas', 'num'],
            ['num', 'cas'],
            ['vinf','cas'],
            ['cas', 'vinf']
            ]
        bigram_feats_ch_head = self.build_bigram_features_child_head(np.copy(features_), targets_and_ixes, descr_chi_head)
        n_per_feature = np.max(bigram_feats_ch_head, axis=0) + 1
        #n_per_feature[0] = 0 # skip the lexical feature
        n_child_head_features = np.sum(n_per_feature)
        cs = np.cumsum(n_per_feature, dtype=np.int32)#[:-1]
        ''' cs contains the offsets for each position, so shift it to the right '''
        cs = np.concatenate((np.zeros(1, np.int32), cs[:-1]))
        for i in range(a):
            bigram_feats_ch_head[i,:]+=cs
        self.fch = tf.constant(bigram_feats_ch_head, name='f_ch')
        self.num_child_head_features = bigram_feats_ch_head.shape[1]
        self.child_head_embeddings = tf.get_variable('child_head_embeddings', shape=[n_child_head_features,constants.NONLEX_EMB_SIZE], dtype=tf.float32,
                                                 initializer = tf.truncated_normal_initializer(0, 0.1))
        
        ''' index in targets -> index of the resp. sequence '''
        self.tar2seq       = tf.constant(tar2seq, shape=tar2seq.shape, name='tar2seq', dtype=tf.int32)
        self.targets       = tf.constant(targets_and_ixes[:,0], shape=targets_and_ixes[:,0].shape, name='targets', dtype=tf.int32)
        ''' position of child i in the sequence with index tar2seq[i] '''
        chixnp = targets_and_ixes[:,1] 
        self.child_ixes    = tf.constant(chixnp, shape=targets_and_ixes[:,1].shape, name='child_ixes', dtype=tf.int32)
        self.parent_ixes   = tf.constant(targets_and_ixes[:,2], shape=targets_and_ixes[:,2].shape, name='parent_ixes', dtype=tf.int32)
        
        self.ixes = tf.placeholder(tf.int32, [batch_size_], 'ixes')# which sequences are considered?
        self.instance_wts = tf.get_variable(name='instance_wts', shape=targets_and_ixes.shape[0], dtype=tf.float32, initializer=tf.constant_initializer(0.0), trainable=False)
        funs = {
            'baseline' : self.baseline,
            'bidirnn' : self.bidirnn,
            'sum' : self.sum
            }
        funs[params['model']](params)
    def build_bigram_features_child_head(self, F, tar_and_ixes, descr):
        ''' 
        special function for building bigrams involving the child and the head of relations
        @return: B x n matrix with indices of the newly created features. 
        '''
        a = F.shape[0]; b = F.shape[1]; c = F.shape[2]
        tmp = np.reshape(np.copy(F), newshape=[a*b,c])
        n_per_feature = np.max(tmp, axis=0) + 1
        ch = tar_and_ixes[:,1]; he = tar_and_ixes[:,2]
        bigrams = np.zeros([a,len(descr)], np.int32)
        for bgix,des in enumerate(descr):
            ix1 = settings.feature2ix[des[0]]
            ix2 = settings.feature2ix[des[1]]
            
            N = n_per_feature[ix2]
            for i in range(a): # for each child->head record ...
                bigrams[i,bgix] = self.bi_ix(F[i,ch[i],ix1], F[i,he[i],ix2], N)
        return bigrams
    def build_bigram_features(self, F, descr):
        a = F.shape[0]; b = F.shape[1]; c = F.shape[2]
        tmp = np.reshape(np.copy(F), newshape=[a*b,c])
        n_per_feature = np.max(tmp, axis=0) + 1
        bigrams = []
        for des in descr:
            ix1 = settings.feature2ix[des[0]]
            ix2 = settings.feature2ix[des[1]]
            ba  = des[2] # before/after
            if ba==False and ix1==ix2:
                print('Invalid description')
                print(des)
                exit()
            if ba==True:
                N = n_per_feature[ix2]
                bigrams_n = np.zeros([a,b,1], np.int32)
                bigrams_p = np.zeros([a,b,1], np.int32)
                for i in range(a):
                    for j in range(b-1): # for each word in sequence i
                        bigrams_n[i,j,0] = self.bi_ix(F[i,j,ix1], F[i,j+1,ix2], N)
                    for j in range(1,b):
                        bigrams_p[i,j,0] = self.bi_ix(F[i,j,ix1], F[i,j-1,ix2], N)
                bigrams.append(bigrams_p)
                bigrams.append(bigrams_n)
            else:
                N = n_per_feature[ix2]
                bi = np.zeros([a,b,1], np.int32)
                for i in range(a):
                    for j in range(b): # for each word in sequence i
                        bi[i,j,0] = self.bi_ix(F[i,j,ix1], F[i,j,ix2], N)
                bigrams.append(bi)
        return np.concatenate(bigrams,axis=2)
    def build_trigram_features(self, F, descr):
        a = F.shape[0]; b = F.shape[1]; c = F.shape[2]
        tmp = np.reshape(np.copy(F), newshape=[a*b,c])
        n_per_feature = np.max(tmp, axis=0) + 1
        trigrams = []
        for des in descr:
            ixp = settings.feature2ix[des[0]]
            ixc = settings.feature2ix[des[1]]
            ixn = settings.feature2ix[des[2]]
            nc = n_per_feature[ixc]
            nn = n_per_feature[ixn]
            tri = np.zeros([a,b,1], np.int32)
            if des[3]==True:# before/after
                for i in range(a):
                    for j in range(b): # for each word in sequence i
                        if j==0:
                            tri[i,j,0] = self.tri_ix(0, F[i,j,ixc], F[i,j+1,ixn], nc, nn)
                        elif j==(b-1):
                            tri[i,j,0] = self.tri_ix(F[i,j-1,ixp], F[i,j,ixc], 0, nc, nn)
                        else:
                            tri[i,j,0] = self.tri_ix(F[i,j-1,ixp], F[i,j,ixc], F[i,j+1,ixn], nc, nn)
                trigrams.append(tri)
            else:
                for i in range(a):
                    for j in range(b): # for each word in sequence i
                        tri[i,j,0] = self.tri_ix(F[i,j,ixp], F[i,j,ixc], F[i,j,ixn], nc,nn)
                trigrams.append(tri)
        return np.concatenate(trigrams,axis=2)
    def bi_ix(self, f1,f2,mult):
        return f1*mult+f2
    def tri_ix(self, f1,f2,f3,mul2,mul3):
        return f1*mul2*mul3+f2*mul3 + f3
    def baseline(self, params):
        ''' just uses the information about parent and child, no sentence context
        called feedforward in the paper
        '''
        input_, ch_ixes, pa_ixes = self.build_3d_input(params=params)
        hi = self.build_pairwise_input(input_, ch_ixes, pa_ixes, params) # B, 2*E
        self.cost_and_train(
            self.penult_op(hi, params), 
            params)

    def sum(self, params):
        ''' skip connections and sum(sentence) '''
        input_, ch_ixes, pa_ixes = self.build_3d_input(params=params)
        su = self.sum_op(input_, params['penult.size'], 'sum_op')
        pw = self.build_pairwise_input(input_, ch_ixes, pa_ixes, params)
        #pw = self.fc(pw, params['penult.size'], tf.nn.tanh, 'pairwise_penult')
        hi = tf.concat([pw,su],axis=1)
        
        self.cost_and_train(self.penult_op(hi, params), params)
     
    def get_sequence_mask(self, for_tensor, d_type=tf.float32):
        '''
        sequence mask for a three-dimensional tensor
        index j of dim 1 is set to 1 in record i, if j<slens[i]  
        '''
        slens = tf.gather(params=self.seqlens, indices=self.ixes) # lengths of the current sequences
        mask = tf.reshape(
            tf.sequence_mask(lengths=slens, maxlen=self.max_sen_len, dtype=d_type),
            [-1,self.max_sen_len,1]) # B, S, 1
        return tf.tile(mask, multiples=[1,1,tf.shape(for_tensor)[2]]) # B, S, E
    def bidirnn(self, params):
        ''' solid default model '''
        #input_, ch_ixes, pa_ixes = self.build_3d_input(params=params)
        input_, ch_ixes, pa_ixes = self.build_3d_input(params=params)
        O,_ = self.rnn_op(input_, params, params['penult.size'], 'rnn_op')
        pw = self.build_pairwise_input(input_, ch_ixes, pa_ixes, params)
        #pw = self.fc(pw, params['penult.size'], tf.nn.tanh, 'pairwise_penult')
        hi = tf.concat([pw,O],axis=1)
        
        self.cost_and_train(self.penult_op(hi, params), params)

    def build_3d_input(self, params, mark_chi_par_positions=False):
        '''
        Creates a representation of the current text lines
        '''
        seq_ixes = tf.gather(params=self.tar2seq, indices=self.ixes) # indices into flex and fnlex
        ch_ixes  = tf.gather(params=self.child_ixes, indices=self.ixes) # positions of the children in the sequences
        pa_ixes  = tf.gather(params=self.parent_ixes, indices=self.ixes)
        
        # get the embeddings
        lex    = tf.nn.embedding_lookup(params=self.lex_embeddings, ids=tf.gather(params=self.flex, indices=seq_ixes), name='lex') # [batch, max_seq_len, 100]
             
        ''' 
        Lexical embeddings are not adapted during training in some settings.
        Therefore, allow an additional non-linear transformation for adapting them
        to the syntactic task.
         '''
        if params['lex.emb.adapt.size'] is not None and params['lex.emb.adapt.size']>0:
            asz = params['lex.emb.adapt.size']
            tmp = tf.reshape(lex, [-1, self.lex_size])
            tmp = self.fc(tmp, asz, tf.nn.tanh, 'lex_transform')
            lex = tf.reshape(tmp, [-1, self.max_sen_len, asz])
            # @todo should be deactivated
            #self.lex_embeddings_adapted = self.fc(self.lex_embeddings, asz, tf.nn.tanh, 'lex_transform', reuse=True)
        # concatenate lexical and non-lexical features    
        nonlex = tf.nn.embedding_lookup(params=self.nonlex_embeddings, ids=tf.gather(params=self.fnlex, indices=seq_ixes), name='nonlex') # [batch, max_seq_len, LR_IX, NONLEX_EMB_SIZE]
        input_ = tf.concat([lex,
                            tf.reshape(nonlex, [-1,self.max_sen_len, (self.num_nonlex_feature_types)*constants.NONLEX_EMB_SIZE])], axis=2) # B, S, 100+LR_IX*NONLEX_EMB_SIZE 
        if mark_chi_par_positions:
            chi_pos = tf.one_hot(indices=ch_ixes, depth=self.max_sen_len, on_value=1.0, off_value=0.0, dtype=tf.float32)
            chi_pos = tf.reshape(chi_pos,[-1,self.max_sen_len,1])
            par_pos = tf.reshape(
                tf.one_hot(indices=pa_ixes, depth=self.max_sen_len, on_value=1.0, off_value=0.0, dtype=tf.float32),
                [-1,self.max_sen_len,1])
            input_ = tf.concat([input_,chi_pos,par_pos], axis=2)
        return input_, ch_ixes, pa_ixes
    def build_pairwise_input(self, input_, ch_ixes, pa_ixes, params):
        chi = self.extract_axis_1(input_, ch_ixes) # B, E
        par = self.extract_axis_1(input_, pa_ixes) # B, E
        ten = tf.concat([chi,par],axis=1)
        if params['use.child.head.bigrams']==True:
            ch = tf.nn.embedding_lookup(params=self.child_head_embeddings, ids=tf.gather(params=self.fch, indices=self.ixes)) # B, num_ch_features, non_lex_emb_sz
            ten = tf.concat([ten, tf.reshape(ch, [-1, self.num_child_head_features*constants.NONLEX_EMB_SIZE]) ],axis=1)
        return ten 
    def penult_op(self, input_, params):
        hi = input_
        for i,sz in enumerate(params['hidden.sizes']):
            with tf.variable_scope('hi_{0}'.format(i)):
                hi = self.fc(hi, sz, tf.nn.tanh, 'hidden')
        return hi
    def rnn_op(self, input_, params, out_sz, name='rnn_op'):
        ''' RNN(input) -> out_sz '''
        input_ = input_ * self.get_sequence_mask(for_tensor=input_, d_type=tf.float32)
        with tf.variable_scope(name):
            H, O = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=self.rnn_cell(params, out_sz),
                cell_bw=self.rnn_cell(params, out_sz),
                inputs=input_,
                sequence_length=tf.gather(params=self.seqlens, indices=self.ixes),
                # initial_state=initial_state_enc,
                dtype=tf.float32,
                time_major=False)
        ''' B, S, 2*hidden '''
        H = tf.concat([H[0], H[1]], axis=2)
        return tf.concat([O[0][1], O[1][1] ], axis=1), H
        #chiH = self.extract_axis_1(H, ch_ixes)
        #parH = self.extract_axis_1(H, pa_ixes)
    def sum_op(self, input_, out_sz, name='sum_op'):
        with tf.variable_scope(name):
            # create a mask based on seqlens
            mask = self.get_sequence_mask(for_tensor=input_, d_type=tf.float32) # B, S, E
            su  = tf.reduce_sum( input_ * mask, axis=1 ) # B, E
            #slens = tf.gather(params=self.seqlens, indices=self.ixes) # lengths of the current sequences
            #div = tf.tile( tf.reshape(slens,[-1,1]), multiples = [1, tf.shape(su)[1] ]) # B, E
            #su = su/tf.cast(div, tf.float32)
            return self.fc(su, out_sz, tf.nn.tanh, 'su_fc', True) 
    def cost_and_train(self, penult, params, add_cost=None):
        self.logits = self.fc(penult, self.num_output_classes, None, 'fc_logits', False)
        self.softmax_values = tf.nn.softmax(self.logits, axis=-1, name='softmax_values')
        self.predictions = tf.cast( tf.argmax( self.softmax_values, axis=1), tf.int32 )
        tar = tf.gather(params=self.targets, indices=self.ixes)
        self.num_correct = tf.reduce_sum(tf.cast(tf.equal(self.predictions, tar), tf.float32))
        self.cost   = tf.reduce_mean( 
                        #tf.gather(self.instance_wts, indices=self.ixes) * 
                        tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tar, logits=self.logits, name="cost") )
        if not add_cost is None:
            self.cost+=add_cost
        if self.is_training:
            self.train_op = tf.train.AdagradOptimizer(learning_rate=0.1).minimize(self.cost) # good for baseline
            
    def rnn_cell(self, config_, hidden_size=0):
        hsz = hidden_size if not hidden_size == 0 else config_['rnn.size']
        kp = 1.0-self.dropout_rate if self.is_training else 1.0
        return tf.nn.rnn_cell.DropoutWrapper(
            cell=tf.nn.rnn_cell.LSTMCell(hsz),
            # cell = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(config_[constants.HIDDEN_SIZE]), 
            input_keep_prob=kp,
            output_keep_prob=kp,
            state_keep_prob=kp
            )
        
    def weight(self, insize, outsize, name):
        '''
        Creates a weight matrix.
        '''
        return tf.get_variable(name, shape=[insize, outsize], dtype=tf.float32, 
                               initializer=tf.truncated_normal_initializer(0,stddev=1.0/math.sqrt(float(insize) ))
                               )
    def bias(self, outsize, name, bias_init_val = 0):
        '''
        Creates a bias vector
        '''
        return tf.get_variable(name, shape=[outsize], dtype=tf.float32, initializer=tf.constant_initializer(bias_init_val))
    def fc(self, inp, size, activation, name, use_dropout = True, reuse = None):
        with tf.variable_scope(name, reuse=reuse):
            w = self.weight( int(inp.get_shape()[1]), size, name='wt' )
            #self.weights.append(w)
            b = self.bias(size, name='bias', bias_init_val = 0.1)
            #self.biases.append(b)
            out = tf.matmul(inp,w) + b
            if not activation is None:
                out = activation(out)
            if self.is_training and use_dropout==True:
                out = tf.nn.dropout(out, rate=self.dropout_rate)
            return out
    def extract_axis_1(self, data, ind):
        """
        source:
        https://stackoverflow.com/questions/36817596/get-last-output-of-dynamic-rnn-in-tensorflow
        Get specified elements along the first axis of tensor.
        :param data: Tensorflow tensor that will be subsetted.
        :param ind: Indices to take (one for each element along axis 0 of data). OH: must by tf.int32
        @return: the nth elm in each row (= batch) of data
        """
        batch_range = tf.range(tf.shape(data)[0])
        '''
        indices = 
        [[0,5], # = seq. len=5 in batch 0
        [1,3], # = seq.len = 3 in batch 1
        ...
        ]
        '''
        indices = tf.stack([batch_range, ind], axis=1)
        res = tf.gather_nd(data, indices)
        return res
