import tensorflow as tf
import math
import defines

class Model():
    '''
    LSTM => convolution => skip connections
    '''
    def __init__(self, config, n_input, n_classes, n_split_cnts):
        ''' Create the variables '''
        self.x = tf.placeholder(tf.int64, [None, config['max_sequence_length_sen']], name='inputs')
        '''
        target classes for char-wise Sandhi/compound splitting
        '''
        self.y = tf.placeholder(tf.int64, [None, config['max_sequence_length_sen']], name='targets')
        self.split_cnts = tf.placeholder(tf.float32, [None, config['max_sequence_length_sen'], n_split_cnts], name='split_cnts')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        ''' lengths of character sequences in self.x '''
        self.seqlen = tf.placeholder(tf.int32, [None], name = 'seqlens')
        self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        
        with tf.device("/cpu:0"):
            ''' character embedding weights '''
            self.embedding_weights = tf.get_variable('embeddings', shape = [n_input,config['emb_size']], dtype=tf.float32, 
                                                     initializer=tf.random_uniform_initializer(-1.0, 1.0))
            ''' B, T, E [looked-up inputs] '''
            self.embedded_inputs = tf.nn.embedding_lookup(params=self.embedding_weights, ids=self.x, name="embedded_inputs")
        ''' Graph '''
        if self.get_config_option(config, 'use_split_cnts', 0) ==1:
            ''' use split counts as additional input features '''
            inputs = tf.concat([self.embedded_inputs, self.split_cnts], axis=2)
        else:
            inputs = tf.nn.tanh(self.embedded_inputs)
        ''' RNN '''
        max_seq_len = config['max_sequence_length_sen']
        lstm_outputs = self.bidi_rnn(config['cell_type'], config['n_hidden'], inputs, self.seqlen, name='')
        
        conv = self.convolution(lstm_outputs, max_seq_len, config['filter_sizes'], config['num_filters'], scope_affix='_conv_1')
        
        res = tf.concat([conv, lstm_outputs, inputs], axis=2) # creates the skip connections
        
        self.penult_and_classification(res, config, n_classes)
        self.merged_summary = tf.summary.merge_all()

    def bidi_rnn(self, cell_type, n_hidden, inputs, lens, name):
        '''
        Creates a bidir. RNN with given cell type
        '''
        with tf.variable_scope('bidi_{0}'.format(name)):
            lstm_outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw = self.build_cell(cell_type, n_hidden), 
                cell_bw = self.build_cell(cell_type, n_hidden), 
                inputs = inputs, 
                sequence_length = lens, 
                dtype = tf.float32, 
                time_major = False
                )
        ''' B, T, 2*n_hidden '''
        return tf.concat(lstm_outputs, axis=2)
    def build_cell(self, cell_type, n_hidden):
        '''
        Creates a single recurrent cell of indicated type and size.
        @param n_hidden: Number of inner units
        @param cell_type: Which one? LSTM, GRU, ...  
        '''
        if cell_type=='lstm':
            cell = tf.nn.rnn_cell.LSTMCell(n_hidden,forget_bias=1.0)
        elif cell_type=='lstm-block':
            '''
            @attention: When loading a saved model, it complains about key error for LSTMBlockCell
            '''
            cell = tf.contrib.rnn.LSTMBlockCell(n_hidden, forget_bias=1.0)
        elif cell_type=='gru':
            cell = tf.nn.rnn_cell.GRUCell(n_hidden)
        return tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.dropout_keep_prob)
        
    def convolution(self, input_, max_sequence_length, filter_sizes, num_filters, scope_affix='', reuse = None):
        '''
        Performs a Kim-convolution on the input embeddings
        @param input: Tensor of shape [batch, max_seq_len, some size] 
        @return: Tensor of shape [batch, max_seq_len, X], with X = filter_sizes * num_filters
        '''
        input_size = int(input_.get_shape()[-1])
        ''' B, T, E, 1 '''
        input_ = tf.expand_dims(input_,-1)
        '''
        CNN of the embeddings
        Create a convolution + maxpool layer for each filter size
        @note: This function does NOT stack cnns, but convolves
            the input with different filter sizes. Their outputs are concatenated.
        '''
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.variable_scope("conv_maxpool_{0}{1}".format(filter_size, scope_affix), reuse=reuse):
                '''
                Filter with height = input_size size.
                '''
                filter_shape = [filter_size, input_size, 1, num_filters]
                ''' FS, E, 1, NF '''
                W = tf.get_variable('W_filter_{0}'.format(i), shape=filter_shape, dtype=tf.float32,
                                    initializer=tf.truncated_normal_initializer(0, stddev=0.1))
                b = self.bias(num_filters, 'B_filter_{0}'.format(i))
                ''' 
                Convolution layer; apply padding (w) to keep the number of time steps.
                @todo: What about even filter sizes?  
                '''
                w = int(filter_size/2)
                padded_embedded_inputs = tf.pad(input_,  [[0, 0], [w, w], [0, 0], [0, 0]], "CONSTANT")
                '''
                - default: [B, T, 1, NF]
                - extended: [B, T, E-FS/2, NF]
                '''
                conv = tf.nn.conv2d(
                    padded_embedded_inputs,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv_op")
                
                ''' nonlinearity; "relu" in Kim paper '''
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="conv_nonlin")
                pooled_outputs.append(tf.reshape(h, [-1, max_sequence_length, num_filters]))

        '''
        Combine all pooled features. This is the input to the next layer.
        B, T, \sum filter_sizes
        for a constant filter size and u different filters, the size is:
        B, T, u*FS
        '''
        return tf.concat(pooled_outputs, 2)
    def penult_and_classification(self, rnn_outputs, config, n_classes):
        '''
        Builds the penult dense layer and creates the classification mechanism for a
        standard architecture.
        @param rnn_outputs: Tensor of size [batch] x [sequence length] x [some hidden dimension]. Can be created by concatenating step-wise LSTM outputs.  
        '''
        
        ''' B*T, last_dimension '''
        rnn_outputs = tf.reshape(rnn_outputs, [-1, int(rnn_outputs.get_shape()[-1])])
        
        ''' B, T, n_classes '''
        self.logits = tf.reshape(
                                 tf.matmul(
                                           rnn_outputs,
                                           self.weight(int(rnn_outputs.get_shape()[-1]), n_classes, 'out_weights')
                                           ) + self.bias(n_classes, 'out_bias'), 
                                 shape=[-1, config['max_sequence_length_sen'], n_classes])
        ''' B, T '''
        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=self.logits, name="sparse_softmax_fun")
        
        '''
        shape of @soft==shape of @logits, but normalized
        '''
        self.soft = tf.nn.softmax(self.logits, axis=2)
        ''' B, T '''
        self.predictions = tf.argmax(self.soft, axis=2, name='predictions')
        '''
        @todo: use the seq. lengths!
        '''
        ''' has 0 or 1 at each step '''
        nc = tf.cumsum(tf.cast(tf.equal(self.predictions, self.y), tf.float32), axis=1)
        ''' cumsum(len)==num_correct for a seq. with len '''
        cs = tf.reduce_sum( 
                           tf.gather_nd(params=nc, indices=tf.stack([tf.range(tf.shape(self.x)[0]), self.seqlen-1], axis=1)) 
                           )
        len_sum = tf.cast(tf.reduce_sum(self.seqlen), dtype=tf.float32)
        self.accuracy = tf.div(cs, len_sum)
        tf.summary.scalar('accuracy', tensor=self.accuracy)
        self.num_correct = cs
        
        
        self.cost = tf.reduce_mean(cost, name='mean_cost')
        tf.summary.scalar('cost', self.cost)
        optimizer_type = config['optimizer']
        if optimizer_type=='rms':
            optim = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate, decay=0.99)
        elif optimizer_type=='adam':
            optim = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        elif optimizer_type=='sgd':
            optim = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        elif optimizer_type=='mom':
            optim = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.1)
        if 'gradient_clipping' in config and config['gradient_clipping'] > 0:
            gradient_clip = config['gradient_clipping']
            params = tf.trainable_variables()
            gradients = tf.gradients(self.cost, params)
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, gradient_clip)
            self.optimizer = optim.apply_gradients(zip(clipped_gradients, params), name='optim_apply_gradients')
        else:
            self.optimizer = optim.minimize(self.cost)
        
    def weight(self, insize, outsize, name):
        '''
        Creates a weight matrix.
        @todo Xavier initialization?
        '''
        return tf.get_variable(name, shape=[insize, outsize], dtype=tf.float32, 
                               initializer=tf.truncated_normal_initializer(0,stddev=1.0/math.sqrt(float(insize) ))
                               )
    def bias(self, outsize, name, bias_init_val = 0):
        '''
        Creates a bias vector
        '''
        return tf.get_variable(name, shape=[outsize], dtype=tf.float32, initializer=tf.constant_initializer(bias_init_val))
    def get_config_option(self, config, key, default_value):
        if key in config:
            return config[key]
        return default_value
    def get_save_name(self):
        return 'cnn_lstm'
    