import numpy as np
import tensorflow as tf
import codecs,os
import settings,model,constants

parameters = {
    'use.child.head.bigrams' : False,
    'lex.emb.adapt.size' : constants.LEX_ADAPT_SIZE,
    'train.lex.emb' : True,
    'learning_rate' : 0.001,
    'dropout_rate' : 0.2,
    'epochs' : 2,
    'batch_size' : 32, # don't use larger values
    'hidden.sizes' : [20],
    'rnn.type' : 'lstm',
    'rnn.size' : 20, # 10 better?
    'pyr.size' : 20, #
    'penult.size' : 40,
    'model' : 'baseline' # # 'bidirnn' # 'sum'#
    }


''' 
Read the data. 
'''
if not os.path.exists(settings.npz_path):
    exit()
zfile = np.load(settings.npz_path)
folds = zfile['folds']
feat  = zfile['features']
tars  = zfile['targets']
seq_lens = zfile['seqlens']
occs = zfile['occs']
if occs.shape[0]!=tars.shape[0]:
    print('occs <> tars')
    exit()
''' for each row in @tars: to which sentence = row in @feat does it belong? '''
tar2seq = zfile['tar2seq']

pretrained_embs = np.genfromtxt(settings.data_input_directory + 'embeddings.dat', delimiter = ' ')

unigrams = np.arange(feat.shape[2])
with codecs.open(settings.data_output_directory + '{0}.result'.format(parameters['model']), 'w', 'UTF-8') as f,\
    codecs.open(settings.data_output_directory + '{0}.details'.format(parameters['model']), 'w', 'UTF-8') as f_details,\
    codecs.open(settings.data_output_directory + '{0}.occs'.format(parameters['model']), 'w', 'UTF-8') as f_occs:
    f.write('fold gold silver\n')
    for fold in range(settings.num_folds):
        with tf.Graph().as_default():
            test_ixes = np.where(folds==fold)[0]
            train_ixes= np.where(folds!=fold)[0]
            bigr = settings.bigrams_seq
            trigr= settings.trigrams_seq
            if parameters['model']=='bidirnn2':
                bigr = []
                trigr= []
            with tf.variable_scope('model', reuse=None):
                model_train = model.Model(pretrained_embs, tars, tar2seq, np.copy(feat), seq_lens, parameters, 
                                          unigrams, bigr,trigr, True)
            with tf.variable_scope('model', reuse=True):
                model_test  = model.Model(pretrained_embs, tars, tar2seq, np.copy(feat), seq_lens, parameters, 
                                          unigrams, bigr,trigr, False)
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                for epoch in range(parameters['epochs']):
                    train_ixes = np.random.permutation(train_ixes)
                    total_cost = 0.0
                    total_corr = 0
                    bs = parameters['batch_size']
                    batch_steps = int(len(train_ixes)/bs)
                    for b in range(batch_steps):
                        ixes = train_ixes[b*bs:((b+1)*bs)]
                        _,co,corr = sess.run([model_train.train_op, model_train.cost, model_train.num_correct], feed_dict={
                            model_train.ixes : ixes
                            })
                        total_cost+=co
                        total_corr+=corr
                    print('{0} {1}, correct: {2:.2f}'.format(epoch,total_cost, 100*float(total_corr)/float(batch_steps * bs) ))
                    if epoch % 5==0:
                        corr,preds = sess.run([model_test.num_correct, model_test.predictions], feed_dict={model_test.ixes:test_ixes})
                        print('  - Test [fold {1}]: {0:.2f} correct ---'.format(100*float(corr)/float(preds.shape[0]), fold ))
                # testing
                corr,preds,logits = sess.run([model_test.num_correct, model_test.predictions,model_test.logits], feed_dict={model_test.ixes:test_ixes})
                print('--- Test: {0:.2f} correct ---'.format(100*float(corr)/float(preds.shape[0]) ))
                test_tar = tars[test_ixes,0]
                test_occs = occs[test_ixes]
                for i in range(preds.shape[0]):
                    f.write('{0} {1} {2}\n'.format(fold, test_tar[i], preds[i] ))
                    r = np.arange(logits.shape[1])
                    l = r[(-logits[i,:]).argsort()]
                    f_details.write(' '.join([str(int(x)) for x in l]) + '\n')
                    f_occs.write(test_occs[i] + '\n')
                f.flush()
                f_details.flush()
                f_occs.flush()