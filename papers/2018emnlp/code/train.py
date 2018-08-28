'''

THIS IS THE MAIN SCRIPT!

'''
import tensorflow as tf
import numpy as np
import helper_functions,data_loader,model,configuration,defines
import sys,time,os,json,datetime,shutil

def save_model(mo, model_dir_, session):
    model_dir = os.path.normpath( os.path.join(os.getcwd(), model_dir_) ) # join an absolute (cwd) and a relative path.
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir, ignore_errors=True)
        print('---- rmtree for {0}'.format(model_dir))
        time.sleep(0.5)
    '''
    @attention: LSTMBlockCell makes problems. 
    '''
    inputs = {
        defines.KEY_X : mo.x,
        defines.KEY_SEQLENS : mo.seqlen,
        defines.KEY_DROPOUT : mo.dropout_keep_prob,
        defines.KEY_SPLIT_CNTS : mo.split_cnts
        }
    outputs = {
        defines.KEY_PREDICTIONS : mo.predictions
        }
    tf.saved_model.simple_save(session, model_dir, inputs, outputs)
    
    

config = configuration.config
language = 'sanskrit' #'german' # 

if language=='sanskrit':
    data_directory_input = '../data/input'
    data_file_names = ['sandhi-data-sentences-train.dat', 'sandhi-data-sentences-test.dat', 'sandhi-data-sentences-validation.dat']
    data_directory_result= '../data/output'
    protocol_directory = '../data/protocol' # for the json file
    test_text_path =  data_directory_input + '/trbh.txt'

load_data_into_ram = True
with data_loader.DataLoader(data_directory_input, config, load_data_into_ram) as data:
    try:
        config['data'] = data
        print(' data: Got {0} records'.format( data.seq_lens.shape[0]))
        
        
        n_input = len(data.deenc_input.sym2idx)
        n_classes = len(data.deenc_output.sym2idx)
        
        ''' Create the model '''
        graph_train = tf.Graph()
        #with tf.variable_scope('model'):
        with graph_train.as_default():
            M = model.Model(config, n_input, n_classes, data.n_split_cnts)
        
        current_learning_rate = config['learning_rate']
        lr_schedule_type = M.get_config_option(config, 'has_lr_schedule', 0)
        progress_step = 5
        
        helper_functions.clean_dir('tf-log/*')
        
        session_conf = tf.ConfigProto(intra_op_parallelism_threads=2, inter_op_parallelism_threads=2)
        with tf.Session(config = session_conf, graph=graph_train) as sess, open(data_directory_result + '/tf-output.dat', "w") as protfile:
            sess.run(tf.global_variables_initializer())
            summary_writer = tf.summary.FileWriter('tf-log', sess.graph)
            ''' 
            This is the best F score on the validation set.
            If new_F > best_avg_f_score, the model records the values on the test set,
            and the model is stored.
            '''
            best_avg_f_score = 0
            best_test_result = None
            ''' training '''
            max_sequence_length = config['max_sequence_length_sen']
            train_time_start = time.time()
            global_step = 0 # for the summary
            for epoch in range(config['max_epochs']):
                data.initialize_batch(config['batch_size'])
                step = 0
                total_corr_train = 0
                total_steps_train = 0
                prot_train = ""
                start_time = time.time()
                n_trained = 0
                avg_train_cost = 0.0
                while data.get_next_batch():
                    ''' training '''
                    n_trained+=config['batch_size'] * max_sequence_length
                    #total_steps_train+=config['batch_size'] * max_sequence_length
                    total_steps_train+=np.sum(data.batch_seq_lens)
                    
                    fd = {
                          M.x: data.batch_x,
                          M.y: data.batch_y,
                          M.seqlen:data.batch_seq_lens,
                          M.dropout_keep_prob:config['dropout'],
                          M.learning_rate:current_learning_rate
                             }
                    if config['use_split_cnts']==1:
                        fd[M.split_cnts] = data.batch_split_cnts
                    
                    _, c, num_corr, acc, ms = sess.run([M.optimizer,M.cost, M.num_correct, M.accuracy, M.merged_summary], 
                                              feed_dict=fd)
                    summary_writer.add_summary(ms, global_step)
                    global_step+=1
                    avg_train_cost+=c
                    total_corr_train+= num_corr
                    step+=1
                    if step % progress_step==0:
                        sys.stdout.write(' step {0}, train cost: {1:.5f}, train corr: {2} ({3:.2f})\r'.format(step, 
                            avg_train_cost/float(n_trained), int(total_corr_train), 100.0*float(total_corr_train)/float(total_steps_train)
                            ))
                        sys.stdout.flush()
                    if (step % config['display_step']==0) or data.has_more_data()==False:
                        end_time = time.time()
                        print('')
                        print('step {0}, train cost: {1:.5f}, train corr: {2} ({3:.2f}), duration: {4:.2f}'.format(step, 
                            avg_train_cost/float(n_trained), int(total_corr_train), 100.0*float(total_corr_train)/float(total_steps_train),
                            float(end_time - start_time)))
                        '''
                        run on an external text
                        '''
                        if len(test_text_path) > 0:
                            #M.analyze_text(test_text_path, data_directory_result + '/extern-result-final.dat', data, sess)
                            helper_functions.analyze_text(test_text_path, data_directory_result + '/extern-result-final.dat', 
                                                          M.predictions, M.x, M.split_cnts, M.seqlen, M.dropout_keep_prob, 
                                                          data, sess, verbose=False)
                        
                        ''' evaluation '''
                        if not data.valid_ixes is None:
                            res = helper_functions.sandhi_validation(M, True, # = is_validation 
                                data, sess, data_directory_result, config)
                            avg_f = 0.5 * (res['eq_F'] + res['di_F'])
                            if avg_f > best_avg_f_score:
                                best_avg_f_score = avg_f
                                save_model(M, config['model_directory'], sess)
                                print(' *** model with avg F score of {0:.2f} stored'.format(avg_f))
                                ''' evaluate on the test set '''
                                best_test_result = helper_functions.sandhi_validation(M, False, # = is_validation 
                                    data, sess, data_directory_result, config)
                                best_test_result['best_train_epoch'] = epoch
                                if lr_schedule_type==2:
                                    current_learning_rate = min(config['learning_rate'], current_learning_rate*1.2)
                                    print(' ^^^^ changed learning rate to {0}'.format(current_learning_rate))
                            else:
                                if lr_schedule_type!=0:
                                    current_learning_rate*=0.5
                                    print(' ____ changed learning rate to {0}'.format(current_learning_rate))
                        else: # test set only
                            res = helper_functions.sandhi_validation(M, False, # = is_validation 
                                data, sess,  data_directory_result, config)
                            avg_f = 0.5 * (res['eq_F'] + res['di_F'])
                            best_test_result = res#
                            if avg_f > best_avg_f_score:
                                best_avg_f_score = avg_f
                                '''
                                @todo store the model
                                '''
                                print(' *** avg F score: {0:.2f}'.format(avg_f))
                            if lr_schedule_type==3:
                                current_learning_rate*=0.9
                                print(' --[lr type 3] changed learning rate to {0}'.format(current_learning_rate))
                            
                        start_time = time.time()
                        total_corr_train = 0
                        total_steps_train = 0
                    #step+=1
                print(" --- Epoch {0} finished".format(epoch))
            print("Optimization Finished!")
            ''' write the data of the final evaluation in a protocol file '''
            if not best_test_result is None:
                res = best_test_result
            else:
                res = helper_functions.sandhi_validation(M, False, # = is_validation 
                    data, sess, data_directory_result, config)
            res['date_time'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            res['train_duration'] = time.time() - train_time_start
            res['full_name'] = M.get_save_name()
            res['num_train_samples'] = data.train_ixes.shape[0]
            for key,val in config.items():
                if key!='filter_sizes' and key!='data':
                    res[key] = val
            for fs in range(4):
                res['filter_size_{0}'.format(fs+1)] = config['filter_sizes'][fs] if fs < len(config['filter_sizes']) else 0
            json_path = protocol_directory + '/training-protocol-final.json'
            json_data = []
            if os.path.exists(json_path):
                with open(json_path, 'r') as res_file:
                    json_data = json.load(res_file)
            json_data.append(res)
            with open(json_path, 'w') as res_file:
                json.dump(json_data, res_file)
    except KeyboardInterrupt:
        print('interrupted!')