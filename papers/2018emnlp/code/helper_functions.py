import numpy as np
import os,glob,codecs,sys
from collections import Counter
import defines


def clean_dir(directory):
    logfiles = glob.glob(directory)
    for f in logfiles:
        os.remove(f)


def calc_pr(P, other):
    return 100.0 * float(P)/float(P+other) if P>0 or other > 0 else 0.0


def sandhi_validation(model, is_validation, data, sess, data_directory, config):
    '''
    Evaluates on the validation or test sets.
    Greedy decoding
    Need to provide the functionality for splitting the validation set, because CNNs will take too much RAM
    when processing the complete valid. set.
    @return A dictionary with results
    '''
    max_sequence_length = config['max_sequence_length_sen']
    n_per_batch = config['valid_batch_size']
    _v_ixes = data.valid_ixes if is_validation else data.test_ixes
    if _v_ixes is None or _v_ixes.shape[0]==0:
        return 0,0
    if n_per_batch==0:
        n_per_batch = _v_ixes.shape[0]
    di_F = 0
    eq_F = 0
    prefix = 'validation' if is_validation else 'test'
    with codecs.open(data_directory + r'\{0}-details-{1}.dat'.format(prefix, model.get_save_name()), 'w', 'UTF-8') as val_file:
        max_valid_steps = int(np.ceil(_v_ixes.shape[0]/n_per_batch))
        '''
        1st row: TP, FP, FN for eq
        2nd row: ... for diff
        '''
        eval_ = np.zeros((2,3), np.int32)
        eq_ix = data.deenc_output.get_index(defines.SYM_IDENT)
        space_ix = data.deenc_input.get_index(defines.SYM_SPACE)
        seq_errs = []
        seq_no = 0
        errs_in_seq = 0
        num_corr_sens = 0
        for step in range(max_valid_steps):
            start = step * n_per_batch
            end   = min((step+1)*n_per_batch, _v_ixes.shape[0])
            v_ixes = _v_ixes[start:end]
#             N = v_ixes.shape[0]
#             pad_sym = data.deenc_input.get_index(defines.SYM_PAD)
#             in_ = np.full(shape=[N, data.max_sequence_length], fill_value=pad_sym, dtype=np.int32)
#             out_= np.full(shape=[N, data.max_sequence_length], fill_value=pad_sym, dtype=np.int32)
#             lens_ = np.zeros(shape=[N], dtype=np.int32)
#             
#             for row,ix in enumerate(v_ixes):
#                 sys.stdout.write('{0} {1} {2}\r'.format(step,row,ix));sys.stdout.flush()
#                 l = len(data.inputs[ix])
#                 lens_[row] = l
#                 for col in range(l):
#                     in_[row,col] = data.inputs[ix][col]
#                     out_[row,col] = data.outputs[ix][col]
#             spcnts = data.get_split_cnts(in_, lens_, config)        
            valid_corr, v_p, v_logits = sess.run([model.num_correct,model.predictions,model.soft], 
                feed_dict={
                           model.x:data.inputs[v_ixes,:], 
                           model.y:data.outputs[v_ixes,:],
                           model.split_cnts:data.split_cnts[v_ixes,:,:],
                           model.seqlen:data.seq_lens[v_ixes],
                           model.dropout_keep_prob:1.0
                           })
            
            
            out = data.outputs[v_ixes,:]
            inp = data.inputs[v_ixes,:]
            
            for row in range(v_p.shape[0]): # each row = 1 line / sentence
                val_file.write('#--------\n')
                errs_in_sen = 0
                for i in range(data.seq_lens[v_ixes][row]):
                    eval_str = 'SAME_COR'
                    if out[row,i]==eq_ix: # gold: equal
                        if v_p[row,i]==eq_ix: 
                            eval_[0,0]+=1
                        else:
                            eval_str = 'SAME_ERR'
                            eval_[0,2]+=1 # fn for eq
                            eval_[1,1]+=1 # fp for diff
                            errs_in_seq+=1
                            errs_in_sen+=1
                    else: # gold: different
                        if v_p[row,i]==out[row,i]: 
                            eval_[1,0]+=1
                            eval_str = 'DIFF_COR'
                        else: 
                            eval_str = 'DIFF_ERR'
                            eval_[0,1]+=1 # fp for eq
                            eval_[1,2]+=1 # fn for diff
                            errs_in_seq+=1
                            errs_in_sen+=1
                    if inp[row,i]==space_ix:
                        ''' Update the string-based statistics. '''
                        seq_errs.append(errs_in_seq)
                        errs_in_seq = 0
                    line_det = '{0}\t{1}\t{2}\t{3}\t{4}\t{5:.3f}\t{6}\n'.format(seq_no, i,
                        data.deenc_input.get_sym(inp[row,i]),
                        data.deenc_output.get_sym(out[row,i]),
                        data.deenc_output.get_sym(v_p[row,i]),
                        v_logits[row,i, v_p[row,i] ], eval_str)
                    val_file.write(line_det)
                if errs_in_sen==0:
                    num_corr_sens+=1
                seq_no+=1
        t_num = np.sum( data.seq_lens[_v_ixes] )
        eq_P = calc_pr(eval_[0,0], eval_[0,1])
        eq_R = calc_pr(eval_[0,0], eval_[0,2])
        if eq_P > 0 or eq_R > 0:
            eq_F = 2*eq_P*eq_R/(eq_P+eq_R)
        di_P = calc_pr(eval_[1,0], eval_[1,1])
        di_R = calc_pr(eval_[1,0], eval_[1,2])
        if di_P > 0 or di_R > 0:
            di_F = 2*di_P*di_R/(di_P + di_R)
        print(' {0} (approx.): {1:.2f} ({2})'.format(prefix, 100.0*float(int(valid_corr))/float(max_sequence_length*len(v_ixes)), valid_corr))
        print('  details (character-based): {0:.2f}'.format(100.0*float(eval_[0,0] + eval_[1,0])/float(t_num)))
        print('    equal PRF: {0:.2f} {1:.2f} {2:.2f}'.format(eq_P, eq_R, eq_F))
        print('     diff PRF: {0:.2f} {1:.2f} {2:.2f}'.format(di_P, di_R, di_F))
        co = Counter(seq_errs)
        string_eval = [0,0,0,0]
        for num,cnt in co.items():
            if num==0 or num==1 or num==2:
                string_eval[num] = cnt
            else:
                string_eval[3]+=cnt
        string_acc = 'NA'
        if len(seq_errs) > 0:
            ''' can be empty, e.g. for evaluation on German compounds (no sentences, only single strings) '''
            seq_errs = np.asarray(seq_errs, np.float32)
            num_corr_strings = np.where(seq_errs==0)[0].shape[0]
            string_acc = 100.0 * float(num_corr_strings) / float(seq_errs.shape[0])
            print('  details (string-based): {0:.2f}'.format(string_acc))
        print('    errs/string: 0: {0}, 1: {1}, 2: {2}, 3+: {3}'.format(string_eval[0], string_eval[1], string_eval[2], string_eval[3] ))
    res = {'eq_P':eq_P, 'eq_R':eq_R, 'eq_F':eq_F, 'di_P':di_P, 'di_R':di_R, 'di_F':di_F, 'string_acc':string_acc, 
           'sen_acc' : 100.0 * float(num_corr_sens) / float(seq_no),
           'str_err_0':string_eval[0], 'str_err_1': string_eval[1], 'str_err_2':string_eval[2], 'str_err_3p':string_eval[3]}
    return res


def analyze_text(path_in, path_out, predictions_ph, x_ph, split_cnts_ph, seqlen_ph, dropout_ph, loader, session, verbose = False):
    '''
    Apply a trained model to a text.
    
    The xxx_ph parameters are the placeholders that are fed + the prediction.
    Required to make the function compatible with train and application settings.
    '''
    if verbose==True:
        print('Analyzing {0} ...'.format(path_in) )
    seqs,lens,splitcnts,lines_orig = loader.load_external_text(path_in)
    if seqs is None:
        print('Something went wrong while loading {0}'.format(path_in) )
        return
    with codecs.open(path_out, 'w', 'UTF-8') as f:
        batch_size = 500
        start = 0
        P = None
        while True:
            end = min(start+batch_size, seqs.shape[0])
            if end<=start:
                break
            if verbose==True:
                sys.stdout.write(' lines {0} => {1}\r'.format(start,end) ); sys.stdout.flush();
            p = session.run(predictions_ph, feed_dict = {
                x_ph:seqs[start:end,:],
                split_cnts_ph:splitcnts[start:end,:,:],
                seqlen_ph:lens[start:end],
                dropout_ph:1.0
                })
            if P is None:
                P = p
            else:
                P = np.concatenate([P,p], axis=0)
            start = end
        if verbose==True:
            print('')
        ''' decode and write to the file '''
        for i in range(P.shape[0]):
            pred_sym_seq = [loader.deenc_output.get_sym(x) for x in P[i, :lens[i] ] ] # skip the last symbol
            pred_str = ''
            for p_s, o_s in zip(pred_sym_seq[1:], lines_orig[i][1:] ):
                if p_s==defines.SYM_IDENT:
                    pred_str+=o_s
                elif p_s==defines.SYM_SPLIT:
                    pred_str+=o_s + '-'
                else:
                    pred_str+=p_s
            pred_str = loader.internal_transliteration_to_unicode(pred_str).replace('- ', ' ').replace('= ', ' ')
            f.write(pred_str + '\n')
        if verbose==True:
            print(' results written into {0}'.format(path_out) )