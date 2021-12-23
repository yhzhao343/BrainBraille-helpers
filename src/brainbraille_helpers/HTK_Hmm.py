import os
import time
import numpy as np
from struct import pack
from .helpers import *
import subprocess
# from datetime import datetime
import datetime
from joblib import Parallel, delayed
from numpy.random import default_rng
rng = default_rng(42)

class HTK_Hmm:
  def __init__(self, temp_dir='/tmp/.gt2k_py', num_cpu=30, HTK_PATH=None,
               SRILM_PATH=None, num_states=3, skip=0, num_mixtures=1,
               use_full_cov=False, label_time_period_ms=200,
               init_HRest_min_var = 0.0001, init_HRest_num_iter = 40,
               init_HRest_min_samp=1, HCompV_min_var = 0.0001,
               bi_tri_phone_HERest_min_var = 0.0001,
               embedded_training_HERest_min_var = 0.0001,
               embedded_training_HERest_t_f = 0,
               embedded_training_HERest_t_i = None,
               embedded_training_HERest_t_l = None,
               HInit_min_samp=2, HInit_min_var=0.0001, DEBUG= True,
               SUPRESS_ALL_SUBPROCESS_OUTPUT = False, PRINT_CMD=True,
               TRACE_LEVEL=1, num_embedded_training=6,
               custom_hmm_obj_for_token_dict = {},
               custom_hmm_string_for_token_dict = {},
               bi_tri_phone_edcmd = None, num_bi_tri_phone_re_estimate = 2,
               tied_state_bi_tri_phone_edcmd = None, add_monophone_back = False,
               use_tied_states = False,
               num_tied_state_bi_tri_phone_re_estimate = 2,
               bi_tri_phone_tied_HERest_min_var = 0.0001,
               bi_tri_phone_tied_HERest_min_samp = 1,
               bi_tri_phone_tied_HERest_t_f = 0,
               bi_tri_phone_tied_HERest_t_i = None,
               bi_tri_phone_tied_HERest_t_l = None,
               TB_threshold = 5, convert_to_tied_state_threshold=5.0,
               dict_string=None, grammar_string = None,
               word_lattice_string = None):

    # NOTE: label_time_period_ms is just a number to make HTK happy! Not actual value!
    # TODO: add pruning and other parameters.
    t = str(datetime.datetime.now()).replace(':', '_').replace(' ', '_')
    self.temp_path = f'{temp_dir.rstrip()}/{t}-{str(rng.random())[2:]}'
    os.makedirs(self.temp_path)
    self.num_cpu = num_cpu if num_cpu is not None else mp.cpu_count() - 1
    self.HTK_PATH = HTK_PATH if HTK_PATH is not None else os.environ.get('HTK_PATH')
    self.SRILM_PATH = SRILM_PATH if SRILM_PATH is not None else os.environ.get('SRILM_PATH')
    self.num_states= num_states + 2
    self.label_time_period_ms = label_time_period_ms
    self.skip = skip
    self.num_mixtures =  num_mixtures
    self.use_full_cov = use_full_cov
    self.tokens_path = f'{self.temp_path}/tokens'
    self.init_hmm_path = f'{self.temp_path}/init_hmm'
    self.hmm0_path = f'{self.temp_path}/hmm.0'
    self.hmm1_path = f'{self.temp_path}/hmm.1'
    self.hmm2_path = f'{self.temp_path}/hmm.2'
    self.hmm3_path = f'{self.temp_path}/hmm.3'
    os.mkdir(self.init_hmm_path)
    os.mkdir(self.hmm0_path)
    os.mkdir(self.hmm1_path)
    os.mkdir(self.hmm2_path)
    os.mkdir(self.hmm3_path)
    self.label_mlf_path = f'{self.temp_path}/labels.mlf'
    self.ext_path = f'{self.temp_path}/ext'
    os.mkdir(self.ext_path)
    self.sample_list_path = f'{self.temp_path}/sample.list'
    self.DEBUG = DEBUG
    self.SUPRESS_ALL_SUBPROCESS_OUTPUT = SUPRESS_ALL_SUBPROCESS_OUTPUT
    self.PRINT_CMD = PRINT_CMD
    self.TRACE_LEVEL = TRACE_LEVEL
    self.num_embedded_training = num_embedded_training if (num_embedded_training > 1) else 1
    self.custom_hmm_obj_for_token_dict = custom_hmm_obj_for_token_dict
    self.custom_hmm_string_for_token_dict = custom_hmm_string_for_token_dict
    self.init_HRest_min_samp = init_HRest_min_samp
    self.init_HRest_min_var = init_HRest_min_var
    self.init_HRest_num_iter = init_HRest_num_iter
    self.HInit_min_samp = HInit_min_samp
    self.HInit_min_var = HInit_min_var
    self.HCompV_min_var = HCompV_min_var
    self.embedded_training_HERest_t_f = embedded_training_HERest_t_f
    self.embedded_training_HERest_t_i = embedded_training_HERest_t_i
    self.embedded_training_HERest_t_l = embedded_training_HERest_t_l
    self.bi_tri_phone_HERest_min_var = bi_tri_phone_HERest_min_var
    self.embedded_training_HERest_min_var = embedded_training_HERest_min_var
    self.bi_tri_phone_tied_HERest_min_var = bi_tri_phone_tied_HERest_min_var
    self.bi_tri_phone_tied_HERest_min_samp = bi_tri_phone_tied_HERest_min_samp
    self.bi_tri_phone_tied_HERest_t_f = bi_tri_phone_tied_HERest_t_f
    self.bi_tri_phone_tied_HERest_t_i = bi_tri_phone_tied_HERest_t_i
    self.bi_tri_phone_tied_HERest_t_l = bi_tri_phone_tied_HERest_t_l
    self.add_monophone_back = add_monophone_back

    self.bi_tri_phone_edcmd = bi_tri_phone_edcmd if isinstance(bi_tri_phone_edcmd, str) else None
    if self.bi_tri_phone_edcmd is not None:
      self.bi_tri_phone_edcmd = self.bi_tri_phone_edcmd.rstrip()
      if not (self.bi_tri_phone_edcmd.endswith('LC') or self.bi_tri_phone_edcmd.endswith('RC') or self.bi_tri_phone_edcmd.endswith('TC')):
        self.bi_tri_phone_edcmd = None

    if self.bi_tri_phone_edcmd is None:
      self.bi_tri_phone_edcmd_path = None
    else:
      file_name = self.bi_tri_phone_edcmd.replace('\n', '')
      file_name = file_name.replace(' ', '-')
      self.bi_tri_phone_edcmd_path = f'{self.temp_path}/mk_{file_name}.hed'
      write_file(f'{self.bi_tri_phone_edcmd}\n', self.bi_tri_phone_edcmd_path)

    self.use_tied_states = use_tied_states
    self.tied_state_bi_tri_phone_edcmd_path = f'{self.temp_path}/mk_tied_states.hed'
    self.tied_state_bi_tri_phone_edcmd = tied_state_bi_tri_phone_edcmd if isinstance(tied_state_bi_tri_phone_edcmd, str) else None
    if self.tied_state_bi_tri_phone_edcmd is not None:
      write_file(self.tied_state_bi_tri_phone_edcmd, self.tied_state_bi_tri_phone_edcmd_path)
      self.use_tied_states = True
    self.TB_threshold = TB_threshold
    self.bi_tri_phone_mlf_path = f'{self.temp_path}/bi_or_tri_phone_label.mlf'
    self.bi_tri_tokens_path = f'{self.temp_path}/bi_or_tri_tokens'
    self.bi_tri_phone_stats_path = f'{self.temp_path}/bi_tri_phone_stats'
    self.bi_tri_tokens = None
    self.bi_tri_phone_mlf_content = None
    self.num_bi_tri_phone_re_estimate = num_bi_tri_phone_re_estimate
    self.predict_ext_path = f'{self.temp_path}/predict_ext'
    os.mkdir(self.predict_ext_path)
    self.dict_path = f'{self.temp_path}/dict'
    self.grammar_path = f'{self.temp_path}/grammar'
    self.word_lattice_path = f'{self.temp_path}/word_lattice.slf'
    self.tiedlist_path = f'{self.temp_path}/bi_or_tri_tiedlist'
    self.wordlist_path = f'{self.temp_path}/wordlist'
    self.tiedlist = None
    self.convert_to_tied_state_threshold = convert_to_tied_state_threshold
    self.num_tied_state_bi_tri_phone_re_estimate = num_tied_state_bi_tri_phone_re_estimate

    if isinstance(dict_string, str):
      self.dict_string = f'{dict_string.rstrip()}\n'
      self.wordlist = list(set([l.split(' ')[0] for l in self.dict_string.split('\n') if len(l)]))
      write_file(self.dict_string, self.dict_path)
      write_file('\n'.join(self.wordlist) + '\n', self.wordlist_path)
    else:
      self.dict_string = None

    if isinstance(grammar_string, str):
      self.grammar_string = f'{grammar_string.rstrip()}\n'
      write_file(self.grammar_string, self.grammar_path)
    else:
      self.grammar_string = None

    if isinstance(word_lattice_string, str):
      self.word_lattice_string = f'{word_lattice_string.rstrip()}\n'
      write_file(self.word_lattice_string, self.word_lattice_path)
    else:
      self.word_lattice_string = None

  def _hmm_HResults(self, X, y, y_timestamp = None, x_sample_periods_ms=None, i_tokens_each_state = 3, n_best = 1, insertion_penalty = 0.0, token_label=True, use_nist = False):
    self.predict(X, x_sample_periods_ms=None, i_tokens_each_state = i_tokens_each_state, n_best = n_best, insertion_penalty = insertion_penalty, token_label = token_label)
    if y_timestamp is None:
      y_timestamp = [[[i*self.label_time_period_ms, (i + 1)*self.label_time_period_ms] for i in range(len(y_i))] for y_i in y]
    label_file_lab_paths = [f'{self.predict_ext_path}/predict_sample_{i + 1}.lab' for i in range(len(y))]
    label_contents = [ ''.join([f'{w}\n' for start_end, w in zip(sent_y_timestamp, y_i)]) + '.\n' for sent_y_timestamp, y_i in zip(y_timestamp, y)]
    results = Parallel(n_jobs = self.num_cpu)(delayed(self.run_HResults_helper)(content, label_path, predict_mlf_file_name, token_label, use_nist) for content, label_path, predict_mlf_file_name in zip(label_contents, label_file_lab_paths, self.predict_mlf_file_names))
    return results


  def run_HResults_helper(self, content, label_path, predict_mlf_file_name, token_label=True, use_nist=False):
    write_file(content, label_path)
    cmd = f'{self.HTK_PATH}/HResults'
    token_path = self.tokens_path if token_label else self.wordlist_path
    param_list = [
      '-s',
      '-p',
      # '-n',
      token_path,
      predict_mlf_file_name
    ]
    if use_nist:
      param_list = ['-n'] + param_list
    self._run_subprocess_helper(cmd, param_list)

  def check_tied_state_to_state(self, state_str):
    plus_ind = state_str.find('+')
    minus_ind = state_str.find('-')
    state = ''
    if (plus_ind >= 0) and (minus_ind >= 0):
      state = state_str[minus_ind+1 : plus_ind]
    elif plus_ind >= 0:
      state = state_str.split('+')[0]
    elif minus_ind >= 0:
      state = state_str.split('-')[1]
    else:
      state = state_str
    return state

  def predict(self, X, x_sample_periods_ms=None, i_tokens_each_state = 3, n_best = 1, insertion_penalty = 0.0, token_label=True):
    len_X = len(X)
    predict_ext_file_path = [f'{self.predict_ext_path}/predict_sample_{i+1}' for i in range(len_X)]
    if x_sample_periods_ms is None:
      x_sample_periods_ms = [x_i.shape[1] * self.sample_period_ms for x_i in X]
    results = Parallel(n_jobs=self.num_cpu)(delayed(self._export_to_ext)(x, ext_file_path, sample_data_period_ms) for x, ext_file_path, sample_data_period_ms in zip(X, predict_ext_file_path, x_sample_periods_ms))

    predict_ext_file_names = [f'{path}.ext' for path in predict_ext_file_path]
    predict_mlf_file_names = [f'{path}.mlf' for path in predict_ext_file_path]
    self.fitted_hmm_model_path = f'{self.predict_ext_path}/fitted_hmm_model'
    write_file(self.fitted_hmm_model, self.fitted_hmm_model_path)
    self.predict_mlf_file_names = predict_mlf_file_names
    results = Parallel(n_jobs=self.num_cpu)(delayed(self._run_hvite)(input_data_x_file, output_file, i_tokens_each_state, n_best, insertion_penalty, token_label) for input_data_x_file, output_file in zip(predict_ext_file_names, predict_mlf_file_names))
    parsed_results = Parallel(n_jobs=self.num_cpu)(delayed(self._parse_hvite_mlf)(name) for name in self.predict_mlf_file_names)

    return parsed_results

  def _parse_hvite_mlf(self, mlf_file_path):
    mlf_content = load_file(mlf_file_path)
    return [self.check_tied_state_to_state(l.split(' ')[2]) for l in mlf_content.split('\n')[2:-2]]

  def _run_hvite(self, input_data_x_file, output_file, i_tokens_each_state = 3, n_best=1, insertion_penalty = 0.0, token_label=True):
    cmd = f'{self.HTK_PATH}/HVite'
    tokens_path = self.bi_tri_tokens_path if self.bi_tri_tokens is not None else self.tokens_path
    tokens_path = self.tiedlist_path if (self.tiedlist is not None) else tokens_path
    # output_l_file = ''.join(output_file.split('.')[0:-1]) + '_l.mlf'
    param_list = [
      '-p', f'{insertion_penalty:.6f}',
      '-H', self.fitted_hmm_model_path,
      '-i', output_file,
      '-w', self.word_lattice_path,
      self.dict_path,
      tokens_path,
      input_data_x_file
    ]
    if token_label:
      param_list = ['-m'] + param_list
    if n_best > 1:
      param_list = ['-n',  int(i_tokens_each_state), int(n_best)] + param_list
    res = self._run_subprocess_helper(cmd, param_list)
    return res.stdout.decode('utf-8')
    # print('----------')
    # print(str(res.stdout))

  def fit(self, X, y, y_timestamp=None):
    # print((len(X), len(y)), (X[0].shape, len(y[0])))
    if len(X) != len(y):
      raise Exception('X and y input length does not match')
    X = [np.array(x_i) for x_i in X]
    self.vec_size = X[0].shape[1]
    self.tokens = sorted(np.unique([label for sent in y for label in sent]))
    write_file(''.join([f'{tok}\n' for tok in self.tokens]), self.tokens_path)
    if self.dict_string is None:
      self.dict_string = ''.join([f'{tok} {tok}\n' for tok in self.tokens])
      write_file(self.dict_string, self.dict_path)
      write_file(''.join([f'{tok}\n' for tok in self.tokens]), self.wordlist_path)
    if self.grammar_string is None:
      words_dictionary =  {l[0]:l[1:] for l in [line.split() for line in self.dict_string.split('\n')] if (len(l) > 0)}
      self.grammar_string = f'$word = {" | ".join(words_dictionary.keys())} ;\n\n( < $word > )'
      write_file(self.grammar_string, self.grammar_path)
    if self.word_lattice_string is None:
      self._hmm_gen_word_lattice_from_grammar()

    default_htk_hmm_obj = self._gen_htk_hmm_obj(self.num_states, self.vec_size, self.skip, self.num_mixtures, self.use_full_cov)
    default_htk_hmm_def_string = self._hmm_def_2_hmm_str(default_htk_hmm_obj)
    token_htk_def_string_dict = {t: default_htk_hmm_def_string for t in self.tokens}
    # Override default simple hmm def with custom defination if provided
    for token, custom_hmm_obj in self.custom_hmm_obj_for_token_dict.items():
      token_htk_def_string_dict[token] = self._hmm_def_2_hmm_str(custom_hmm_obj)
    for token, custom_hmm_string in self.custom_hmm_string_for_token_dict.items():
      token_htk_def_string_dict[token] = custom_hmm_string

    Parallel(n_jobs=self.num_cpu)(delayed(write_file)(content, path) for content, path in zip(token_htk_def_string_dict.values(), [f'{self.init_hmm_path}/{tok}' for tok in list(token_htk_def_string_dict.keys())]))

    if y_timestamp is None:
      y_timestamp = [[[i*self.label_time_period_ms, (i + 1)*self.label_time_period_ms] for i in range(len(y_i))] for y_i in y]

    ext_file_paths = [f'{self.ext_path}/sample_{i+1}' for i in range(len(y))]
    sample_data_periods_ms = [sample_i[-1][-1] for sample_i in y_timestamp]
    self.sample_period_ms = np.mean(np.array(sample_data_periods_ms) / np.array([len(x_i) for x_i in X]))

    results = Parallel(n_jobs=self.num_cpu)(delayed(self._export_to_ext)(data, data_path, sample_length_ms) for data, data_path, sample_length_ms in zip(X, ext_file_paths, sample_data_periods_ms))

    mlf_content = '#!MLF!# \n' + '\n.\n'.join([ f'"{ext_file_path}.lab"\n'+ '\n'.join([f'{start_end[0] * 10} {start_end[1] * 10} {w}' for start_end, w in zip(sent_y_timestamp, y_i)]) for ext_file_path, sent_y_timestamp, y_i in zip(ext_file_paths, y_timestamp, y)])
    write_file(mlf_content, self.label_mlf_path)

    data_sample_list = [f'{ext_file_path}.ext' for ext_file_path in ext_file_paths]
    write_file('\n'.join(data_sample_list), self.sample_list_path)

    Parallel(n_jobs=self.num_cpu)(delayed(self._hmm_init_and_isolated_word_train)(tok) for tok in self.tokens)

    # for tok in self.tokens:
      # self._hmm_init_and_isolated_word_train(tok)

    # Embedded training
    self._hmm_run_embedded_training()

    if self.bi_tri_phone_edcmd is not None:
      max_possible_n = np.min([len(y_i) for y_i in y])
      n_token = 3 if self.bi_tri_phone_edcmd.endswith('TC') else 2
      if n_token > max_possible_n:
        raise Exception(f'supplied y samples does not support training for {n_token}-tokens. Can support at most {max_possible_n}-tokens')
      self._hmm_fit_bi_tri_phone()

  def _hmm_gen_word_lattice_from_grammar(self):
    cmd = f'{self.HTK_PATH}/HParse'
    param_list = [self.grammar_path ,self.word_lattice_path]
    self._run_subprocess_helper(cmd, param_list)
    self.word_lattice_string = load_file(self.word_lattice_path)

  def _hmm_fit_bi_tri_phone(self):
    self._hmm_convert_label_to_bi_tri_phone()
    self._hmm_convert_to_bi_tri_phone()
    self._hmm_bi_tri_phone_re_estimate()
    if self.use_tied_states:
      self._hmm_convert_to_tied_state_tri_phone()
      self._hmm_bi_tri_tied_state_re_estimate()

  def _hmm_bi_tri_tied_state_re_estimate(self):
    current_hmm_num = 4 + self.num_embedded_training + self.num_bi_tri_phone_re_estimate
    t_params = [self.embedded_training_HERest_t_f, self.embedded_training_HERest_t_i, self.embedded_training_HERest_t_l]
    t_param_str = ' '.join([str(p) if p is not None else '' for p in t_params]).rstrip()
    cmd = f'{self.HTK_PATH}/HERest'
    for i in range(current_hmm_num, current_hmm_num + self.num_tied_state_bi_tri_phone_re_estimate):
      input_path = f'{self.temp_path}/hmm.{i}/newMacros'
      output_path = f'{self.temp_path}/hmm.{i + 1}'
      os.mkdir(output_path)
      param_list = [
        '-m', self.bi_tri_phone_tied_HERest_min_samp,
        '-v', self.bi_tri_phone_tied_HERest_min_var,
        '-S', self.sample_list_path,
        '-H', input_path,
        '-M', output_path,
        '-I', self.bi_tri_phone_mlf_path,
        self.tiedlist_path
      ]
      if len(t_param_str) > 0:
        param_list = ['-t', t_param_str] + param_list
      self._run_subprocess_helper(cmd, param_list)
    self.fitted_hmm_model = load_file(f'{output_path}/newMacros')

  def _hmm_convert_to_tied_state_tri_phone(self):
    cmd = f'{self.HTK_PATH}/HHEd'
    if self.tied_state_bi_tri_phone_edcmd is None:
      self.tied_state_bi_tri_phone_edcmd = f'RO {str(self.convert_to_tied_state_threshold)} {self.bi_tri_phone_stats_path}\n'
      for tok in self.tokens:
        self.tied_state_bi_tri_phone_edcmd += f'QS "L_{tok}" {{{tok}-*}}\nQS "R_{tok}" {{*+{tok}}}\n'
      for state_i in range(2, self.num_states):
        for tok in self.tokens:
          self.tied_state_bi_tri_phone_edcmd += f'TB {str(self.TB_threshold)} "ST_{tok}_{state_i}_" {{("{tok}","*-{tok}+*","{tok}+*","*-{tok}").state[{state_i}]}}\n'
      # print(self.tied_state_bi_tri_phone_edcmd)
      self.tied_state_bi_tri_phone_edcmd += f'\nAU {self.bi_tri_tokens_path}\nCO {self.tiedlist_path}\n\nST "trees"\n'
    write_file(self.tied_state_bi_tri_phone_edcmd, self.tied_state_bi_tri_phone_edcmd_path)
    # print(self.tied_state_bi_tri_phone_edcmd)
    current_hmm_num = 3 + self.num_embedded_training + self.num_bi_tri_phone_re_estimate
    triphone_hmm_path = f'{self.temp_path}/hmm.{current_hmm_num}/newMacros'
    tied_triphone_hmm_folder = f'{self.temp_path}/hmm.{current_hmm_num + 1}'
    tied_triphone_hmm_path = f'{tied_triphone_hmm_folder}/newMacros'
    os.mkdir(tied_triphone_hmm_folder)
    param_list = [
      '-H', triphone_hmm_path,
      '-w', tied_triphone_hmm_path,
      self.tied_state_bi_tri_phone_edcmd_path,
      self.bi_tri_tokens_path
    ]
    self._run_subprocess_helper(cmd, param_list)
    self.tiedlist = [tied_tok for tied_tok in load_file(self.tiedlist_path).split('\n') if (len(tied_tok) > 0)]

  def _hmm_bi_tri_phone_re_estimate(self):
    hmm_num = 3 + self.num_embedded_training
    cmd = f'{self.HTK_PATH}/HERest'
    for i in range(hmm_num, hmm_num + self.num_bi_tri_phone_re_estimate):
      input_path = f'{self.temp_path}/hmm.{i}/newMacros'
      output_path = f'{self.temp_path}/hmm.{i + 1}'
      os.mkdir(output_path)
      param_list = [
        '-v', self.bi_tri_phone_HERest_min_var,
        '-S', self.sample_list_path,
        '-H', input_path,
        '-M', output_path,
        '-I', self.bi_tri_phone_mlf_path,
        self.bi_tri_tokens_path
      ]
      if (i + 1) == (hmm_num + self.num_bi_tri_phone_re_estimate):
        param_list = ['-s', self.bi_tri_phone_stats_path] + param_list
      self._run_subprocess_helper(cmd, param_list)
    self.fitted_hmm_model = load_file(f'{output_path}/newMacros')
    self.tied_state_stats = load_file(self.bi_tri_phone_stats_path)

  def _hmm_convert_to_bi_tri_phone(self):
    hhed_cmd = self._hmm_make_hhed_cmd(self.tokens, self.bi_tri_tokens_path)
    self.convert_bi_tri_hhed_cmd_path = f'{self.temp_path}/convert_bi_tri_hhed_cmd.hed'
    write_file(hhed_cmd, self.convert_bi_tri_hhed_cmd_path)
    hmm_num = 3 + self.num_embedded_training
    mono_state_hmm_path = f'{self.temp_path}/hmm.{hmm_num - 1}/newMacros'
    tied_state_hmm_path = f'{self.temp_path}/hmm.{hmm_num}/newMacros'
    os.mkdir(f'{self.temp_path}/hmm.{hmm_num}')
    cmd = f'{self.HTK_PATH}/HHEd'
    param_list = [
      '-H', mono_state_hmm_path,
      '-w', tied_state_hmm_path,
      self.convert_bi_tri_hhed_cmd_path,
      self.tokens_path
    ]
    self._run_subprocess_helper(cmd, param_list)


  def _hmm_convert_label_to_bi_tri_phone(self):
    cmd = f'{self.HTK_PATH}/HLEd'
    param_list = [
      '-n', self.bi_tri_tokens_path,
      '-i', self.bi_tri_phone_mlf_path,
      self.bi_tri_phone_edcmd_path,
      self.label_mlf_path
    ]
    self._run_subprocess_helper(cmd, param_list)
    self.bi_tri_tokens = load_file(self.bi_tri_tokens_path).rstrip().split('\n')
    if self.add_monophone_back:
      self.bi_tri_tokens = list(set(self.tokens + self.bi_tri_tokens))
      write_file('\n'.join(self.bi_tri_tokens) + '\n', self.bi_tri_tokens_path)
    self.bi_tri_phone_mlf_content = load_file(self.bi_tri_phone_mlf_path)

  def _hmm_make_hhed_cmd(self, monotokens, tritoken_file):
    hhed_content = f'CL {tritoken_file}\n'
    if self.bi_tri_phone_edcmd.endswith('TC'):
      hhed_content += ''.join([f'TI T_{token} {{(*-{token}+*,{token}+*,*-{token}).transP}}\n' for token in monotokens])
    else:
      # bigram not tested yet
      if self.bi_tri_phone_edcmd.endswith('LC'):
        hhed_content += ''.join([f'TI T_{token} {{(*-{token}).transP}}\n' for token in monotokens])
      elif self.bi_tri_phone_edcmd.endswith('RC'):
        hhed_content += ''.join([f'TI T_{token} {{({token}+*).transP}}\n' for token in monotokens])
    return hhed_content

  def _hmm_run_embedded_training(self):
    cmd = f'{self.HTK_PATH}/HERest'
    output_path = self.hmm3_path
    t_params = [self.embedded_training_HERest_t_f, self.embedded_training_HERest_t_i, self.embedded_training_HERest_t_l]
    t_param_str = ' '.join([str(p) if p is not None else '' for p in t_params]).rstrip()
    param_list = [
      '-v', self.embedded_training_HERest_min_var,
      '-S', self.sample_list_path,
      '-d', self.hmm2_path,
      '-M', output_path,
      '-I', self.label_mlf_path,
      self.tokens_path
    ]
    if len(t_param_str) > 0:
      param_list = ['-t', t_param_str] + param_list
    self._run_subprocess_helper(cmd, param_list)

    for i in range(3, 3 + self.num_embedded_training - 1):
      input_path = f'{self.temp_path}/hmm.{i}/newMacros'
      output_path = f'{self.temp_path}/hmm.{i + 1}'
      os.mkdir(output_path)
      param_list = [
        '-v', self.embedded_training_HERest_min_var,
        '-S', self.sample_list_path,
        '-H', input_path,
        '-M', output_path,
        '-I', self.label_mlf_path,
        self.tokens_path
      ]
      if len(t_param_str) > 0:
        param_list = ['-t', t_param_str] + param_list
      self._run_subprocess_helper(cmd, param_list)
    self.fitted_hmm_model = load_file(f'{output_path}/newMacros')

  def _hmm_init_and_isolated_word_train(self, token):
    # Create flat start model based on the template, Init means & covariances
    cmd = f'{self.HTK_PATH}/HCompV'
    param_list = [
      '-S', self.sample_list_path,
      '-v', self.HCompV_min_var,
      '-l', token,
      '-I', self.label_mlf_path,
      '-o', token,
      '-M', self.hmm0_path,
      f'{self.init_hmm_path}/{token}'
    ]
    self._run_subprocess_helper(cmd, param_list)

    # Update transitions
    cmd = f'{self.HTK_PATH}/HInit'
    param_list = [
      '-m', int(self.HInit_min_samp),
      '-v', self.HInit_min_var,
      '-M', self.hmm1_path,
      '-l', token,
      '-S', self.sample_list_path,
      '-I', self.label_mlf_path,
      '-o', token,
      f'{self.hmm0_path}/{token}'
    ]
    self._run_subprocess_helper(cmd, param_list)

    # Baum-Welch re-estimation
    cmd = f'{self.HTK_PATH}/HRest'
    param_list = [
      '-m', int(self.init_HRest_min_samp),
      '-i', int(self.init_HRest_num_iter),
      '-v', self.init_HRest_min_var,
      '-l', token,
      '-M', self.hmm2_path,
      '-S', self.sample_list_path,
      '-I', self.label_mlf_path,
      f'{self.hmm1_path}/{token}'
    ]
    self._run_subprocess_helper(cmd, param_list)

  def _run_subprocess_helper(self, command, input_params):
    input_params = [p if isinstance(p, str) else str(p) for p in input_params]
    param_list = [command]
    if self.PRINT_CMD:
      param_list.append('-A')
    if self.TRACE_LEVEL is not None:
      param_list.append('-T')
      param_list.append(str(self.TRACE_LEVEL))
    result = subprocess.run(param_list + input_params, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if not self.SUPRESS_ALL_SUBPROCESS_OUTPUT:
      if len(result.stdout) > 0:
        print(str(result.stdout, 'utf-8').lstrip().rstrip())
    if result.stderr:
      print(str(result.stderr, 'utf-8'))
      raise subprocess.CalledProcessError(returncode=result.returncode, cmd=result.args, stderr=result.stderr)
    return result

  # def _export_to_ext_wrapper(self, args):
  #   return self._export_to_ext(*args)

  def _export_to_ext(self, data, data_path, sample_length_ms):
    # Reference: https://labrosa.ee.columbia.edu/doc/HTKBook21/node58.html
    # TODO: Would having a read parser for this be useful?
    num_sample, sample_vec_size = data.shape
    sample_data_period_ms = sample_length_ms / num_sample
    # print(f'data num_sample: {num_sample} sample_vec_size: {sample_vec_size} sample_data_period_ms: {sample_data_period_ms}')
    with open(f'{data_path}.ext', 'wb') as fp:
      # print(f'{num_sample} {int(sample_data_period_ms * 10)} {int(sample_vec_size * 4)} {9}')
      fp.write(pack('>IIHH', num_sample, int(sample_data_period_ms * 10), int(sample_vec_size * 4), 9))
      fp.write(pack(f'>{"f" * (data.size)}', *(data.reshape(data.size))))

  def _gen_htk_hmm_obj(self, num_states=8, vector_size=1, skip=0, num_mixtures=1, use_full_cov=False):
    '''
    TODO: I am not satisfied with this implementation. Would be
    nice to be able to define tied space and more complicated DAG
    structures. Like variable skip for each state?
    '''
    custom_hmm_struct = {
      'VecSize': vector_size,
      'States': [],
      '_skip': skip
    }
    for state_i in range(num_states - 2):
      custom_hmm_struct['States'].append([])
      for mix_i in range(num_mixtures):
        mixture = {
          'Weight': 1 / num_mixtures,
          'Mean': [0] * vector_size
        }
        if use_full_cov:
          mixture['InvCovar'] = np.diag([1] * vector_size).tolist()
        else:
          mixture['Variance'] = [1] * vector_size
        custom_hmm_struct['States'][-1].append(mixture)
    return custom_hmm_struct

  def _hmm_def_2_hmm_str(self, hmm_def):
    '''
    TODO: Expand this hmmdef generator to support more htk features
    TODO: Make a HTK HMM definition string to custom_hmm_def_obj parser
    '''
    VecSize = hmm_def["VecSize"]
    NumStates = len(hmm_def["States"]) + 2
    hmm_str = f'~o\n<STREAMINFO> 1 {VecSize}\n<VECSIZE> {VecSize} <NULLD><USER>\n'
    hmm_str = f'{hmm_str}<BeginHMM>\n<NumStates> {NumStates}\n'
    for state_i, state in enumerate(hmm_def['States'], 2):
      hmm_str = f'{hmm_str}<State> {state_i} <NumMixes> {len(state)}\n'
      for mix_i, mix in enumerate(state, 1):
        if 'weight' not in mix:
          mix['weight'] = 1 / mix_i
        hmm_str = f'{hmm_str}\t<Mixture> {mix_i} {mix["Weight"]:0.6e}\n'
        hmm_str = f'{hmm_str}\t<Mean> {VecSize}\n'
        if VecSize != len(mix['Mean']):
          raise ValueError(
            f'length of mean vector ({len(mix["Mean"])}) of state {state_i} mixer {mix_i} does not match definition VecSize: {VecSize}'
          )
        hmm_str = f'{hmm_str}\t{" ".join([f"{n:f}" for n in mix["Mean"]])}\n'
        if ('InvCovar' not in mix) and ('Variance' not in mix):
          if self.use_full_cov:
            mix['InvCovar'] = np.diag([1] * VecSize).tolist()
          else:
            mix['Variance'] = [1] * VecSize
        if 'InvCovar' in mix:
          hmm_str = f'{hmm_str}\t<InvCovar> {VecSize}\n'
          for invcovar_i, invcovar_r in enumerate(mix['InvCovar']):
            hmm_str = f'{hmm_str}\t{" ".join([f"{n:f}" for n in invcovar_r[invcovar_i:]])}\n'
        elif 'Variance' in mix:
          hmm_str = f'{hmm_str}\t<Variance> {VecSize}\n'
          hmm_str = f'{hmm_str}\t{" ".join([f"{n:f}" for n in mix["Variance"]])}\n'
        if 'GCONST' in mix:
          hmm_str = f'{hmm_str}\t<GCONST> {mix["GCONST"]}\n'
    hmm_str = f'{hmm_str}<TransP> {NumStates}\n'
    transp_row = [0.0] * NumStates
    transp_row[1] = 1
    hmm_str = f'{hmm_str}\t{" ".join([f"{n:f}" for n in transp_row])}\n'
    skip = hmm_def['_skip']
    for i in range(1, NumStates - skip - 1):
        hmm_str = f'{hmm_str}\t{" ".join([f"{n:f}" for n in [0.0] * i + [1 / (skip + 2)] * (skip + 2) + [0.0] * (NumStates - i - skip - 2)])}\n'
    for j in range(1, skip + 1):
        hmm_str = f'{hmm_str}\t{" ".join([f"{n:f}" for n in [0.0] * (i + j) + [1 / (skip + 2 - j)] * (skip + 2 - j)])}\n'
    hmm_str = f'{hmm_str}\t{" ".join([f"{n:f}" for n in [0.0] * NumStates])}\n'
    hmm_str = f'{hmm_str}<EndHMM>\n\n'
    return hmm_str