from .helpers import *
from .glm import *
from .HTK_Hmm import *
import numpy as np
from scipy.signal import butter, sosfilt
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from joblib import Parallel, delayed
import copy
import dlib

class HTKHMMDecoder():
  def __init__(self, dict_string, grammar_string, add_monophone_back=False, insertion_penalty=0):
    self.dict_string = dict_string
    self.grammar_string = grammar_string
    self.insertion_penalty = insertion_penalty
    self.add_monophone_back = add_monophone_back


  def fit(self, X, y, insertion_penalty = None):
    if insertion_penalty is not None:
      self.insertion_penalty = insertion_penalty
    y = [[l if l != ' ' else '_space_' for l in y_i] for y_i in y]
    self.clf = HTK_Hmm(
                num_states = 4,
                bi_tri_phone_edcmd = 'WB _space_\nTC',
                skip = 0,
                use_tied_states = True,
                add_monophone_back = self.add_monophone_back,
                dict_string = self.dict_string,
                grammar_string = self.grammar_string,
                init_HRest_min_var = 0.001,
                HCompV_min_var = 0.001,
                bi_tri_phone_HERest_min_var = 0.001,
                embedded_training_HERest_min_var = 0.001,
                HInit_min_var = 0.001,
                bi_tri_phone_tied_HERest_min_var = 0.001,
                TB_threshold = 0,
                convert_to_tied_state_threshold = 0,
                num_cpu = 1,
                SUPRESS_ALL_SUBPROCESS_OUTPUT = True
              )
    self.clf.fit(X, y)
    return self

  def predict(self, X, insertion_penalty = None, token_label=True):
    if insertion_penalty is not None:
      self.insertion_penalty = insertion_penalty
    pred = self.clf.predict(X, token_label=token_label, insertion_penalty=self.insertion_penalty)
    if token_label:
      return [[l if l != '_space_' else ' ' for l in pred_i] for pred_i in pred]

class ButterworthBandpassFilter():
  def __init__(self, lowcut, highcut, fs, order=4, axis=0):
    self.sos = butterworth_bandpass(lowcut, highcut, fs, order)
    self.axis = axis

  def fit(self, X, y=None):
    return self

  def transform(self, X, axis=None):
    axis_to_use = self.axis if axis is None else axis
    return np.array([sosfilt(self.sos, x_i, axis_to_use) for x_i in X])

  def fit_transform(self, X, y=None, axis=None):
    self.fit(X, y)
    return self.transform(X, axis)

class DataTrimmer():
  def __init__(self, num_delay_frame, num_frame_per_label):
    self.num_delay_frame = num_delay_frame
    self.num_frame_per_label = num_frame_per_label
    self.num_frame_to_trim_at_end = 0

  def fit(self, X, y = None):
    # print([len(x_i) for x_i in X])
    # print([len(y_i) * self.num_frame_per_label for y_i in y])
    self.num_frame_to_trim_at_end = int(np.mean([len(x_i) - (len(y_i) * self.num_frame_per_label) - self.num_delay_frame for x_i, y_i in zip(X, y)]))
    return self

  def transform(self, X):
    return [x_i[self.num_delay_frame: len(x_i)-self.num_frame_to_trim_at_end, :] for x_i in X]

  def fit_transform(self, X, y = None):
    self.fit(X, y)
    return self.transform(X)

class ZNormalizeBySub():
  def __init__(self, x_subjects, y_subjects):
    self.x_subjects = np.array(x_subjects)
    self.y_subjects = np.array(y_subjects)
    self.z_norm_params_by_sub = {}

  def fit(self, X, y=None):
    X = np.array(X)
    self.X = X.copy()
    return self

  def transform(self, X):
    unique_train = np.unique(self.x_subjects)
    for sub_i in unique_train:
      sub_mask = self.x_subjects == sub_i
      X_sub = self.X[sub_mask, :]
      X_sub_mean = X_sub.mean(axis=(0,1))
      X_sub_std  = X_sub.std(axis =(0,1))
      self.z_norm_params_by_sub[sub_i] = {'mean': X_sub_mean, 'std':X_sub_std}
    X = np.array(X)
    if np.allclose(X.shape, self.X.shape):
      if np.allclose(X, self.X):
        unique_x_subjects = np.unique(self.x_subjects)
        for sub_i in unique_x_subjects:
          sub_mask = self.x_subjects == sub_i
          X_sub = X[sub_mask, :]
          X[sub_mask, :] = (X[sub_mask, :] - self.z_norm_params_by_sub[sub_i]['mean']) / self.z_norm_params_by_sub[sub_i]['std']
    else:
      unique_y_subjects = np.unique(self.y_subjects)
      for sub_i in unique_y_subjects:
        sub_mask = self.y_subjects == sub_i
        X_sub = X[sub_mask, :]
        X[sub_mask, :] = (X[sub_mask, :] - self.z_norm_params_by_sub[sub_i]['mean']) / self.z_norm_params_by_sub[sub_i]['std']
    return X

def parseLatticeString(latticeString):
  lines = latticeString.split('\n')
  lattice_size_info = lines[1].split()
  num_nodes, num_link = int(lattice_size_info[0].split('=')[1]), int(lattice_size_info[1].split('=')[1])
  nodes_lines = lines[2: 2 + num_nodes]
  node_symbols = [line.split()[1].split('=')[1] for line in nodes_lines]
  link_lines  = lines[2 + num_nodes: 2 + num_nodes + num_link]
  link_id_start_end = [line.split() for line in link_lines]
  link_start_end = [(int(line[1].split('=')[1]), int(line[2].split('=')[1])) for line in link_id_start_end]
  return node_symbols, link_start_end

def get_word_lattice_from_grammar(htk_grammar_string, HTK_PATH = None):
  if HTK_PATH is None:
    HTK_PATH = os.environ.get('HTK_PATH')
  cmd = f'{HTK_PATH}/HParse'
  grammar_path = './grammar_string'
  word_lattice_path = './lattice_string'
  write_file(htk_grammar_string, grammar_path)
  params = [grammar_path ,word_lattice_path]
  result = subprocess.run([cmd] + params, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  word_lattice_string = load_file(word_lattice_path)
  delete_file_if_exists(grammar_path)
  delete_file_if_exists(word_lattice_path)
  return word_lattice_string

class InsertionPenaltyTunedHTKDecoder():
  def __init__(self, decoder, insertion_penalty_range, random_state, n_splits = None, n_calls=16):
    self.decoder = decoder
    self.insertion_penalty_range = insertion_penalty_range
    self.random_state = random_state
    self.n_splits = n_splits
    self.n_calls = n_calls
    self.best_insertion_penalty = 0

  def fit(self, X, y):
    previous_n_splits = self.n_splits
    if self.n_splits is None:
      self.n_splits = len(X)
    kf = KFold(n_splits=self.n_splits, random_state=self.random_state, shuffle=True)
    cv_decoders = [copy.deepcopy(self.decoder) for i in range(self.n_splits)]
    x_test_all = []
    y_test_all = []
    x_train_all = []
    y_train_all = []
    for i, train_test_i in enumerate(kf.split(X)):
      train_i, test_i = train_test_i
      x_train_i = [X[i] for i in train_i]
      y_train_i = [y[i] for i in train_i]
      x_test_i = [X[i] for i in test_i]
      y_test_i = [y[i] for i in test_i]
      x_test_all.append(x_test_i)
      y_test_all.append(y_test_i)
      x_train_all.append(x_train_i)
      y_train_all.append(y_train_i)
      decoder_i = cv_decoders[i]
      for step_name, step_obj in decoder_i.steps:
        if step_name == 'ZNormalizeBySub':
          step_obj.y_subjects = step_obj.x_subjects[test_i]
          step_obj.x_subjects = step_obj.x_subjects[train_i]
          step_obj.fit(x_train_i)

    def train(decoder, x, y):
      return decoder.fit(x, y)

    def test(decoder, x, y, insertion_penalty):
      res = decoder.predict(x, insertion_penalty = insertion_penalty)
      if len(res[0]) != len(y[0]):
        return 0
      return accuracy_score([l for run in y for l in run], [l for run in res for l in run])

    trained_cv_decoders = Parallel(n_jobs=-1)(delayed(train) (decoder, x, y) for decoder, x, y in zip(cv_decoders, x_train_all, y_train_all))

    def cost(param):
      acc = np.mean(Parallel(n_jobs=-1)(delayed(test)(decoder, x, y, param) for decoder, x, y in zip(trained_cv_decoders, x_test_all, y_test_all)))
      # print(f'insertion panelty: {param:.4f} acc: {acc:.4f}')
      return -acc
    # trained_cv_decoders = Parallel(n_jobs=-1)(delayed(train) (decoder, x, y) for decoder, x, y in zip(cv_decoders, x_train_all, y_train_all))
    res = dlib.find_min_global(cost, [self.insertion_penalty_range[0]], [self.insertion_penalty_range[1]], self.n_calls)
    self.best_insertion_penalty = res[0][0]
    self.decoder = self.decoder.fit(X, y)
    self.n_splits = previous_n_splits
    return self

  def predict(self, X):
    return self.decoder.predict(X, insertion_penalty = self.best_insertion_penalty)

def get_srilm_ngram(content, n = 2, SRILM_PATH = None, **kwargs):
  if SRILM_PATH is None:
    SRILM_PATH = os.environ.get('SRILM_PATH')
  cmd = f'{SRILM_PATH}/ngram-count'
  content_path = './content.txt'
  write_file(content, content_path)
  ngram_out_path = f'./{n}gram.lm'
  params = ['-text', content_path, '-order', str(n), '-lm', ngram_out_path] + [f'{key_value[i].replace("_", "-")}' if i==0 else str(key_value[i])  for key_value in kwargs.items() for i in range(2)]
  result = subprocess.run([cmd] + params, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  ngram_content = load_file(ngram_out_path)
  delete_file_if_exists(content_path)
  delete_file_if_exists(ngram_out_path)
  return ngram_content

def log_double_dict(dict_to_log):
  for key in dict_to_log:
    for sub_key in dict_to_log[key]:
      dict_to_log[key][sub_key] = np.log10(dict_to_log[key][sub_key])
  return dict_to_log


def get_ngram_prob_dict(content_string, n, space_tok = '_space_', use_log_prob=True):
  content_string_bigram_string = get_srilm_ngram(content_string, n=n, _no_sos='', _no_eos='', _sort='')
  content_string_bigram_string = content_string_bigram_string.split('\n\n\\end\\')[0]
  log_prob_list = []
  for i in range(n, 0, -1):
    splited_string = content_string_bigram_string.split(f'\\{i}-grams:\n')
    content_string_bigram_string = splited_string[0]
    i_gram_string = splited_string[1].rstrip()
    i_gram_string_lines = [line.split() for line in i_gram_string.split('\n')]
    i_gram_string_lines = [[item if item != space_tok else ' ' for item in line[0:i+1]] for line in i_gram_string_lines]
    log_prob_dict = {}

    probs = np.array([float(line[0]) for line in i_gram_string_lines])
    if not use_log_prob:
      probs = np.power(10, probs)

    for line_i, items in enumerate(i_gram_string_lines):
      if ('<s>' in items) or ('</s>' in items):
        continue
      second_level_key = items[-1]
      if i > 1:
        key = ''.join(items[1: -1])
        if key not in log_prob_dict:
          log_prob_dict[key] = {}
        log_prob_dict[key][second_level_key] = probs[line_i]
      else:
        log_prob_dict[second_level_key] = probs[line_i]
    log_prob_list.append(log_prob_dict)
  return log_prob_list

def get_srilm_ngram(content, n = 2, SRILM_PATH = None, **kwargs):
  if SRILM_PATH is None:
    SRILM_PATH = os.environ.get('SRILM_PATH')
  cmd = f'{SRILM_PATH}/ngram-count'
  content_path = './content.txt'
  write_file(content, content_path)
  ngram_out_path = f'./{n}gram.lm'
  params = ['-text', content_path, '-order', str(n), '-lm', ngram_out_path] + [f'{key_value[i].replace("_", "-")}' if i==0 else str(key_value[i])  for key_value in kwargs.items() for i in range(2)]
  result = subprocess.run([cmd] + params, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  ngram_content = load_file(ngram_out_path)
  delete_file_if_exists(content_path)
  delete_file_if_exists(ngram_out_path)
  return ngram_content

class DataSlice():
  def __init__(self, extra_frame, delay_frame, event_len_frame, event_interval_frame, feature_mask = None):
    self.extra_frame = extra_frame
    self.delay_frame = delay_frame
    self.event_len_frame = event_len_frame
    self.event_interval_frame = event_interval_frame
    self.num_event = 0
    self.feature_mask = feature_mask
    pass

  def fit(self, X, y=None):
    self.num_event = len(y[0])
    return self

  def transform(self, X):
    slice_indice_start = np.arange(0, self.event_len_frame * (self.num_event - 1), self.event_len_frame) + self.delay_frame
    slice_indice_end = slice_indice_start + self.event_len_frame * 2 + self.extra_frame
    sliced_x = np.array([[x_i[start_i:end_i, self.feature_mask if (self.feature_mask is not None) else np.arange(x_i.shape[1])] for start_i, end_i in zip(slice_indice_start, slice_indice_end)] for x_i in X])
    # print(f'transformed x shape: {np.array(sliced_x).shape}')
    return sliced_x

def log_double_dict(dict_to_log):
  for key in dict_to_log:
    for sub_key in dict_to_log[key]:
      dict_to_log[key][sub_key] = np.log10(dict_to_log[key][sub_key])
  return dict_to_log

def key_of_max_val_in_dict(dict_input):
  return max(dict_input, key=dict_input.get)

def add_bigram_probabilities(letter_probs, bigram_prob_dict):
  letter_list = letter_probs[0].keys()
  all_prob_vals = np.array([val for dict_i in bigram_prob_dict.values() for val in dict_i.values()])
  default_prob = np.min(all_prob_vals[all_prob_vals > 0])
  bigram_correction_matrix = np.array([[bigram_prob_dict[prev_l][l] if l in bigram_prob_dict[prev_l] else default_prob for l in letter_list] for prev_l in letter_list ])
  letter_probs_table = np.array([[l_prob[l] for l in letter_list] for l_prob in letter_probs])
  corrected_probs = np.zeros(letter_probs_table.shape)
  corrected_probs[0] = letter_probs_table[0]
  for i in range(1, len(letter_probs)):
    prev_prob, curr_prob = letter_probs_table[i-1], letter_probs_table[i]
    correction = (prev_prob.reshape(prev_prob.size ,1) * bigram_correction_matrix).sum(axis=0)
    corrected_probs[i] = correction * curr_prob
  return [{l:prob for l, prob in zip(letter_list, corrected_prob)} for corrected_prob in corrected_probs]

def letter_level_bigram_viterbi_decode(letter_probs, bigram_log_prob_dict):
  # log(A*B) = log(A) + log(B);
  # log(A+B) = logsumexp(log(A) + log(B))
  # Check log(1+x) taylor series?
  # Check how to use log(A) and log(B) to get log(A+B)?
  letter_list = list(letter_probs[0].keys())
  all_prob_vals = np.array([val for dict_i in bigram_log_prob_dict.values() for val in dict_i.values()])
  default_prob = np.min(all_prob_vals[all_prob_vals < 0])
  # bigram_log_prob_dict = {l: defaultdict(lambda: default_prob, bigram_log_prob_dict[l]) for l in letter_list}
  # transition_log_prob_matrix = np.array([[bigram_log_prob_dict[prev_l][l] for l in letter_list] for prev_l in letter_list ])
  transition_log_prob_matrix = np.array([[bigram_log_prob_dict[prev_l][l] if l in bigram_log_prob_dict[prev_l] else default_prob for l in letter_list] for prev_l in letter_list ])
  log_prob_table = np.log10( np.array([[l_prob[l] for l in letter_list] for l_prob in letter_probs]) ).T
  prev_table = np.zeros((log_prob_table.shape[0], log_prob_table.shape[1]-1), dtype=np.int32)
  emission_log_prob_table = np.zeros(log_prob_table.shape)
  emission_log_prob_table[:, 0] = log_prob_table[:, 0]
  letter_list_array = np.ones((len(letter_list), 1))
  for i in range(1, emission_log_prob_table.shape[1]):
    prev_node_probs = (emission_log_prob_table[:, i-1] * letter_list_array).T
    transition_correction = prev_node_probs + transition_log_prob_matrix
    best = np.argmax(transition_correction, axis=0)
    prev_table[:, i - 1] = best
    best_transition = np.take_along_axis(transition_correction, np.expand_dims(best, axis=0), axis=0)
    emission_log_prob_table[:, i] = best_transition + log_prob_table[:, i]

  last_letter_log_prob = emission_log_prob_table[:, -1]
  last_letter_index = np.argmax(last_letter_log_prob)
  letter_index_array = np.zeros(emission_log_prob_table.shape[1], dtype=np.int32)
  letter_index_array[-1] = last_letter_index
  for i in range(prev_table.shape[1] - 1, -1, -1):
    letter_index_array[i] = prev_table[letter_index_array[i+1]][i]
  letters = [letter_list[l_i] for l_i in letter_index_array]
  return letters

def letter_bigram_viterbi_with_grammar_decode(letter_probs, bigram_log_prob_dict, grammar_node_symbols, grammar_link_start_end, dictionary, insert_panelty = 0, max_null_resolve_count = 3):
  letter_list, num_entry = list(letter_probs[0].keys()), len(letter_probs)
  letter_to_ind = {l:i for i, l in enumerate(letter_list)}
  number_of_symbols = len(grammar_node_symbols)
  node_symbol_index_arr = np.arange(number_of_symbols)
  symbols_eye_matrix = np.eye(number_of_symbols, dtype=bool)
  node_symbol_to_ind = {symbol : i for i, symbol in enumerate(grammar_node_symbols)}
  node_letters_spelling = [dictionary[symbol] for symbol in grammar_node_symbols]
  grammar_node_symbols = np.array(grammar_node_symbols)
  node_letters_ind_spelling = np.array([np.array([letter_to_ind[l] for l in node_i], dtype=np.int32) for node_i in node_letters_spelling], dtype=object)
  node_letters_ind_first_letter = np.array([letter_to_ind[node[0]] if len(node) > 0 else len(letter_list) for node in node_letters_spelling], dtype=np.int32)
  node_letters_ind_last_letter = np.array([letter_to_ind[node[-1]] if len(node) > 0 else len(letter_list) for node in node_letters_spelling], dtype=np.int32)
  node_letters_len = np.array([len(spell) for spell in node_letters_spelling])
  non_end_null_node_mask = node_letters_len == 0
  # print(node_letters_spelling)

  dummy_start_node_ind = number_of_symbols - 1
  end_node_index = number_of_symbols - 2
  non_end_null_node_mask[end_node_index] = False
  # print(end_node_index)

  node_index_range = np.arange(number_of_symbols, dtype=np.int32)
  start_nodes, end_nodes = zip(*grammar_link_start_end)
  # Initialize the tables for the tables
  emission_word_log_prob_table = np.ones((number_of_symbols, num_entry)) * -np.inf
  # Initialize to invalid states to expose error in debugging
  prev_word_table = np.ones((number_of_symbols, num_entry), dtype=np.int32) * number_of_symbols
  # Table that records and updates all rechability of an entry by the symbol

  all_prob_vals = np.array([val for dict_i in bigram_log_prob_dict.values() for val in dict_i.values()])
  default_prob = np.min(all_prob_vals[all_prob_vals < 0])
  transition_log_prob_matrix = np.array([[bigram_log_prob_dict[prev_l][l] if l in bigram_log_prob_dict[prev_l] else default_prob for l in letter_list] for prev_l in letter_list ])
  log_prob_table = np.log10( np.array([[l_prob[l] for l in letter_list] for l_prob in letter_probs]) ).T
  letter_list_array = np.ones((len(letter_list), 1))

  grammar_node_transition_table = np.zeros((number_of_symbols, number_of_symbols), dtype=bool)
  # grammar_node_reverse_transition_table = np.zeros((number_of_symbols, number_of_symbols), dtype=bool)

  node_letters_ind_spelling_transition_modifier = np.array([0 if len(spelling) <= 1 else np.sum([transition_log_prob_matrix[prev][curr] for prev, curr in zip(spelling[0:-1], spelling[1:])]) for  spelling in node_letters_ind_spelling])
  # node_letters_ind_spelling_transition_modifier = np.zeros(node_letters_ind_spelling_transition_modifier.shape)

  for link in grammar_link_start_end:
    grammar_node_transition_table[link[0]][link[1]] = True
    # grammar_node_reverse_transition_table[link[1]][link[0]] = True

  def resolve_null_helper(curr_nodes_mask):
    next_nodes_null_mask = np.logical_and(curr_nodes_mask, non_end_null_node_mask)
    has_null = np.any(next_nodes_null_mask)
    if has_null:
      null_ind = np.arange(len(next_nodes_null_mask))[next_nodes_null_mask] # node_symbol_index_arr[next_nodes_null_mask]
      curr_nodes_mask = np.logical_xor(curr_nodes_mask, next_nodes_null_mask)
      null_next_nodes = (grammar_node_transition_table[null_ind][:]).any(axis=0)
      curr_nodes_mask = np.logical_or(curr_nodes_mask, null_next_nodes)
    return has_null, curr_nodes_mask

  def resolve_null(curr_nodes_mask, max_null_resolve_count = 3):
    next_nodes_null_mask = np.logical_and(curr_nodes_mask, non_end_null_node_mask)
    has_null = np.any(next_nodes_null_mask)
    num_resolve_counter = max_null_resolve_count
    while (has_null and (num_resolve_counter > 0)):
      has_null, curr_nodes_mask = resolve_null_helper(curr_nodes_mask)
      num_resolve_counter -= 1
    if num_resolve_counter == 0:
      raise Exception('Invalid grammar!', f'Consecutive !Null node chain with length larger than max_null_resolve_count: {max_null_resolve_count}')
    return curr_nodes_mask

  # resolve_null_cache = np.array([resolve_null(symbols_eye_matrix[i], max_null_resolve_count) for i in range(number_of_symbols)])
  resolve_next_null_cahce = np.array([resolve_null(grammar_node_transition_table[i], max_null_resolve_count) for i in range(number_of_symbols)])
  next_node_indices = node_symbol_index_arr[resolve_next_null_cahce[dummy_start_node_ind]]

  curr_node_index = -1
  curr_node_end_i = -1
  bad_start_counter = 0
  for next_node_index in next_node_indices:
    next_node_len = node_letters_len[next_node_index]
    if (next_node_len + curr_node_end_i) >= num_entry:
      bad_start_counter += 1
      continue
    next_nodes_spelling = node_letters_ind_spelling[next_node_index]
    next_nodes_spelling_transition_modifier = node_letters_ind_spelling_transition_modifier[next_node_index]

    next_nodes_spelling_prob = np.sum([log_prob_table[letter][curr_node_end_i + 1 + letter_i] for letter_i, letter in enumerate(next_nodes_spelling)])
    next_nodes_emission_prob = next_nodes_spelling_prob + next_nodes_spelling_transition_modifier
    if next_nodes_emission_prob > emission_word_log_prob_table[next_node_index, curr_node_end_i + next_node_len]:
      emission_word_log_prob_table[next_node_index, curr_node_end_i + next_node_len] = next_nodes_emission_prob
      prev_word_table[next_node_index, curr_node_end_i + next_node_len] = curr_node_index

  if bad_start_counter == len(next_node_indices):
    raise Exception('Invalid grammar!' 'Start node spelling length is longer than the entire sequence')

  for curr_node_end_i in range(0, num_entry):
    curr_node_mask = emission_word_log_prob_table[:, curr_node_end_i] > -np.inf
    if not np.any(curr_node_mask):
      continue
    curr_node_indice = node_symbol_index_arr[curr_node_mask]
    all_nodes_end_i = node_letters_len + curr_node_end_i
    all_nodes_end_i_no_overflow = all_nodes_end_i < num_entry
    if not np.any(all_nodes_end_i_no_overflow):
      continue

    for curr_node_index in curr_node_indice:
      cur_log_prob = emission_word_log_prob_table[curr_node_index][curr_node_end_i]
      next_nodes_mask = resolve_next_null_cahce[curr_node_index]
      next_nodes_mask = np.logical_and(next_nodes_mask, all_nodes_end_i_no_overflow)
      has_end = next_nodes_mask[end_node_index]
      next_nodes_mask[end_node_index] = False

      if np.any(next_nodes_mask):
        next_nodes_end_i = all_nodes_end_i[next_nodes_mask]
        next_nodes_spelling = node_letters_ind_spelling[next_nodes_mask]
        next_nodes_spelling_transition_modifier = node_letters_ind_spelling_transition_modifier[next_nodes_mask]
        next_nodes_spelling_probs = np.array([np.sum([log_prob_table[letter, curr_node_end_i + 1 + letter_i] for letter_i, letter in enumerate(node)]) for node in next_nodes_spelling])
        next_nodes_transition_log_prob_from_current_node = transition_log_prob_matrix[node_letters_ind_last_letter[curr_node_index], node_letters_ind_first_letter[next_nodes_mask]]
        next_nodes_new_emission_log_probs = insert_panelty + cur_log_prob + next_nodes_transition_log_prob_from_current_node + next_nodes_spelling_transition_modifier + next_nodes_spelling_probs
        next_nodes_emission_log_probs = emission_word_log_prob_table[next_nodes_mask, next_nodes_end_i]
        new_probs_better_mask = next_nodes_new_emission_log_probs > next_nodes_emission_log_probs
        next_nodes_emission_log_probs[new_probs_better_mask] = next_nodes_new_emission_log_probs[new_probs_better_mask]
        emission_word_log_prob_table[next_nodes_mask, next_nodes_end_i] = next_nodes_emission_log_probs
        next_nodes_prev_word =  prev_word_table[next_nodes_mask, next_nodes_end_i]
        next_nodes_prev_word[new_probs_better_mask] = curr_node_index
        prev_word_table[next_nodes_mask, next_nodes_end_i] = next_nodes_prev_word

      if has_end:
        if cur_log_prob > emission_word_log_prob_table[end_node_index, curr_node_end_i]:
          emission_word_log_prob_table[end_node_index, curr_node_end_i] = cur_log_prob
          prev_word_table[end_node_index, curr_node_end_i] = curr_node_index

  curr_node_entry_index = num_entry - 1
  countdown = curr_node_entry_index
  curr_node_index = prev_word_table[end_node_index, curr_node_entry_index]
  reverse_nodes = []
  while (curr_node_index != -1) and (countdown > 0):
    node_len = node_letters_len[curr_node_index]
    reverse_nodes.append(curr_node_index)
    curr_node_index = prev_word_table[curr_node_index, curr_node_entry_index]
    curr_node_entry_index -= node_len
    countdown -= 1
  letters = [j for i in reversed(reverse_nodes) for j in node_letters_spelling[i]]
  return letters

def clf_pred_proba(clf, X):
  return clf.predict_proba(X)

def letter_label_to_transition_label_by_type(y, LETTERS_TO_DOT, region_order):
    dot_label = [[[LETTERS_TO_DOT[l_i][region] for region in region_order] for l_i in run_i] for run_i in y]
    transition_label_by_type = [{r:  [ prev[i] * 2 + curr[i] for prev, curr in zip(run_i[0:-1], run_i[1:])] for i, r in enumerate(region_order)} for run_i in dot_label]
    return transition_label_by_type

def letter_label_to_transition_label(y, LETTERS_TO_DOT, region_order):
    dot_label = [[[LETTERS_TO_DOT[l_i][region] for region in region_order] for l_i in run_i] for run_i in y]
    transition_label = [[[(prev[i] * 2 + curr[i]) for i in range(len(region_order))] for prev, curr in zip(run_i[0:-1], run_i[1:])] for run_i in dot_label]
    return transition_label

class SVMProbDecoder():
  def __init__(self, LETTERS_TO_DOT, region_order, bigram_dict = None, words_node_symbols = None, words_link_start_end = None, words_dictionary = None, SVM_params=None, insertion_penalty=0.0):
    self.LETTERS_TO_DOT = LETTERS_TO_DOT
    self.region_order = region_order
    self.DOT_TO_LETTERS = { ''.join([str(val[r]) for r in self.region_order]): key for key, val in self.LETTERS_TO_DOT.items()}
    self.SVM_params = [SVM_params] * len(region_order)
    self.clfs = [None] * len(region_order)
    self.add_bigram_dict(bigram_dict)
    self.words_node_symbols = words_node_symbols
    self.words_link_start_end = words_link_start_end
    self.words_dictionary = words_dictionary
    self.insertion_penalty = insertion_penalty
    self.trans_prob_by_type = None
    self.trans_class = None
    self.trans_class_by_type = None
    self.tuple_label_by_type = None
    self.label_1 = None
    self.label_2 = None
    self.string_label_1 = None
    self.string_label_2 = None
    self.direct_decode_letter_label_1 = None
    self.direct_decode_letter_label_2 = None
    self.state_prob_by_type_1 = None
    self.state_prob_by_type_2 = None
    self.state_prob_by_type = None
    self.naive_letter_prob = None
    self.naive_prob_letter_label = None
    self.bigram_weighted_prob = None
    self.bigram_weighted_letter_label = None
    self.letter_viterbi_decode_letter_label = None
    self.X_cache = None

  def add_bigram_dict(self, bigram_dict):
    if bigram_dict is not None:
      self.bigram_dict = bigram_dict
      self.bigram_log_dict = log_double_dict(copy.deepcopy(self.bigram_dict))
    else:
      self.bigram_dict = None
      self.bigram_log_dict = None

  def fit(self, X, y=None, r_i=None):
    # transition_label = self.letter_label_to_transition_label(y)
    transition_label = letter_label_to_transition_label(y, self.LETTERS_TO_DOT, self.region_order)
    transition_label = np.array([entry for run in transition_label for entry in run])
    X = np.array([entry_i for run_i in X for entry_i in run_i])
    num_entry, num_timeframe, num_region = X.shape
    X_expanded = X.reshape((num_entry, num_timeframe * num_region))

    def fit_svm_for_one_region(X, y_label, SVM_params):
      clf = SVC(kernel='rbf', probability=True, break_ties=True)
      if SVM_params is not None:
        clf.set_params(**SVM_params)
      clf.fit(X, y_label)
      return clf

    if r_i is None:
      self.clfs = Parallel(n_jobs=-1)(delayed(fit_svm_for_one_region)(X_expanded, transition_label[:, i], self.SVM_params[i]) for i in range(len(self.region_order)))
      # self.clfs = [fit_svm_for_one_region(X_expanded, train_transition_label[:, i], SVM_params) for i in range(len(self.region_order))]
    else:
      self.clfs[r_i] = fit_svm_for_one_region(X_expanded, transition_label[:, r_i], self.SVM_params[r_i])
    return self

  def predict_svm_proba(self, X, r_i = None):
    X = np.array(X)
    X_each_run_len = [len(x_i) for x_i in X]
    X_each_run_start_end = [(end - X_each_run_len[j] , end) for j, end in enumerate([np.sum(X_each_run_len[: (i + 1)]) for i in range(len(X_each_run_len))])]
    X = np.array([e_i for x_i in X for e_i in x_i])
    num_entry, num_timeframe, num_region = X.shape
    X_expanded = X.reshape((num_entry, num_timeframe * num_region))
    if r_i is None:
      prob_flatten = np.array(Parallel(n_jobs=-1)(delayed(clf_pred_proba)(clf_i, X_expanded) for clf_i in self.clfs))
      prob = [prob_flatten[:, start:end] for start, end in X_each_run_start_end]
    else:
      prob_flatten = clf_pred_proba(self.clfs[r_i], X_expanded)
      prob = [prob_flatten[start:end] for start, end in X_each_run_start_end]
    return prob

  def predict_svm_transition(self, X):
    prob = self.predict_svm_proba(X)
    trans_class = [[{r: np.argmax(prob_each_r[i]) for i, r in enumerate(self.region_order)} for prob_each_r in zip(*run_i)] for run_i in prob]
    return trans_class

  def predict(self, X, r_i=None, svm_proba = False, svm_transition=False, bigram_dict = None, words_node_symbols = None, words_link_start_end = None, words_dictionary = None, insertion_penalty = None, token_label=True):
    if svm_proba:
      return self.predict_svm_proba(X, r_i)
    if svm_transition:
      return self.predict_svm_transition(X)
    X = np.array(X)
    if insertion_penalty is None:
      insertion_penalty = self.insertion_penalty
    has_cache = self.X_cache is not None
    if has_cache and np.allclose(X.flatten(), self.X_cache.flatten()):
      use_cache = True
    else:
      use_cache = False
      self.X_cache = X.copy()
    latest_results = None
    if not use_cache:
      X_each_run_len = [len(x_i) for x_i in X]
      X_each_run_start_end = [(end - X_each_run_len[j] , end) for j, end in enumerate([np.sum(X_each_run_len[: (i + 1)]) for i in range(len(X_each_run_len))])]
      X = np.array([e_i for x_i in X for e_i in x_i])
      num_entry, num_timeframe, num_region = X.shape
      X_expanded = X.reshape((num_entry, num_timeframe * num_region))
      prob_flatten = np.array(Parallel(n_jobs=-1)(delayed(clf_pred_proba)(clf_i, X_expanded) for clf_i in self.clfs))
      prob = [prob_flatten[:, start:end] for start, end in X_each_run_start_end]
      self.trans_prob_by_type = [[{r: prob_each_r[i] for i, r in enumerate(self.region_order)} for prob_each_r in zip(*run_i)] for run_i in prob]
      self.trans_class = [[{r: np.argmax(prob_each_r[i]) for i, r in enumerate(self.region_order)} for prob_each_r in zip(*run_i)] for run_i in prob]
      self.trans_class_by_type = [{r: [e_i[r] for e_i in run_i] for r in self.region_order} for run_i in self.trans_class]
      self.tuple_label_by_type = [[({key: val // 2 for key, val in entry_i.items()},{key: val % 2 for key, val in entry_i.items()}) for entry_i in run_i] for run_i in self.trans_class]
      self.label_1 = [[entry[0] for entry in run_i] + [run_i[-1][1]] for run_i in self.tuple_label_by_type]
      self.label_2 = [[run_i[0][0]] + [entry[1] for entry in run_i] for run_i in self.tuple_label_by_type]

      self.string_label_1 = [[''.join([str(label_i_j[r]) for r in self.region_order]) for label_i_j in run_i] for run_i in self.label_1]
      self.string_label_2 = [[''.join([str(label_i_j[r]) for r in self.region_order]) for label_i_j in run_i] for run_i in self.label_2]

      self.direct_decode_letter_label_1 = [[self.DOT_TO_LETTERS.setdefault(label_i_j, '?') for label_i_j in run_i] for run_i in self.string_label_1]
      self.direct_decode_letter_label_2 = [[self.DOT_TO_LETTERS.setdefault(label_i_j, '?') for label_i_j in run_i] for run_i in self.string_label_2]

      # A note on converting transition to state ON probability state transition classification 0, 1, 2, 3 conrespond to 00, 01, 10, 11.
      # The sum of proba of 10, 11 is when the first state in ON, the sum of proba of 01 and 11 is when the second state in ON
      self.state_prob_by_type_1 = np.array([[[entry[r][3] + entry[r][2] for r in self.region_order] for entry in run_i] + [[run_i[-1][r][3] + run_i[-1][r][1] for r in self.region_order]] for run_i in self.trans_prob_by_type])
      self.state_prob_by_type_2 = np.array([[[run_i[-1][r][3] + run_i[-1][r][2] for r in self.region_order]] + [[entry[r][3] + entry[r][1] for r in self.region_order] for entry in run_i] for run_i in self.trans_prob_by_type])

      self.state_prob_by_type = (self.state_prob_by_type_1 + self.state_prob_by_type_2) / 2
      self.naive_letter_prob = [ [ {l: np.prod([val if l2d[r] else 1-val for r_i, r, val in zip(range(len(self.region_order)), self.region_order, e_i)]) for l, l2d in self.LETTERS_TO_DOT.items()} for e_i in run_i] for run_i in self.state_prob_by_type]
      self.naive_prob_letter_label = [[ key_of_max_val_in_dict(e_i) for e_i in run_i] for run_i in self.naive_letter_prob]

    if self.naive_prob_letter_label is not None:
      latest_results = self.naive_prob_letter_label

    if bigram_dict is not None:
      if bigram_dict != self.bigram_dict:
        use_cache = False
        self.add_bigram_dict(bigram_dict)

    if (not use_cache) and (self.bigram_dict is not None):
      self.bigram_weighted_prob = [add_bigram_probabilities(run_i, self.bigram_dict) for run_i in self.naive_letter_prob]
      self.bigram_weighted_letter_label = [[ key_of_max_val_in_dict(e_i) for e_i in run_i] for run_i in self.bigram_weighted_prob]
      self.letter_viterbi_decode_letter_label = [letter_level_bigram_viterbi_decode(run_i, self.bigram_log_dict) for run_i in self.naive_letter_prob]

    if self.letter_viterbi_decode_letter_label is not None:
      latest_results = self.letter_viterbi_decode_letter_label

    if words_node_symbols is not None :
      if self.words_node_symbols != words_node_symbols:
        use_cache = False
        self.words_node_symbols = words_node_symbols

    if words_link_start_end is not None:
      if self.words_link_start_end != words_node_symbols:
        use_cache = False
        self.words_link_start_end = words_node_symbols

    if words_dictionary is not None:
      if self.words_dictionary != words_dictionary:
        use_cache = False
        self.words_dictionary = words_dictionary

    if insertion_penalty != self.insertion_penalty:
      use_cache = False
      self.insertion_penalty = insertion_penalty

    if (self.words_node_symbols is not None) and (self.words_link_start_end is not None) and (self.words_dictionary is not None):
      self.grammar_viterbi_decode_letter_label = Parallel(n_jobs=-1)(delayed(letter_bigram_viterbi_with_grammar_decode)(run_i, self.bigram_log_dict, self.words_node_symbols, self.words_link_start_end, self.words_dictionary, self.insertion_penalty) for run_i in self.naive_letter_prob)
      # self.grammar_viterbi_decode_letter_label = [letter_bigram_viterbi_with_grammar_decode(run_i, self.bigram_log_dict, self.words_node_symbols, self.words_link_start_end, self.words_dictionary, self.insertion_penalty) for run_i in self.naive_letter_prob]

    if self.grammar_viterbi_decode_letter_label is not None:
      latest_results = self.grammar_viterbi_decode_letter_label
    return latest_results

class SVMandInsertionPenaltyTunedSVMProbDecoder():
  def __init__(self, decoder, C_power_range=(0, 4), gamma_power_range=(0, 4), random_state=42, insertion_penalty_range=(-20, 0), n_splits = None, SVM_n_calls = 16, insertion_n_calls=16):
    self.decoder = decoder
    self.C_power_range = C_power_range
    self.gamma_power_range = gamma_power_range
    self.random_state = random_state
    self.insertion_penalty_range = insertion_penalty_range
    self.n_splits = n_splits
    self.insertion_n_calls = insertion_n_calls
    self.SVM_n_calls = SVM_n_calls

  def fit(self, X, y):
    previous_n_splits = self.n_splits
    if self.n_splits is None:
      self.n_splits = int(len(X)/2)
      # self.n_splits = len(X)
    kf = KFold(n_splits=self.n_splits, random_state=self.random_state, shuffle=True)
    cv_decoders = [copy.deepcopy(self.decoder) for i in range(self.n_splits)]
    x_test_all = []
    y_test_all = []
    x_train_all = []
    y_train_all = []
    for i, train_test_i in enumerate(kf.split(X)):
      train_i, test_i = train_test_i
      # print(train_i, test_i)
      x_train_i = [X[i] for i in train_i]
      y_train_i = [y[i] for i in train_i]
      x_test_i = [X[i] for i in test_i]
      y_test_i = [y[i] for i in test_i]
      x_test_all.append(x_test_i)
      y_test_all.append(y_test_i)
      x_train_all.append(x_train_i)
      y_train_all.append(y_train_i)
      decoder_i = cv_decoders[i]
      for step_name, step_obj in decoder_i.steps:
        if step_name == 'ZNormalizeBySub':
          step_obj.y_subjects = step_obj.x_subjects[test_i]
          step_obj.x_subjects = step_obj.x_subjects[train_i]
          step_obj.fit(x_train_i)

    self.x_test_all = x_test_all
    self.y_test_all = y_test_all
    self.x_train_all = x_train_all
    self.y_train_all = y_train_all
    self.cv_decoders = cv_decoders
    self.tune_SVM_params()
    self.tune_insertion_penalty()
    self.decoder = self.decoder.fit(X, y)
    self.n_splits = previous_n_splits
    return self

  def tune_SVM_params(self):
    x_test_all = self.x_test_all
    y_test_all = self.y_test_all
    x_train_all = self.x_train_all
    y_train_all = self.y_train_all
    cv_decoders = self.cv_decoders
    region_order = self.decoder.steps[-1][1].region_order

    def tune_SVM_for_r_i(r_i):
      def SVM_cost(c_power, gamma_power):
        c = 10 ** c_power
        gamma = 0.1 ** gamma_power

        def calculate_SVM_run_i_accuracy(x_train, y_train, x_test, y_test, decoder):
          decoder.steps[-1][1].SVM_params[r_i] = {'C': c, 'gamma':gamma}
          region_order = decoder.steps[-1][1].region_order
          LETTERS_TO_DOT = decoder.steps[-1][1].LETTERS_TO_DOT
          decoder.fit(x_train, y_train)
          prob = decoder.predict(x_test, svm_proba = True, r_i = r_i)
          y_pred_trans_class = np.array([np.argmax(run_i, axis=1) for run_i in prob])
          y_test_label_by_type = [[item[r_i] for item in run_i] for run_i in letter_label_to_transition_label(y_test, LETTERS_TO_DOT, region_order)]
          accuracy = accuracy_score([item for run_i in y_test_label_by_type for item in run_i], [item for run_i in y_pred_trans_class for item in run_i])
          return accuracy
        acc_all_run = Parallel(n_jobs=-1)(delayed(calculate_SVM_run_i_accuracy) (x_train, y_train, x_test, y_test, decoder) for x_train, y_train, x_test, y_test, decoder in zip(x_train_all, y_train_all, x_test_all, y_test_all, cv_decoders) )
        # acc_all_run = [calculate_SVM_run_i_accuracy(x_train, y_train, x_test, y_test, decoder) for x_train, y_train, x_test, y_test, decoder in zip(x_train_all, y_train_all, x_test_all, y_test_all, cv_decoders)]
        return -np.mean(acc_all_run)
      res = dlib.find_min_global(SVM_cost, [self.C_power_range[0], self.gamma_power_range[0]], [self.C_power_range[1], self.gamma_power_range[1]], self.SVM_n_calls)
      return res
    svm_params_res = Parallel(n_jobs=-1)(delayed(tune_SVM_for_r_i)(i) for i in range(len(region_order)) )

    # raise Exception('exit!')
    SVM_params = [{'C': 10 ** res[0][0], 'gamma': 0.1 ** res[0][1]} for res in svm_params_res]
    print([(res[1], param) for res, param in zip(svm_params_res, SVM_params)])
    self.best_SVM_params = SVM_params
    self.decoder.steps[-1][1].SVM_params = SVM_params
    for decoder in cv_decoders:
      decoder.steps[-1][1].SVM_params = SVM_params
    def fit_each_fold_decoder(x_train, y_train, decoder):
      return decoder.fit(x_train, y_train)
    cv_decoders = Parallel(n_jobs=-1)(delayed(fit_each_fold_decoder)(x_train, y_train, decoder) for x_train, y_train, decoder in zip(x_train_all, y_train_all, cv_decoders))
    self.cv_decoders = cv_decoders

  def tune_insertion_penalty(self):
    cv_decoders = self.cv_decoders
    x_test_all = self.x_test_all
    y_test_all = self.y_test_all
    # print([len(decoder.steps[-1][1].word_dictionary.keys()) for decoder in cv_decoders])
    def insertion_penalty_cost(insertion_penalty):
      def calculate_run_i_insertion_accuracy(x_test, y_test, decoder):
        y_pred = decoder.predict(x_test, insertion_penalty = insertion_penalty)
        return accuracy_score([l for run_i in y_test for l in run_i], [l for run_i in y_pred for l in run_i])
      acc_all_run = Parallel(n_jobs=-1)(delayed(calculate_run_i_insertion_accuracy) (x_test, y_test, decoder) for x_test, y_test, decoder in zip(x_test_all, y_test_all, cv_decoders))
      # acc_all_run = [calculate_run_i_insertion_accuracy(x_test, y_test, decoder) for x_test, y_test, decoder in zip(x_test_all, y_test_all, cv_decoders)]
      return -np.mean(acc_all_run)
    res = dlib.find_min_global(insertion_penalty_cost, [self.insertion_penalty_range[0]], [self.insertion_penalty_range[1]], self.insertion_n_calls)
    print(res)
    self.best_insertion_penalty = res[0]
    self.decoder.insertion_penalty = res[0]

  def predict(self, X):
    return self.decoder.predict(X, insertion_penalty = self.best_insertion_penalty)

  def predict_svm_transition(self, X):
    prob_trans = self.decoder.predict(X, svm_transition = True)
    return np.array(prob_trans)

  def get_trans_class(self):
    return self.decoder.steps[-1][1].trans_class

  def get_region_order(self):
    return self.decoder.steps[-1][1].region_order

  def get_prob_letter_label(self):
    return self.decoder.steps[-1][1].naive_prob_letter_label

  def get_bigram_weighted_letter_label(self):
    return self.decoder.steps[-1][1].bigram_weighted_letter_label

  def get_letter_viterbi_decode_letter_label(self):
    return self.decoder.steps[-1][1].letter_viterbi_decode_letter_label

  def get_naive_prob_letter_label(self):
    return self.decoder.steps[-1][1].naive_prob_letter_label

  def letter_label_to_transition_label(self, y):
    region_order = self.decoder.steps[-1][1].region_order
    LETTERS_TO_DOT = self.decoder.steps[-1][1].LETTERS_TO_DOT
    return letter_label_to_transition_label(y, LETTERS_TO_DOT, region_order)

  def set_grammar_related(self, bigram_prob_dict=None, words_node_symbols=None,
    words_link_start_end = None, words_dictionary = None):

    num_decoders = len(self.cv_decoders)
    if bigram_prob_dict is not None:
      for i in range(num_decoders):
        self.cv_decoders[i].steps[-1][1].add_bigram_dict(bigram_prob_dict)
      self.decoder.steps[-1][1].add_bigram_dict(bigram_prob_dict)
    if words_node_symbols is not None:
      for i in range(num_decoders):
        self.cv_decoders[i].steps[-1][1].words_node_symbols = words_node_symbols
      self.decoder.steps[-1][1].words_node_symbols = words_node_symbols
    if words_link_start_end is not None:
      for i in range(num_decoders):
        self.cv_decoders[i].steps[-1][1].words_link_start_end = words_link_start_end
      self.decoder.steps[-1][1].words_link_start_end = words_link_start_end
    if words_dictionary is not None:
      for i in range(num_decoders):
        self.cv_decoders[i].steps[-1][1].words_dictionary = words_dictionary
      self.decoder.steps[-1][1].words_dictionary = words_dictionary