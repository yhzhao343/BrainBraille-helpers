from .helpers import *
from .glm import *
from .HTK_Hmm import *
import numpy as np
from scipy.signal import butter, sosfilt
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
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
      self.n_splits = len(X) - 1
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