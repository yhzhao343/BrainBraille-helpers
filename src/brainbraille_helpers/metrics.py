import numpy as np

def letter_label_to_word_label(letters_list):
  word_lists = [letters_list[0]]
  for l in letters_list[1:]:
    last_letter = word_lists[-1][-1]
    if ((last_letter == ' ') and (l == ' ')) or ((last_letter != ' ') and (l != ' ')):
      word_lists[-1] += l
    else:
      word_lists.append(l)
  return word_lists

def min_edit_dist(label, pred, delete_weight = 7, substitute_weight = 10, insert_weight = 7, tie_break = (1, 0, 2), style='htk'):
  # The tie_break parameter is an tuple. The content of the tuple is index to
  # the array of actions [delete, sub, insert]. The order denotes priority
  # HTK default tie_break is (1, 0, 2), meaning prefer sub over del over ins
  # NIST default tie_break is (0, 1, 2) meaning prefer del over sub over ins
  if isinstance(style, str):
    style_text = style.lower()
    if style_text == 'htk':
      tie_break = (1, 0, 2)
      delete_weight, substitute_weight, insert_weight = 7, 10, 7
    elif style_text == 'nist':
      tie_break = (0, 1, 2)
      delete_weight, substitute_weight, insert_weight = 3, 4, 3
  num_row = len(label) + 1
  num_col = len(pred) + 1
  # Meaning of the 3rd dimension of the w matrix
  #  0      1     2  3  4  5
  # option, dist, H, D, S, I
  # option is action options.
  # option | action | meaning                    |
  #   0    |  del   | vertival (row - 1)         |
  #   1    | sub/hit| diagonal (row - 1, col - 1)|
  #   2    |  ins   | horizontal (col - 1)       |

  w = np.zeros((num_row, num_col, 6), dtype=int)
  # Initialize edit distance at row 0 col 0
  w[0, :, 1] = np.arange(0, num_col * insert_weight, insert_weight)
  w[:, 0, 1] = np.arange(0, num_row * delete_weight, delete_weight)
  # Initialize action at row 0 col 0
  w[0, :, 0] = 2
  w[:, 0, 0] = 0
  w[0, 0, 0] = -1
  # Initialize ins and del count at row 0 and col 0
  w[0, 1:, 5] = np.arange(1, num_col)
  w[1:, 0, 3] = np.arange(1, num_row)
  D_S_I_col_ind = [3, 4, 5]

  for row in range(1, num_row):
    for col in range(1, num_col):
      s_cost_i = 0 if (pred[col - 1] == label[row - 1]) else substitute_weight
      # vertical, diagonal, horizontal
      #  delete ,   sub   ,   insert
      d_cost, s_cost, i_cost = w[row - 1][col][1] + delete_weight, w[row - 1][col - 1][1] + s_cost_i, w[row][col - 1][1] + insert_weight
      d_s_i_costs = [d_cost, s_cost, i_cost]

      if (d_s_i_costs[tie_break[0]] <= d_s_i_costs[tie_break[1]]) and (d_s_i_costs[tie_break[0]] <= d_s_i_costs[tie_break[2]]):
        action = tie_break[0]
      elif (d_s_i_costs[tie_break[2]] < d_s_i_costs[tie_break[1]]):
        action = tie_break[2]
      else:
        action = tie_break[1]

      if action == 0:
        w[row][col] = w[row - 1][col]
      elif action == 1:
        w[row][col] = w[row - 1][col - 1]
        if s_cost_i == 0:
          w[row][col][2] += 1
          w[row][col][D_S_I_col_ind[action]] -= 1
      else:
        w[row][col] = w[row][col - 1]
      w[row][col][0] = action
      w[row][col][1] = d_s_i_costs[action]
      w[row][col][D_S_I_col_ind[action]] += 1

  unique_tokens = sorted(np.unique([t for t in label + pred]))
  unique_token_dict = {tok: i for i, tok in enumerate(unique_tokens)}
  num_unique_tokens = len(unique_tokens)
  confusion_mat = np.zeros((num_unique_tokens + 1, num_unique_tokens + 1), dtype=int)
  cur_row, cur_col = row, col
  while ((cur_row > 0) and (cur_col > 0)):
    action = w[cur_row][cur_col][0]
    if action == 1:
      i = unique_token_dict[label[cur_row - 1]]
      j = unique_token_dict[pred[cur_col - 1]]
      confusion_mat[i][j] += 1
      cur_row, cur_col = cur_row - 1, cur_col - 1
    elif action == 0:
      i = unique_token_dict[label[cur_row - 1]]
      confusion_mat[i][-1] += 1
      cur_row, cur_col = cur_row - 1, cur_col
    else:
      j = unique_token_dict[pred[cur_col - 1]]
      confusion_mat[-1][j] += 1
      cur_row, cur_col = cur_row, cur_col - 1
  # print(unique_tokens)
  # print(confusion_mat)
  # print(np.sum(confusion_mat.diagonal()))
  return w[row][col][1:], confusion_mat, unique_tokens

def tok_acc(label, pred):
  w, conf_mat, toks = min_edit_dist(label, pred)
  (dist, hit, delete, substitute, insert) = w
  n = len(label)
  return (n - delete - substitute - insert) / n

def tok_corr(label, pred):
  w, conf_mat, toks = min_edit_dist(label, pred)
  (dist, hit, delete, substitute, insert) = w
  n = len(label)
  return (n - delete - substitute) / n

def hresult_style_pprint(label, minimum_edit_dist_result, confusion_mat, labels, default_max_label_len=5):
  (dist, hit, delete, substitute, insert) = minimum_edit_dist_result
  max_label_len = np.max([len(l) for l in labels])
  max_label_len = max_label_len if max_label_len < default_max_label_len else default_max_label_len
  max_number_digit = int(np.log10(np.max(confusion_mat))) + 1
  try:
    max_del_number_digit = int(np.log10(np.max(confusion_mat[:, -1]))) + 1
  except:
    max_del_number_digit = 5
  len_del_spacing = max_del_number_digit if max_del_number_digit > 4 else 4
  spacing = ' ' * (1 + max_number_digit)
  len_spacing = len(spacing)
  len_labels = len(labels)
  corr = (len(label) - delete - substitute) / len(label)
  acc = (len(label) - delete - substitute - insert) / len(label)
  output = ''
  output += '\n------------------------ Overall Results --------------------------\n'
  accuracy_string = f'WORD: %Corr={corr * 100:.2f}, Acc={acc * 100:.2f} [H:{hit} D:{delete} S:{substitute} I:{insert} N:{len(label)}]\n'
  output += accuracy_string
  output += '------------------------ Confusion Matrix -------------------------\n'
  header = '\n'.join([' ' * max_label_len + spacing + spacing.join([l[i] if len(l)>i else ' ' for l in labels]) for i in range(max_label_len)]) + f' {spacing}Del'.rjust(len_del_spacing) + ' [ %c / %e ]\n'
  output += header
  # print(header)
  # print(np.sum(confusion_mat[:-1][:-1]), hit + substitute)
  for i in range(len_labels):
    label_i = labels[i]
    row_sum = np.count_nonzero(confusion_mat[i, 0:-1])
    c = np.count_nonzero(confusion_mat[i][i])/row_sum if row_sum > 0 else float('nan')
    e = (np.count_nonzero(confusion_mat[i, :-1]) - confusion_mat[i][i]) / len(label)
    matrix_line = (label_i.rjust(max_label_len))[:max_label_len] + ''.join([f'{n}'.rjust(len_spacing + 1) for n in confusion_mat[i][0:-1]]) + f'{confusion_mat[i][-1]}'.rjust(len_del_spacing + len_spacing)
    if (not np.isnan(c)) and (c < 1) and ((c > 0) or (e > 0)):
      matrix_line += ' [' + f'{c*100:.1f}'.ljust(4) + '/' + f'{e*100:.1f}'.rjust(4) + ']'
    # print(matrix_line)
    output += (matrix_line + '\n')
  insert_line = ('Ins'.ljust(max_label_len))[:max_label_len] + ''.join([f'{n}'.rjust(len_spacing + 1) for n in confusion_mat[-1][0:-1]]) # + f'{confusion_mat[i][-1]}'.rjust(len_del_spacing)
  output += (insert_line + '\n===================================================================\n')
  return output