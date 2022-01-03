from .helpers import *
import cc3d
import numpy as np
import nibabel as nib
from itertools import combinations

def connected_component_clean(activation_mask, discard_size_ratio = 0.5):
  cleaned_activation_mask = np.zeros(activation_mask.shape, dtype=bool)
  activation_mask_cc, N = cc3d.connected_components(activation_mask, return_N=True)
  components_list = range(1, N+1)
  mask_sizes = [np.count_nonzero(activation_mask_cc == c) for c in components_list]
  max_mask_size = np.max(mask_sizes)
  discard_size_threshold = int(max_mask_size * discard_size_ratio)
  for c, c_component_size in zip(components_list, mask_sizes):
    if c_component_size > discard_size_threshold:
      component_mask = (activation_mask_cc == c)
      cleaned_activation_mask = np.logical_or(cleaned_activation_mask, component_mask)
  return cleaned_activation_mask

def delete_overlap(mask1, mask2):
  overlap_mask_reverse = np.logical_not(np.logical_and(mask1, mask2))
  mask1 = np.logical_and(mask1, overlap_mask_reverse)
  mask2 = np.logical_and(mask2, overlap_mask_reverse)
  return mask1, mask2


def get_roi_from_t_test(bold_info, processed_bold_info, t_threshold_quantile=0.98, use_calibration_masks = False, cc_discard_size_ratio=0.5, no_overlap = True, no_overlap_tilL_n = 6):
  t = processed_bold_info['per_run_results']['t']
  motor_mask = bold_info['mask_image']
  threshold = np.array([np.quantile(t_i[t_i > 0], t_threshold_quantile) for t_i in t.T])
  roi_mask = np.zeros(motor_mask.shape + (t.shape[1],), dtype=bool)
  for i in range(t.shape[1]):
    roi_mask[motor_mask, i] = t[:, i] > threshold[i]
    roi_mask[:, :, :, i] = connected_component_clean(roi_mask[:, :, :, i], cc_discard_size_ratio)

  if use_calibration_masks:
    mask_image_dictionary = bold_info['mask_image_dictionary']
    regressor_types = processed_bold_info['regressor_types']
    calibration_mask_dictionary = bold_info['calibration_mask_dictionary']
    calibration_masks = [mask_image_dictionary[calibration_mask_dictionary[regressor_type]] for regressor_type in regressor_types]
    for i, m in enumerate(calibration_masks):
      roi_mask[:, :, :, i] = np.logical_and(roi_mask[:, :, :, i], m)

  print([roi_mask[:, :, :, i].sum() for i in range(roi_mask.shape[-1])])
  if no_overlap:
    no_overlap_tilL_n = roi_mask.shape[-1] if (no_overlap_tilL_n is None) else no_overlap_tilL_n
    for i_1, i_2 in combinations(range(no_overlap_tilL_n), 2):
      roi_mask[:, :, :, i_1], roi_mask[:, :, :, i_2] = delete_overlap(roi_mask[:, :, :, i_1], roi_mask[:, :, :, i_2])
  print([roi_mask[:, :, :, i].sum() for i in range(roi_mask.shape[-1])])
  return roi_mask

def get_roi_and_calibration_from_t(bold_info, processed_bold_info, t_threshold_quantile=0.98, use_calibration_masks = False, cc_discard_size_ratio=0.5, no_overlap = True, no_overlap_tilL_n = 6):
  roi_mask = get_roi_from_t_test(bold_info, processed_bold_info, t_threshold_quantile, use_calibration_masks, cc_discard_size_ratio, no_overlap, no_overlap_tilL_n)
  calibration_mask = get_calibration_roi(roi_mask, bold_info, processed_bold_info, use_calibration_masks)
  return roi_mask, calibration_mask


def get_calibration_roi(roi_mask, bold_info, processed_bold_info, use_calibration_masks = False):
  base_calibration_mask = get_base_calibration_roi(roi_mask, bold_info['mask_image'])
  calibration_mask = np.stack([base_calibration_mask] * roi_mask.shape[-1], axis=-1)
  if use_calibration_masks:
    mask_image_dictionary = bold_info['mask_image_dictionary']
    regressor_types = processed_bold_info['regressor_types']
    calibration_mask_dictionary = bold_info['calibration_mask_dictionary']
    base_calibrations = [mask_image_dictionary[calibration_mask_dictionary[regressor_type]] for regressor_type in regressor_types]
    for i, base_calibration in enumerate(base_calibrations):
      calibration_mask[:, :, :, i] = np.logical_and(base_calibration_mask, base_calibration)
  # print([calibration_mask[:, :, :, i].sum() for i in range(calibration_mask.shape[-1])])
  return calibration_mask


def get_base_calibration_roi(roi_mask, motor_mask):
  not_roi_mask = np.logical_not(roi_mask)
  base_calibration_mask = np.all(not_roi_mask, axis=-1)
  base_calibration_mask = np.logical_and(base_calibration_mask, motor_mask)
  # print(np.prod(calibration_mask.shape[0:3]))
  # print(not_roi_mask.shape, not_roi_mask.T.shape, calibration_mask.shape, motor_mask.sum(), calibration_mask.sum())
  return base_calibration_mask

def get_aggregated_roi(roi_masks, aggregation_threshold_quantile = 0.8, cc_discard_size_ratio = 0.5, no_overlap = True, no_overlap_tilL_n = 6):
  roi_votes = np.count_nonzero(roi_masks, axis=0)
  threshold = np.array([np.quantile(roi_votes[:, :, :, r_i][roi_votes[:, :, :, r_i] > 0], aggregation_threshold_quantile) for r_i in range(roi_votes.shape[-1])])
  roi_mask = np.stack([connected_component_clean(roi_votes[:, :, :, r_i] >= th_i, cc_discard_size_ratio) for r_i, th_i in enumerate(threshold)], axis=-1)
  # print(roi_mask.shape)
  if no_overlap:
    no_overlap_tilL_n = roi_mask.shape[-1] if (no_overlap_tilL_n is None) else no_overlap_tilL_n
    for i_1, i_2 in combinations(range(no_overlap_tilL_n), 2):
      roi_mask[:, :, :, i_1], roi_mask[:, :, :, i_2] = delete_overlap(roi_mask[:, :, :, i_1], roi_mask[:, :, :, i_2])
  # print([roi_mask[:, :, :, i].sum() for i in range(roi_mask.shape[-1])])
  print([roi_mask[:, :, :, i].sum() for i in range(roi_mask.shape[-1])])
  return roi_mask

def extract_train_test_data_and_label(train_bold_infos_processed_bold_infos, test_bold_infos_processed_bold_infos):
  train_bold_infos, train_processed_bold_infos = zip(*train_bold_infos_processed_bold_infos)
  test_bold_infos,   test_processed_bold_infos = zip(*test_bold_infos_processed_bold_infos)

  train_data = [train_processed_bold_info['per_run_results']['extracted_data'] for train_processed_bold_info in train_processed_bold_infos]
  train_calibration_data = [train_processed_bold_info['per_run_results']['calibration_data'] for train_processed_bold_info in train_processed_bold_infos]
  # train_file_paths = np.array([f'{info["func_path"]}/{info["bold_file_name"]}' for info in train_bold_infos])
  test_file_paths  = np.array([f'{info["func_path"]}/{info["bold_file_name"]}' for info in test_bold_infos])
  # train_bold_images = [processed_bold_info['bold_image'] for processed_bold_info in train_processed_bold_infos]
  # test_bold_images =  [processed_bold_info['bold_image'] for processed_bold_info in test_processed_bold_infos]

  train_roi_masks = np.array([train_processed_bold_info['per_run_results']['roi_mask'] for train_processed_bold_info in train_processed_bold_infos])
  train_calibration_masks = np.array([train_processed_bold_info['per_run_results']['calibration_mask'] for train_processed_bold_info in train_processed_bold_infos])

  train_subs = np.array([info['sub'] for info in train_bold_infos], dtype=int)
  test_subs =  np.array([info['sub'] for info in test_bold_infos], dtype=int)

  unique_subs = np.unique(train_subs)
  aggregated_roi_by_sub = {}
  for sub_i in unique_subs:
    sub_i_mask = train_subs == sub_i
    sub_i_train_roi_masks = train_roi_masks[sub_i_mask]
    aggregated_roi_by_sub[sub_i] = get_aggregated_roi(sub_i_train_roi_masks)
  test_roi_masks = np.array([aggregated_roi_by_sub[s] for s in test_subs])
  test_calibration_masks = np.array([get_calibration_roi(mask, info, processed_bold_info) for mask, info, processed_bold_info in zip(test_roi_masks, test_bold_infos, test_processed_bold_infos)])

  test_data, test_calibration_data   = zip(*[extract_data_from_nifty(path, roi_mask, calibration_mask) for path, roi_mask, calibration_mask in zip(test_file_paths, test_roi_masks, test_calibration_masks)])
  # train_data, train_calibration_data = zip(*[extract_data_from_bold_image(bold, roi_mask, calibration_mask) for bold, roi_mask, calibration_mask in zip(train_bold_images, train_roi_masks, train_calibration_masks)])

  test_relative_data =              [(data_i - data_i[0, :]) / data_i[0, :] for data_i in test_data]
  test_calibration_relative_data =  [(data_i - data_i[0, :]) / data_i[0, :] for data_i in test_calibration_data]
  train_relative_data =             [(data_i - data_i[0, :]) / data_i[0, :] for data_i in train_data]
  train_calibration_relative_data = [(data_i - data_i[0, :]) / data_i[0, :] for data_i in train_calibration_data]

  test_calibrated_data =  [data_i - calib_i  for data_i, calib_i in zip(test_relative_data, test_calibration_relative_data)]
  train_calibrated_data = [data_i - calib_i  for data_i, calib_i in zip(train_relative_data, train_calibration_relative_data)]

  train_events = [processed_bold_info['event_tsv_content'] for processed_bold_info in train_processed_bold_infos]
  train_letter_labels = [[event['letter'] if event['letter'] != '_' else ' ' for i, event in enumerate(events) if (i > 0 and (event['onset'] != events[i - 1]['onset'])) or i == 0] for events in train_events]
  test_events =  [processed_bold_info['event_tsv_content'] for processed_bold_info in test_processed_bold_infos ]
  test_letter_labels =  [[event['letter'] if event['letter'] != '_' else ' ' for i, event in enumerate(events) if (i > 0 and (event['onset'] != events[i - 1]['onset'])) or i == 0] for events in test_events]

  return {
    'train': {
        'data': train_calibrated_data,
        'label': train_letter_labels
      },
    'test': {
        'data': test_calibrated_data,
        'label': test_letter_labels
      }
  }

def load_bold_image(bold_image_path):
  bold_image_handle = nib.load(bold_image_path)
  return bold_image_handle.get_fdata(dtype=np.float32)

def extract_data_from_bold_image(bold_image, mask, calibration_mask):
  extracted_data = np.stack([(bold_image[mask[:, :, :, i], :]).mean(axis=0) for i in range(mask.shape[-1])]).T
  extracted_calibration_data = np.stack([(bold_image[calibration_mask[:, :, :, i], :]).mean(axis=0) for i in range(calibration_mask.shape[-1])]).T
  return extracted_data, extracted_calibration_data

def extract_data_from_nifty(bold_image_path, mask, calibration_mask):
  bold_image = load_bold_image(bold_image_path)
  return extract_data_from_bold_image(bold_image, mask, calibration_mask)

def extract_data_and_label_using_roi(train_bold_infos_processed_bold_infos, test_bold_infos_processed_bold_infos, roi_masks_by_sub):
  train_bold_infos, train_processed_bold_infos = zip(*train_bold_infos_processed_bold_infos)
  test_bold_infos,   test_processed_bold_infos = zip(*test_bold_infos_processed_bold_infos)
  train_subs = np.array([info['sub'] for info in train_bold_infos], dtype=int)
  test_subs = np.array([info['sub'] for info in test_bold_infos], dtype=int)
  train_file_paths = np.array([f'{info["func_path"]}/{info["bold_file_name"]}' for info in train_bold_infos])
  test_file_paths  = np.array([f'{info["func_path"]}/{info["bold_file_name"]}' for info in test_bold_infos])
  extracted_train_data = [extract_data_from_nifty(train_file_paths[i], roi_masks_by_sub[train_subs[i]]) for i, path in enumerate(train_file_paths)]
  extracted_test_data = [extract_data_from_nifty(test_file_paths[i], roi_masks_by_sub[train_subs[i]]) for i, path in enumerate(test_file_paths)]
  train_events = [processed_bold_info['event_tsv_content'] for processed_bold_info in train_processed_bold_infos]
  train_letter_labels = [[event['letter'] if event['letter'] != '_' else ' ' for i, event in enumerate(events) if (i > 0 and (event['onset'] != events[i - 1]['onset'])) or i == 0] for events in train_events]
  test_events =  [processed_bold_info['event_tsv_content'] for processed_bold_info in test_processed_bold_infos ]
  test_letter_labels =  [[event['letter'] if event['letter'] != '_' else ' ' for i, event in enumerate(events) if (i > 0 and (event['onset'] != events[i - 1]['onset'])) or i == 0] for events in test_events]

  return {
      'train': {
          'data': extracted_train_data,
          'label': train_letter_labels
        },
      'test': {
          'data': extracted_test_data,
          'label': test_letter_labels
        }
    }

def extract_label_from_info(bold_info_processed_bold_info, LETTERS_TO_DOT=None):
  bold_info, processed_bold_info =  bold_info_processed_bold_info
  TR_s = bold_info['TR_s']
  events = processed_bold_info['event_tsv_content']
  regressor_types = processed_bold_info['regressor_types']
  # print(regressor_types)
  event_onsets_s = np.array([int(event['onset']) for i, event in enumerate(events) if (i > 0 and (event['onset'] != events[i - 1]['onset'])) or i == 0])
  event_onsets_frame = (event_onsets_s / TR_s).astype(int)
  letter_labels = [event['letter'] for i, event in enumerate(events) if (i > 0 and (event['onset'] != events[i - 1]['onset'])) or i == 0]
  letter_labels = [l if l != '_' else ' ' for l in letter_labels]
  transition_letter_labels = [f'{l}{letter_labels[i + 1]}' for i, l in enumerate(letter_labels[:-1])]
  letters = np.unique(letter_labels)
  if LETTERS_TO_DOT is None:
    LETTERS_TO_DOT = {l: {regressor: 0 for regressor in regressor_types} for l in letters}
    letter_time_dict = {}
    for e in events:
      l = ' ' if e['letter'] == '_' else e['letter']
      t = int(e['onset'])
      if l not in letter_time_dict:
        letter_time_dict[l] = t
      if (letter_time_dict[l] == t) and (e['trial_type'] in regressor_types):
        LETTERS_TO_DOT[l][e['trial_type']] = 1
  dot_labels = [LETTERS_TO_DOT[l] for l in letter_labels]
  return {'dot_labels': dot_labels, 'letter_labels': letter_labels}

def get_LETTERS_TO_DOT_from_processed_info(processed_bold_info):
  events = processed_bold_info['event_tsv_content']
  regressor_types = processed_bold_info['regressor_types']
  letter_labels = [event['letter'] for i, event in enumerate(events) if (i > 0 and (event['onset'] != events[i - 1]['onset'])) or i == 0]
  letter_labels = [l if l != '_' else ' ' for l in letter_labels]
  # transition_letter_labels = [f'{l}{letter_labels[i + 1]}' for i, l in enumerate(letter_labels[:-1])]
  letters = np.unique(letter_labels)
  LETTERS_TO_DOT = {l: {regressor: 0 for regressor in regressor_types} for l in letters}
  letter_time_dict = {}
  for e in events:
    l = ' ' if e['letter'] == '_' else e['letter']
    t = int(e['onset'])
    if l not in letter_time_dict:
      letter_time_dict[l] = t
    if (letter_time_dict[l] == t) and (e['trial_type'] in regressor_types):
      LETTERS_TO_DOT[l][e['trial_type']] = 1
  return LETTERS_TO_DOT
