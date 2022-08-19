from .helpers import *
from functools import partial, reduce
import numbers
import itertools
import nibabel as nib
import templateflow
import os
import numba
import numpy as np
import scipy
from numba import jit, prange
from joblib import Parallel, delayed


def sigma_to_fwhm(sigma):
    # https://brainder.org/2011/08/20/gaussian-kernels-convert-fwhm-to-sigma/
    return sigma * np.sqrt(8 * np.log(2))


def fwhm_to_sigma(fwhm):
    # https://brainder.org/2011/08/20/gaussian-kernels-convert-fwhm-to-sigma/
    return fwhm / np.sqrt(8 * np.log(2))


def get_inv_v_from_ar(a, num_frame):
    # make sure numba parallel is False! parallel=True hurts performance
    len_a = len(a)
    A_upper_left = np.eye(len_a + 1, dtype=np.float32)
    for i in range(len_a):
        A_upper_left[i+1:len_a+1, i] = a[:len_a-i]
    inv_v_upper_left = A_upper_left @ A_upper_left.T
    xcorr_a = inv_v_upper_left[-1, :-1]
    iv = np.eye(num_frame, dtype=np.float32) * inv_v_upper_left[-1, -1]
    iv[:len_a+1, :len_a+1] = inv_v_upper_left
    for i in range(len_a+1, num_frame):
        iv[i-len_a:i, i] = xcorr_a
        iv[i, i-len_a:i] = xcorr_a
    return iv


def iv_to_prewhiten_matrix(iv):
    u, s, ut = np.linalg.svd(iv, hermitian=True)
    w = u @ np.diag(np.sqrt(s)) @ ut
    return w


# TODO:Maybe rewrite all the string parsing using regex?
# I am very bad with regex
# def identify_template(file_path):
# # Identify the template used using file name
#     return [
#         str_seg for str_seg in file_path.split('/')[-1].split('_')
#         if str_seg.startswith('space')
#     ][0].split('-')[-1]


def extract_bold_file_info_from_name(file_path):
    path_segments = file_path.split('/')
    func_path = '/'.join(path_segments[0:-1])
    bold_file_name = path_segments[-1]
    bold_file_name_segments = bold_file_name.split('_')
    sub = [
        str_segment for str_segment in bold_file_name_segments
        if str_segment.startswith('sub')
    ][0].split('-')[-1]
    run = [
        str_segment for str_segment in bold_file_name_segments
        if str_segment.startswith('run')
    ][0].split('-')[-1]
    task = [
        str_segment for str_segment in bold_file_name_segments
        if str_segment.startswith('task')
    ][0].split('-')[-1]
    space = [
        str_segment for str_segment in bold_file_name_segments
        if str_segment.startswith('space')
    ][0].split('-')[-1]
    brain_mask_file_name = \
        f'{bold_file_name.split("_desc-")[0]}_desc-brain_mask.nii.gz'

    ses = file_path.split('ses-')[1].split('/')[0]
    timeseries_file = \
        f'sub-{sub}_task-{task}_run-{run}_desc-confounds_timeseries'
    derivatives_path = f'{file_path.split("derivatives")[0]}derivatives'
    if not os.path.exists(f'{func_path}/{timeseries_file}.tsv'):
        timeseries_file = f'sub-{sub}_ses-{ses}_task-{task}_run-{run}_desc-'\
            'confounds_timeseries'
    info = {
            'sub': sub,
            'run': run,
            'ses': ses,
            'task': task,
            'space': space,
            'func_path': func_path,
            'bold_file_name': bold_file_name,
            'timeseries_json_name': f'{timeseries_file}.json',
            'timeseries_tsv_name': f'{timeseries_file}.tsv',
            'brain_mask_file_name': brain_mask_file_name,
            'brain_mask_file_path': f'{func_path}/{brain_mask_file_name}',
            'derivatives_path': derivatives_path
        }
    info['timeseries_json_path'] = \
        f'{info["func_path"]}/{info["timeseries_json_name"]}'
    info['timeseries_tsv_path'] = \
        f'{info["func_path"]}/{info["timeseries_tsv_name"]}'
    info['bold_file_path'] = f'{info["func_path"]}/{info["bold_file_name"]}'
    return info


def spm_hrf(
    resolution_s, peak_delay_s=6, undershoot_delay_s=16, peak_dispersion=1,
    undershoot_dispersion=1, peak_undershoot_ratio=6, onset_s=0,
    model_length_s=32
):
    time_stamp = np.array(
        np.arange(
            0, np.ceil(model_length_s/resolution_s)
        ) - onset_s/resolution_s
    )
    peak_values = scipy.stats.gamma.pdf(
        time_stamp, peak_delay_s / peak_dispersion, loc=0,
        scale=peak_dispersion / resolution_s
    )
    undershoot_values = scipy.stats.gamma.pdf(
        time_stamp, undershoot_delay_s / undershoot_dispersion, loc=0,
        scale=undershoot_dispersion / resolution_s
    )
    hrf = peak_values - undershoot_values / peak_undershoot_ratio
    return hrf, time_stamp * resolution_s


def spm_d_hrf(
    resolution_s, peak_delay_s=6, undershoot_delay_s=16, peak_dispersion=1,
    undershoot_dispersion=1, peak_undershoot_ratio=6, onset_s=0,
    model_length_s=32, delta_t_s=1
):
    return spm_hrf(
        resolution_s, peak_delay_s, undershoot_delay_s, peak_dispersion,
        undershoot_dispersion,
        peak_undershoot_ratio, onset_s, model_length_s
    )[0] - spm_hrf(
        resolution_s, peak_delay_s, undershoot_delay_s, peak_dispersion,
        undershoot_dispersion, peak_undershoot_ratio, delta_t_s, model_length_s
    )[0]


def spm_dd_hrf(
    resolution_s, peak_delay_s=6, undershoot_delay_s=16, peak_dispersion=1,
    undershoot_dispersion=1, peak_undershoot_ratio=6, onset_s=0,
    model_length_s=32, d_dispersion=0.01
):
    return (
        spm_hrf(
            resolution_s, peak_delay_s, undershoot_delay_s, peak_dispersion,
            undershoot_dispersion, peak_undershoot_ratio, onset_s,
            model_length_s
        )[0] - spm_hrf(
            resolution_s, peak_delay_s, undershoot_delay_s,
            peak_dispersion + d_dispersion, undershoot_dispersion,
            peak_undershoot_ratio, onset_s, model_length_s
        )[0]
    )/0.01


def stimulus_to_stimulus_sequence(
    stimulus, SPACE_BEFORE_RUN, SPACE_BETWEEN_WORD, SPACE_BETWEEN_PHRASE,
    SPACE_AFTER_RUN
):
    stimulus_sequence = ' ' * SPACE_BEFORE_RUN
    for stimuli_phrase in stimulus:
        for word in stimuli_phrase.split(' '):
            stimulus_sequence += (word + ' ' * SPACE_BETWEEN_WORD)
        stimulus_sequence += ' ' * (SPACE_BETWEEN_PHRASE - SPACE_BETWEEN_WORD)
    if (SPACE_AFTER_RUN > SPACE_BETWEEN_PHRASE):
        stimulus_sequence += ' ' * (SPACE_AFTER_RUN - SPACE_BETWEEN_PHRASE)
    elif (SPACE_BETWEEN_PHRASE > SPACE_AFTER_RUN):
        stimulus_sequence = stimulus_sequence[
            0:(SPACE_AFTER_RUN - SPACE_BETWEEN_PHRASE)
        ]
    return stimulus_sequence


MOTION_CONFOUND_LIST = [
    'trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z'
]
MOTION_DERIVATIVE1_CONFOUND_LIST = [
    'trans_x_derivative1', 'trans_y_derivative1', 'trans_z_derivative1',
    'rot_x_derivative1', 'rot_y_derivative1', 'rot_z_derivative1'
]
MOTION_POW2_CONFOUND_LIST = [
    'trans_x_power2', 'trans_y_power2', 'trans_z_power2',
    'rot_x_power2', 'rot_y_power2', 'rot_z_power2'
]
MOTION_DERIVATIVE1_POW2_CONFOUND_LIST = [
    'trans_x_derivative1_power2', 'trans_y_derivative1_power2',
    'trans_z_derivative1_power2', 'rot_x_derivative1_power2',
    'rot_y_derivative1_power2', 'rot_z_derivative1_power2'
]
CSF_CONFOUND_LIST = [
    'csf', 'csf_derivative1', 'csf_derivative1_power2', 'csf_power2'
]
WM_CONFOUND_LIST = [
    'white_matter', 'white_matter_derivative1',
    'white_matter_derivative1_power2', 'white_matter_power2'
]


def extract_confound_regressors(
    confound_dict_list, USE_MOTION=True, USE_MOTION_DERIVATIVE1=True,
    USE_MOTION_POW2=True, USE_MOTION_DERIVATIVE1_POW2=True, USE_CSF=True,
    USE_WM=False, USE_MOTION_OUTLIER=True
):
    total_confound_list = []
    if USE_MOTION:
        total_confound_list += MOTION_CONFOUND_LIST
    if USE_MOTION_DERIVATIVE1:
        total_confound_list += MOTION_DERIVATIVE1_CONFOUND_LIST
    if USE_MOTION_POW2:
        total_confound_list += MOTION_POW2_CONFOUND_LIST
    if USE_MOTION_DERIVATIVE1_POW2:
        total_confound_list += MOTION_DERIVATIVE1_POW2_CONFOUND_LIST
    if USE_CSF:
        CSF_regressor_indices = len(total_confound_list) + \
            np.arange(len(CSF_CONFOUND_LIST))
        total_confound_list += CSF_CONFOUND_LIST
    if USE_WM:
        WM_regressor_indices = len(total_confound_list) + \
            np.arange(len(WM_CONFOUND_LIST))
        total_confound_list += WM_CONFOUND_LIST
    if USE_MOTION_OUTLIER:
        total_confound_list += [
            confound_key for confound_key in
            list(confound_dict_list[0].keys())
            if confound_key.startswith('motion_outlier')
        ]
    confound_regressors = np.array(
        [
            [
                float(row[confound])
                if row[confound] != 'n/a' else float('nan')
                for confound in total_confound_list
            ] for row in confound_dict_list
        ], dtype=np.float32
    )
    derivative1_confound_mask = np.array(
        ['derivative1' in confound for confound in total_confound_list],
        dtype=bool
    )
    confound_regressors[0, derivative1_confound_mask] = \
        confound_regressors[1, derivative1_confound_mask]
    if USE_CSF:
        confound_regressors[:, CSF_regressor_indices] = \
            confound_regressors[:, CSF_regressor_indices] / np.max(
                confound_regressors[:, CSF_regressor_indices], axis=0
            )
    if USE_WM:
        confound_regressors[:, WM_regressor_indices] = \
            confound_regressors[:, WM_regressor_indices] / np.max(
                confound_regressors[:, WM_regressor_indices], axis=0
            )

    return confound_regressors.astype(np.float32)


def event_dict_list_to_event_by_type(
    event_dict_list, trial_types_to_ignore=['rest'], merge_adjacent=True
):
    event_by_type = {}
    for event_dict in event_dict_list:
        trial_type = event_dict['trial_type']
        if trial_type in trial_types_to_ignore:
            continue
        if trial_type not in event_by_type:
            event_by_type[trial_type] = []

        # Check to see if the previous stimulus is adjacent to the current one
        if len(event_by_type[trial_type]):
            previous_event = event_by_type[trial_type][-1]
            previous_stimulus_end_time_s = previous_event['onset_s'] + \
                previous_event['duration_s']
            # If the previous stimulus extends to the onset of the current,
            # merge the two to one
            if merge_adjacent:
                if previous_stimulus_end_time_s == float(event_dict['onset']):
                    event_by_type[trial_type][-1]['duration_s'] += \
                        float(event_dict['duration'])
                    continue

        event_by_type[trial_type].append({
                'onset_s': float(event_dict['onset']),
                'duration_s': float(event_dict['duration'])
            })

    event_list = list(event_by_type.keys())
    return event_by_type


def event_dict_list_to_letter_event_list(event_dict_list):
    letter_event_list = []
    letter_event_list.append({
            'onset_s': float(event_dict_list[0]['onset']),
            'duration_s': float(event_dict_list[0]['duration']),
            'letter': event_dict_list[0]['letter']
        })
    for event_dict in event_dict_list[1:]:
        if float(event_dict['onset']) != letter_event_list[-1]['onset_s']:
            letter_event_list.append({
                'onset_s': float(event_dict['onset']),
                'duration_s': float(event_dict['duration']),
                'letter': event_dict['letter']
            })
    return letter_event_list


def gam(t, discount=0.7):
    return (1 - 0.6658) * (np.exp(-discount * t) / np.exp(-discount))


def event_list_to_onset_array(
    event_list, run_len_s, resolution_s=0.01, method='nonlinear'
):
    onset_array = np.zeros(int(run_len_s / resolution_s), dtype=np.float32)
    if method == 'constant_impulse':
        step = int(4 / resolution_s)
        for event in event_list:
            onset_frame = int(event['onset_s'] / resolution_s)
            stop_frame = int(
                (event['onset_s'] + event['duration_s']) / resolution_s
            )
            onset_array[np.arange(onset_frame, stop_frame, step)] = 1
    elif method == 'constant_epoch_2s':
        for event in event_list:
            onset_frame = int(event['onset_s'] / resolution_s)
            stop_frame = int((event['onset_s'] + 2) / resolution_s)
            onset_array[onset_frame:stop_frame] = 1 / \
                (stop_frame - onset_frame)
    elif method == 'variable_epoch':
        for event in event_list:
            onset_frame = int(event['onset_s'] / resolution_s)
            stop_frame = int(
                (event['onset_s'] + event['duration_s']) / resolution_s
            )
            onset_array[onset_frame:stop_frame] = 1 / \
                (stop_frame - onset_frame)
    elif method == 'nonlinear':
        gam_weight_vector = gam(np.arange(1, run_len_s + 1))
        gam_weight_mask = np.zeros(len(gam_weight_vector), dtype=bool)
        prev_onset_time_s = 0
        for event in event_list:
            onset_frame = int(event['onset_s'] / resolution_s)
            onset_time_s = event['onset_s']
            gam_weight_mask = np.roll(gam_weight_mask, int(
                onset_time_s - prev_onset_time_s)
            )
            sat = np.sum(gam_weight_vector[gam_weight_mask])
            onset_array[onset_frame] = max(0, 1 - sat)
            gam_weight_mask[0] = True
            duration_left = int(round(event['duration_s'] - 1))
            for d_i in range(duration_left):
                onset_frame = int(onset_frame + 1/resolution_s)
                onset_time_s += 1
                sat = np.sum(gam_weight_vector[gam_weight_mask])
                onset_array[onset_frame] = max(0, 1 - sat)
                gam_weight_mask = np.roll(gam_weight_mask, 1)
                gam_weight_mask[0] = True
            prev_onset_time_s = onset_time_s
    return onset_array


def dct_drift_basis(frame_timestamps, hf_cut_hz=0.008):
    num_frame = len(frame_timestamps)
    n_times = np.arange(num_frame)
    order = max(int(np.floor(2 * hf_cut_hz * (
        frame_timestamps[-1] - frame_timestamps[0]))
    ), 1)
    drift = np.zeros((num_frame, order), dtype=np.float32)
    nfct = 1
    for k in range(1, order):
        drift[:, k] = nfct * np.cos((np.pi / num_frame) * (n_times + 0.5) * k)
    drift[:, 0] = 1.0
    return drift


def onset_array_convolve_with_basis(
    onset_array, basis, TR_s, resolution_s=0.01
):
    len_onset_array = len(onset_array)
    steps = int(TR_s / resolution_s)
    num_frames = int(len(onset_array) / steps)
    basis_len, num_basis = basis.shape
    output = np.zeros((num_frames, num_basis), dtype=np.float32)
    # offset = math.floor((basis_len - 1) / 2)
    for i in range(num_basis):
        # output[:, i] = np.convolve(
        #     onset_array, basis[:, i]
        # )[np.arange(offset, len_onset_array + offset, steps)]
        output[:, i] = np.convolve(
            onset_array, basis[:, i]
        )[np.arange(0, len_onset_array, steps)[0:num_frames]]
    return output


# TODO: this is a pretty lazy way of doing things
def scrub_motion_outliner(bold_image, mask, outlier_index):
    x_size, y_size, z_size, num_frame = bold_image.shape
    frame_index = np.arange(num_frame)
    valid_bold_mask = np.alltrue(bold_image != 0, axis=-1)
    mask = np.logical_and(mask, valid_bold_mask)
    for x in range(x_size):
        for y in range(y_size):
            for z in range(z_size):
                if mask[x, y, z]:
                    time_series = bold_image[x, y, z, :]
                    time_series_without_outlier = np.delete(
                        time_series, outlier_index
                    )
                    frame_index_without_outlier = np.delete(
                        frame_index, outlier_index
                    )
                    interpolated_vals = np.interp(
                        outlier_index, frame_index_without_outlier,
                        time_series_without_outlier
                    ).astype(np.float32)
                    bold_image[x, y, z, outlier_index] = interpolated_vals
    return bold_image


@jit(nopython=True, parallel=False, fastmath=True, cache=True)
def matmul_1d_mat1_batch_1d_mat2(mat1, mat2):
    # It seems numba can only speed this a little bit
    num_voxel = mat2.shape[0]
    out_size = mat1.shape[0]
    out = np.zeros((num_voxel, out_size), dtype=np.float32)
    for i in range(num_voxel):
        out[i, :] = mat1 @ mat2[i, :]
    return out


@jit(nopython=True, parallel=False, fastmath=True, cache=True)
def matmul_1d_mat1_batch_2d_mat2(mat1, mat2):
    num_voxel = mat2.shape[0]
    out = np.zeros((num_voxel, mat1.shape[0], mat2.shape[2]), dtype=np.float32)
    for i in range(num_voxel):
        out[i] = mat1 @ mat2[i]
    return out


@jit(nopython=True, parallel=False, fastmath=True, cache=True)
def matmul_2d_batch_1d_mat1_2d_mat2(mat1, mat2):
    num_voxel = mat1.shape[0]
    out = np.zeros((num_voxel, mat1.shape[1], mat2.shape[1]), dtype=np.float32)
    for i in range(num_voxel):
        # print(mat1[i].shape, mat2.shape, (mat1[i] @ mat2).shape)
        out[i] = mat1[i] @ mat2
    return out


@jit(nopython=True, parallel=False, fastmath=True, cache=True)
def matdot_2d_batch_1d_mat1_1d_mat2(mat1, mat2):
    num_voxel = mat1.shape[0]
    out = np.zeros((num_voxel, mat1.shape[1]), dtype=np.float32)
    for i in range(num_voxel):
        for j in range(mat1.shape[1]):
            out[i, j] = np.dot(mat1[i, j, :], mat2[:, j])
    return out


def batch_xcorr(x, order=4):
    num_voxel, len_x = x.shape
    out = np.zeros((num_voxel, order + 1), dtype=np.float32)
    for i in range(order + 1):
        out[:, i] = np.sum(x[:, 0:(len_x-i)] * x[:, i:], axis=1)
    return out


def batch_AR_from_xcorr_using_Yule_Walker(xcorr):
    # NOTE: numba does not have solve_toeplitz
    num_frame, ar_order_plus_1 = xcorr.shape
    out = np.zeros((num_frame, ar_order_plus_1 - 1), dtype=np.float32)
    for i in range(num_frame):
        out[i, :] = scipy.linalg.solve_toeplitz(xcorr[i, :-1], -xcorr[i, 1:])
    return out


@jit(nopython=True, parallel=False, fastmath=True, cache=True)
def batch_sse_from_xcorr_and_ar(xcorr, a):
    num_voxel = xcorr.shape[0]
    sse = np.zeros(num_voxel, dtype=np.float32)
    for i in range(num_voxel):
        sse[i] = xcorr[i, 0] - a[i, :] @ (-xcorr[i, 1:])
    return sse


@jit(nopython=True, parallel=False, fastmath=True, cache=True)
def get_inv_v_from_ar(a, num_frame):
    # make sure numba parallel is False! parallel=True hurts performance
    len_a = len(a)
    A_upper_left = np.eye(len_a + 1, dtype=np.float32)
    for i in range(len_a):
        A_upper_left[i+1:len_a+1, i] = a[:len_a-i]
    inv_v_upper_left = A_upper_left @ A_upper_left.T
    xcorr_a = inv_v_upper_left[-1, :-1]
    iv = np.eye(num_frame, dtype=np.float32) * inv_v_upper_left[-1, -1]
    iv[:len_a+1, :len_a+1] = inv_v_upper_left
    for i in range(len_a+1, num_frame):
        iv[i-len_a:i, i] = xcorr_a
        iv[i, i-len_a:i] = xcorr_a
    return iv


@jit(nopython=True, parallel=False, fastmath=True, cache=True)
def get_ar_filter_from_ar(a, num_frame):
    # make sure numba parallel is False! parallel=True hurts performance
    len_a = len(a)
    num_full_a = num_frame - len_a
    ar_f = np.eye(num_frame, dtype=np.float32)
    # for i in range(len_a):
    #     ar_f = ar_f + a[i] * np.eye(num_frame, dtype=np.float32, k=-i-1)
    for i in range(num_full_a):
        ar_f[i + 1: i + 1 + len_a, i] = a
    for i in range(len_a):
        ar_f[num_full_a + 1 + i:, num_full_a + i] = a[: len_a-i-1]
    return ar_f


def batch_3d_convolve(image, kernel):
    # What the hell is with this 3d convolve thing? Seems to be a lot slower
    # on GPU?
    # convolve_3d_with_kernel = partial(
    #     scipy.signal.convolve, in2=kernel, mode='same'
    # )
    # with mp.Pool() as p:
    #     image_smoothed = p.map(
    #         convolve_3d_with_kernel, np.moveaxis(image, 3, 0)
    #     )
    # convolve_3d_with_kernel = partial(
    #     scipy.signal.convolve, in2=kernel, mode='same'
    # )
    image_smoothed = Parallel(n_jobs=-1)(
        delayed(scipy.signal.convolve)(im, in2=kernel, mode='same')
        for im in np.moveaxis(image, 3, 0)
    )
    return np.stack(image_smoothed, axis=-1)


# def batch_prewhiten_re_estimates(
#     a, regressors, bold_image, error, with_Satterthwaite=True
# ):
#     num_voxel, ar_order = a.shape
#     num_frame, num_regressor = regressors.shape
#     inv_xt_iv_x_out = np.zeros(
#         (num_voxel, num_regressor, num_regressor), dtype=np.float32
#     )
#     b = np.zeros((num_voxel, num_regressor), dtype=np.float32)
#     if with_Satterthwaite:
#         df = np.zeros(num_voxel, dtype=np.float32)
#     else:
#         df =
#     e_var = np.zeros(num_voxel, dtype=np.float32)


@jit(nopython=True, parallel=False, fastmath=True, cache=True)
def batch_prewhiten_re_estimates(a, regressors, bold_image, error):
    # Make sure for numba, parallel=False! parallel=True hurts performance
    # NOTE: cannot seperate this function to multiple steps, inv_v consumes
    # huge RAM
    # NOTE: numba does not like hermitian=True, using numba
    # (whithout hermitian) is slightly faster than no numba with hermitian=True
    num_voxel, ar_order = a.shape
    num_frame, num_regressor = regressors.shape
    inv_xt_iv_x_out = np.zeros(
        (num_voxel, num_regressor, num_regressor), dtype=np.float32
    )
    b = np.zeros((num_voxel, num_regressor), dtype=np.float32)
    sse = np.zeros(num_voxel, dtype=np.float32)
    # This is done in a loop to save memory
    for voxel_i in range(num_voxel):
        # calculate the inv_v matrix from the ar calculation
        inv_v = get_inv_v_from_ar(a[voxel_i], num_frame)
        xt_iv = regressors.T @ inv_v
        # cache inv_xt_iv_x for calculating t map later
        # inv_xt_iv_x = np.linalg.pinv(xt_iv @ regressors, hermitian=True)
        inv_xt_iv_x = np.linalg.pinv(xt_iv @ regressors)
        inv_xt_iv_x_out[voxel_i, :, :] = inv_xt_iv_x
        # re-estimate b with pre-whittened b = inv(x.T @ iv @ X) @ X.T @ iV @ y
        inv_xt_iv_x_xt_iv = inv_xt_iv_x @ xt_iv
        b[voxel_i, :] = inv_xt_iv_x_xt_iv @ bold_image[voxel_i]
        e_i = error[voxel_i]
        sse[voxel_i] = e_i.T @ inv_v @ e_i
    return inv_xt_iv_x_out, b, sse


@jit(nopython=True, parallel=False, fastmath=True, cache=True)
def batch_prewhiten_re_estimates_no_sse(a, regressors, bold_image):
    num_voxel, ar_order = a.shape
    num_frame, num_regressor = regressors.shape
    inv_xt_iv_x_out = np.zeros(
        (num_voxel, num_regressor, num_regressor), dtype=np.float32
    )
    b = np.zeros((num_voxel, num_regressor), dtype=np.float32)
    for voxel_i in range(num_voxel):
        # calculate the inv_v matrix from the ar calculation
        inv_v = get_inv_v_from_ar(a[voxel_i], num_frame)
        xt_iv = regressors.T @ inv_v
        # cache inv_xt_iv_x for calculating t map later
        # inv_xt_iv_x = np.linalg.pinv(xt_iv @ regressors, hermitian=True)
        inv_xt_iv_x = np.linalg.pinv(xt_iv @ regressors)
        inv_xt_iv_x_out[voxel_i, :, :] = inv_xt_iv_x
        # re-estimate b with pre-whittened b = inv(x.T @ iv @ X) @ X.T @ iV @ y
        inv_xt_iv_x_xt_iv = inv_xt_iv_x @ xt_iv
        b[voxel_i, :] = inv_xt_iv_x_xt_iv @ bold_image[voxel_i]
    return inv_xt_iv_x_out, b

# I think this Satterthwaite is wrong?
# the dof calculation is adopted from the canlabcore code, I think the r
# calculation is wrong! should use wy
# @jit(nopython=True, parallel=False, fastmath=True, cache=True)
# def batch_prewhiten_re_estimates_with_Satterthwaite(
#     a, regressors, bold_image, error
# ):
#   # Make sure for numba, parallel=False! parallel=True hurts performance

#   num_voxel, ar_order = a.shape
#   num_frame, num_regressor = regressors.shape
#   inv_xt_iv_x_out = np.zeros(
#       (num_voxel, num_regressor, num_regressor), dtype=np.float32
#   )
#   b = np.zeros((num_voxel, num_regressor), dtype=np.float32)
#   dof = np.zeros(num_voxel, dtype=np.float32)
#   sum_squared_error = np.zeros(num_voxel, dtype=np.float32)
#   # This is done in a loop to save memory
#   for voxel_i in range(num_voxel):
#     ## calculate the inv_v matrix from the ar calculation
#     inv_v = get_inv_v_from_ar(a[voxel_i, :], num_frame)
#     # print(inv_v.shape, regressors.T.shape)
#     xt_iv = regressors.T @ inv_v
#     ## cache inv_xt_iv_x for calculating t map later
#     # inv_xt_iv_x = np.linalg.pinv(xt_iv @ regressors, hermitian=True)
#     inv_xt_iv_x = np.linalg.pinv(xt_iv @ regressors)
#     inv_xt_iv_x_out[voxel_i, :, :] = inv_xt_iv_x
#     ## re-estimate b with pre-whittened b = inv(x.T @ iv @ X) @ X.T @ iV @ y
#     inv_xt_iv_x_xt_iv = inv_xt_iv_x @ xt_iv
#     b[voxel_i, :] = inv_xt_iv_x_xt_iv @ bold_image[voxel_i]
#     ## residual inducing matrix r= I - x @ inv(x.T @ iv @ x) @ x.T @ iV
#     r = np.eye(num_frame, dtype=np.float32) - regressors @ inv_xt_iv_x_xt_iv
#     ## iV == ar_filter @ ar_filter.T
#     ar_filter = get_ar_filter_from_ar(a[voxel_i], num_frame)
#     ## Calculate Satterthwaite approximation for degrees of freedom
#     Wd = r @ ar_filter @ np.linalg.inv(inv_v) @ ar_filter.T
#     dof[voxel_i] = np.power(np.trace(Wd), 2) / np.trace(Wd @ Wd)
#     e_i = error[voxel_i]
#     sum_squared_error[voxel_i] = e_i.T @ inv_v @ e_i
#   e_var = sum_squared_error / dof
#   return inv_xt_iv_x_out, b, e_var, dof


def get_mask_range(mask_image):
    mask_x_size, mask_y_size, mask_z_size = mask_image.shape
    x_indices = np.arange(mask_x_size)
    y_indices = np.arange(mask_y_size)
    z_indices = np.arange(mask_z_size)
    X_mesh, Y_mesh, Z_mesh = np.meshgrid(
        x_indices, y_indices, z_indices, indexing='ij'
    )
    x_index_masked = X_mesh[mask_image]
    y_index_masked = Y_mesh[mask_image]
    z_index_masked = Z_mesh[mask_image]
    mask_x_min, mask_x_max = np.min(x_index_masked), np.max(x_index_masked)
    mask_y_min, mask_y_max = np.min(y_index_masked), np.max(y_index_masked)
    mask_z_min, mask_z_max = np.min(z_index_masked), np.max(z_index_masked)
    return (
        mask_x_min, mask_x_max, mask_y_min, mask_y_max, mask_z_min, mask_z_max
    )


def grow_3d_mask_helper(X_mesh, Y_mesh, Z_mesh, mask_image, grow_len):
    # NOTE: numba hates 3d indexing
    new_mask = np.copy(mask_image)
    mask_x_size, mask_y_size, mask_z_size = mask_image.shape
    x_index_masked = X_mesh[mask_image]
    y_index_masked = Y_mesh[mask_image]
    z_index_masked = Z_mesh[mask_image]

    for i in range(1, grow_len + 1):
        new_x_neg = x_index_masked - i
        new_x_neg[new_x_neg < 0] = 0

        new_x_pos = x_index_masked + i
        new_x_pos[new_x_pos >= mask_x_size] = mask_x_size - 1

        new_y_neg = y_index_masked - i
        new_y_neg[new_y_neg < 0] = 0

        new_y_pos = y_index_masked + i
        new_y_pos[new_y_pos >= mask_y_size] = mask_y_size - 1

        new_z_neg = z_index_masked - i
        new_z_neg[new_z_neg < 0] = 0

        new_z_pos = z_index_masked + i
        new_z_pos[new_z_pos >= mask_z_size] = mask_z_size - 1

        new_mask[new_x_neg, y_index_masked, z_index_masked] = True
        new_mask[new_x_pos, y_index_masked, z_index_masked] = True
        new_mask[x_index_masked, new_y_neg, z_index_masked] = True
        new_mask[x_index_masked, new_y_pos, z_index_masked] = True
        new_mask[x_index_masked, y_index_masked, new_z_neg] = True
        new_mask[x_index_masked, y_index_masked, new_z_pos] = True

        new_mask[new_x_neg, new_y_neg, z_index_masked] = True
        new_mask[new_x_neg, new_y_pos, z_index_masked] = True
        new_mask[new_x_pos, new_y_neg, z_index_masked] = True
        new_mask[new_x_pos, new_y_pos, z_index_masked] = True

        new_mask[new_x_neg, y_index_masked, new_z_neg] = True
        new_mask[new_x_neg, y_index_masked, new_z_pos] = True
        new_mask[new_x_pos, y_index_masked, new_z_neg] = True
        new_mask[new_x_pos, y_index_masked, new_z_pos] = True

        new_mask[x_index_masked, new_y_neg, new_z_neg] = True
        new_mask[x_index_masked, new_y_neg, new_z_pos] = True
        new_mask[x_index_masked, new_y_pos, new_z_neg] = True
        new_mask[x_index_masked, new_y_pos, new_z_pos] = True

        new_mask[new_x_neg, new_y_neg, new_z_neg] = True
        new_mask[new_x_neg, new_y_neg, new_z_pos] = True
        new_mask[new_x_neg, new_y_pos, new_z_neg] = True
        new_mask[new_x_neg, new_y_pos, new_z_pos] = True
        new_mask[new_x_pos, new_y_neg, new_z_neg] = True
        new_mask[new_x_pos, new_y_neg, new_z_pos] = True
        new_mask[new_x_pos, new_y_pos, new_z_neg] = True
        new_mask[new_x_pos, new_y_pos, new_z_pos] = True
    return new_mask


def grow_3d_mask(mask_image, grow_len):
    if (grow_len <= 0):
        return np.copy(mask_image)
    mask_x_size, mask_y_size, mask_z_size = mask_image.shape
    x_indices = np.arange(mask_x_size).astype(np.int32)
    y_indices = np.arange(mask_y_size).astype(np.int32)
    z_indices = np.arange(mask_z_size).astype(np.int32)
    X_mesh, Y_mesh, Z_mesh = np.meshgrid(
        x_indices, y_indices, z_indices, indexing='ij'
    )
    return grow_3d_mask_helper(X_mesh, Y_mesh, Z_mesh, mask_image, grow_len)


# Create smoothing kernel
def create_3d_gaussian_smoothing_kernel(fwhm_mm, bold_resolution_x_y_z_mm):
    # print(fwhm_mm, bold_resolution_x_y_z_mm)
    if isinstance(bold_resolution_x_y_z_mm, numbers.Number):
        bold_resolution_x_y_z_mm = np.array(
            [
                bold_resolution_x_y_z_mm,
                bold_resolution_x_y_z_mm,
                bold_resolution_x_y_z_mm
            ], dtype=np.float32
        )
    else:
        bold_resolution_x_y_z_mm = np.array(
            bold_resolution_x_y_z_mm, dtype=np.float32
        )
    sigma = fwhm_to_sigma(bold_resolution_x_y_z_mm * fwhm_mm)
    half_kernel_len = (fwhm_mm * bold_resolution_x_y_z_mm / 4 * 3).astype(int)
    # print(half_kernel_len)
    x = np.arange(-half_kernel_len[0], half_kernel_len[0] + 1)
    y = np.arange(-half_kernel_len[1], half_kernel_len[1] + 1)
    z = np.arange(-half_kernel_len[2], half_kernel_len[2] + 1)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    pos = np.stack([X, Y, Z], axis=-1)
    image_smoothing_kernel = scipy.stats.multivariate_normal.pdf(
        pos, cov=sigma * np.eye(3)
    ).astype(np.float32)
    return image_smoothing_kernel


def filter_bold_image(
    bold_image, mask_image=None, expanded_mask=None,
    expanded_mask_image_range=None, mask_threshold=0,
    image_smoothing_kernel=None
):
    # TODO: Add more code for input checks
    # check bold_image size matches regresors matches mask_image
    # TODO: implement robust iterative reweighted least-square
    # TODO: abstract out the mask calculation and expanded mask calculation so
    # they can be parameters that get passed in
    if mask_image is None:
        print(
            'You sure you don\'t want a mask???? Please just supply a mask to'
            ' save calculation, a brain mask for your template will do.'
        )
        print(
            'Unless you have a very low resolution or you pre-cropped the '
            'image or you have huge RAM (like 128gb+ huge), you will run out '
            'of RAM! Just kill the process now!'
        )
        mask_image = np.alltrue(bold_image > mask_threshold, axis=-1)
        expanded_mask = None
        expanded_mask_image_range = None
    if mask_image.dtype != 'bool':
        raise Exception(
            'Something wrong with the mask_image, '
            'expect it to be a type bool image'
        )

    # Since the edge of a convolution is is bad, use an expanded mask so the
    # edge of the real mask is well defined
    if expanded_mask is None:
        # Calculate the half kernel length of the smoothing kernels to expand
        # the range of the mask
        kernel_len = 0
        if image_smoothing_kernel is not None:
            image_smoothing_kernel_len = max(image_smoothing_kernel.shape)
            if image_smoothing_kernel_len % 2 == 0:
                raise Exception(
                    'Use an image_smoothing_kernel_len with odd number of '
                    'length so it has a center'
                )
            if not np.alltrue(
                image_smoothing_kernel_len == np.array(
                    image_smoothing_kernel.shape, dtype=int
                )
            ):
                raise Exception(
                    'Invalid image_smoothing_kernel. '
                    'Make sure the 3 dimensions are the same'
                )
            kernel_len = image_smoothing_kernel_len
        if (ar_parameters is None) and (ar_smoothing_kernel is not None):
            ar_smoothing_kernel_len = max(ar_smoothing_kernel.shape)
            if ar_smoothing_kernel_len % 2 == 0:
                raise Exception(
                    'Use an ar_smoothing_kernel_len with odd number of '
                    'length so it has a center'
                )
            if not np.alltrue(
                ar_smoothing_kernel_len == np.array(
                    ar_smoothing_kernel.shape, dtype=int
                )
            ):
                raise Exception(
                    'Invalid ar_smoothing_kernel. '
                    'Make sure the 3 dimensions are the same'
                )
            if ar_smoothing_kernel_len > kernel_len:
                kernel_len = ar_smoothing_kernel_len
        half_kernel_len = int(kernel_len / 2)

        # grow the mask with a range of half kernel len so the smoothing
        # convolution can behave nicely at the edge of your region of interest
        expanded_mask = grow_3d_mask(mask_image, half_kernel_len)

    # get the new mask x, y, z range
    if expanded_mask_image_range is not None:
        mask_x_min, mask_x_max, mask_y_min, mask_y_max, mask_z_min, mask_z_max\
         = expanded_mask_image_range
    else:
        mask_x_min, mask_x_max, mask_y_min, mask_y_max, mask_z_min, mask_z_max\
         = get_mask_range(expanded_mask)

    # Shrink the bold_image and mask_image to only the region of interests
    compact_bold = bold_image[
        mask_x_min: mask_x_max + 1,
        mask_y_min: mask_y_max + 1,
        mask_z_min: mask_z_max + 1,
        :
    ]
    compact_expanded_mask = expanded_mask[
        mask_x_min: mask_x_max + 1,
        mask_y_min: mask_y_max + 1,
        mask_z_min: mask_z_max + 1
    ]
    compact_mask = mask_image[
        mask_x_min: mask_x_max + 1,
        mask_y_min: mask_y_max + 1,
        mask_z_min: mask_z_max + 1
    ]

    x_size, y_size, z_size, num_frame = compact_bold.shape

    num_voxel = x_size * y_size * z_size

    # apply smoothing to bold image
    if image_smoothing_kernel is not None:
        compact_bold = batch_3d_convolve(compact_bold, image_smoothing_kernel)

    return compact_bold[compact_mask]

# ar_parameters overrides ar_smoothing_kernel and ar_order
# when ar_parameters is supplied, ar_order and ar_smoothing_kernel is ignored
# do initial estimation with ar_parameters=None to get initial estimation of ar
# Then say, average ar across several runs, and pass the results in for an
# estimation with the averaged ar_parameters


def glm_with_ar(
    bold_image, regressors, mask_image=None, expanded_mask=None,
    expanded_mask_image_range=None, mask_threshold=0,
    image_smoothing_kernel=None, ar_smoothing_kernel=None, ar_order=4,
    ar_parameters=None, estimate_ar_without_whitening=False, use_ar_sse=False
):
    # TODO: Add more code for input checks
    # check bold_image size matches regresors matches mask_image
    # TODO: implement robust iterative reweighted least-square
    # TODO: abstract out the mask calculation and expanded mask calculation so
    # they can be parameters that get passed in
    if mask_image is None:
        print(
            'You sure you don\'t want a mask???? Please just supply a mask '
            'to save calculation, a brain mask for your template will do.'
        )
        print(
            'Unless you have a very low resolution or you pre-cropped the '
            'image or you have huge RAM (like 128gb+ huge), you will run '
            'out of RAM! Just kill the process now!'
        )
        mask_image = np.alltrue(bold_image > mask_threshold, axis=-1)
        expanded_mask = None
        expanded_mask_image_range = None
    if mask_image.dtype != 'bool':
        raise Exception(
            'Something wrong with the mask_image, '
            'expect it to be a type bool image'
        )
    if ar_parameters is None:
        if ar_order < 0:
            raise Exception('Cannot do a negative ar_order')
        ar_order = int(ar_order)
        if ar_order < 3:
            print(
                'You probably want to use a higher ar_oder unless the '
                'scanner is an ancient potato with very low TR'
            )
    else:
        ar_oder = ar_parameters.shape[1]
        ar_smoothing_kernel = None
        # If ar_parameters is supplied, by default don't do ar parameter
        # spatial smoothing

    # Since the edge of a convolution is is bad, use an expanded mask so the
    # edge of the real mask is well defined
    if expanded_mask is None:
        # Calculate the half kernel length of the smoothing kernels to expand
        # the range of the mask
        kernel_len = 0
        if image_smoothing_kernel is not None:
            image_smoothing_kernel_len = max(image_smoothing_kernel.shape)
            if image_smoothing_kernel_len % 2 == 0:
                raise Exception(
                    'Use an image_smoothing_kernel_len with odd number of '
                    'length so it has a center'
                )
            if not np.alltrue(
                image_smoothing_kernel_len == np.array(
                    image_smoothing_kernel.shape, dtype=int
                )
            ):
                raise Exception(
                    'Invalid image_smoothing_kernel. '
                    'Make sure the 3 dimensions are the same'
                )
            kernel_len = image_smoothing_kernel_len
        if (ar_parameters is None) and (ar_smoothing_kernel is not None):
            ar_smoothing_kernel_len = max(ar_smoothing_kernel.shape)
            if ar_smoothing_kernel_len % 2 == 0:
                raise Exception(
                    'Use an ar_smoothing_kernel_len with odd number of '
                    'length so it has a center'
                )
            if not np.alltrue(
                ar_smoothing_kernel_len == np.array(
                    ar_smoothing_kernel.shape, dtype=int
                )
            ):
                raise Exception(
                    'Invalid ar_smoothing_kernel. '
                    'Make sure the 3 dimensions are the same'
                )
            if ar_smoothing_kernel_len > kernel_len:
                kernel_len = ar_smoothing_kernel_len
        half_kernel_len = int(kernel_len / 2)

        # grow the mask with a range of half kernel len so the smoothing
        # convolution can behave nicely at the edge of your region of interest
        expanded_mask = grow_3d_mask(mask_image, half_kernel_len)

    # get the new mask x, y, z range
    if expanded_mask_image_range is not None:
        mask_x_min, mask_x_max, mask_y_min, mask_y_max, mask_z_min, mask_z_max\
            = expanded_mask_image_range
    else:
        mask_x_min, mask_x_max, mask_y_min, mask_y_max, mask_z_min, mask_z_max\
            = get_mask_range(expanded_mask)

    # Shrink the bold_image and mask_image to only the region of interests
    compact_bold = bold_image[
        mask_x_min: mask_x_max + 1,
        mask_y_min: mask_y_max + 1,
        mask_z_min: mask_z_max + 1, :
    ]
    compact_expanded_mask = expanded_mask[
        mask_x_min: mask_x_max + 1,
        mask_y_min: mask_y_max + 1,
        mask_z_min: mask_z_max + 1
    ]
    compact_mask = mask_image[
        mask_x_min: mask_x_max + 1,
        mask_y_min: mask_y_max + 1,
        mask_z_min: mask_z_max + 1
    ]

    x_size, y_size, z_size, num_frame = compact_bold.shape
    num_regressor = regressors.shape[1]
    num_voxel = x_size * y_size * z_size

    # apply smoothing to bold image
    if image_smoothing_kernel is not None:
        compact_bold = batch_3d_convolve(compact_bold, image_smoothing_kernel)

    # For values you expect to do filter on, use compact_expanded_mask,
    # for results in mask you actually care, use compact_mask
    compact_bold = compact_bold.reshape((num_voxel, num_frame))
    compact_expanded_mask = compact_expanded_mask.reshape((num_voxel,))
    compact_mask = compact_mask.reshape((num_voxel,))
    mask_in_expanded_mask = compact_mask[compact_expanded_mask]
    # print(
    #     compact_bold.shape, compact_expanded_mask.shape, compact_mask.shape,
    #     mask_in_expanded_mask.shape, np.sum(compact_expanded_mask),
    #     np.sum(compact_mask), np.sum(mask_in_expanded_mask)
    # )

    # cache reusable calculation results
    inv_xtx = np.linalg.pinv(regressors.T @ regressors)
    inv_xtx_xt = inv_xtx @ regressors.T

    # For the actual calculation, only calculated parameters within the
    # compact  mask only need to calculated values in the compact_expanded_mask
    # range when ar parameter smoothing is required
    if ((ar_order == 0) and (ar_parameters is None)) or \
            (ar_parameters is not None) or (ar_smoothing_kernel is None):
        mask_to_use = compact_mask
    else:
        mask_to_use = compact_expanded_mask

    # first round of glm y = x @ b + e, b = pinv(x.T @ x) @ x.T @ y
    b = matmul_1d_mat1_batch_1d_mat2(inv_xtx_xt, compact_bold[mask_to_use, :])
    # print(b.shape, b.dtype)

    # calculate the fit which is y_fit = x @ b
    y_fit = matmul_1d_mat1_batch_1d_mat2(regressors, b)
    # print(y_fit.shape, y_fit.dtype)

    # num_regressor = regressors.shape[1]
    # calculate the error of the fit: e
    e = compact_bold[mask_to_use, :] - y_fit

    if (ar_order == 0) and (ar_parameters is None):
        print(
            'Why would you even do AR=0? Be a good person and model '
            'the error autocorrelation!'
        )
        # This is assuming inv(V) is I, so the variance for each sample is
        # assumed to be the same
        sse = e.T @ e
        # return inv_xtx_xt, b, sse
        return np.stack([inv_xtx] * time_series.shape[-1]), b, None, sse
    else:  # ar_order is positive and integer or ar_parameters are supplied
        if ar_parameters is not None:  # if ar_parameters is supplied, use if
            a = ar_parameters
        else:  # estimate AR parameter
            # calculate xcorr of the error with order = ar_order
            xcorr = batch_xcorr(e, ar_order)
            a = np.zeros((xcorr.shape[0], ar_order), np.float32)
            xcorr_not_zero_mask = np.all(xcorr != 0, axis=1)
            a[xcorr_not_zero_mask, :] = batch_AR_from_xcorr_using_Yule_Walker(
                xcorr[xcorr_not_zero_mask, :]
            )
            # If ar_smoothing_kernel supplied, do the smoothing
            if ar_smoothing_kernel is not None:
                temp_a = np.zeros((num_voxel, ar_order), dtype=np.float32)
                temp_a[compact_expanded_mask, :] = a
                temp_a = temp_a.reshape((x_size, y_size, z_size, ar_order))
                a = batch_3d_convolve(temp_a, ar_smoothing_kernel).reshape(
                    (num_voxel, ar_order)
                )[compact_expanded_mask, :]
                a = a[mask_in_expanded_mask, :]
                e = e[mask_in_expanded_mask]
                xcorr = xcorr[mask_in_expanded_mask, :]
            if estimate_ar_without_whitening:
                return a

        if use_ar_sse:
            # This is an alternative to the 'proper' way. Saves computation,
            # but is it the right way? This seem to inflate the sse by a lot?
            sse = batch_sse_from_xcorr_and_ar(xcorr, a)
            inv_xt_iv_x, b = batch_prewhiten_re_estimates_no_sse(
                a, regressors, compact_bold[compact_mask, :]
            )
        else:
            inv_xt_iv_x, b, sse = batch_prewhiten_re_estimates(
                a, regressors, compact_bold[compact_mask, :], e
            )

        return inv_xt_iv_x, b, a, sse


def prep_bold_info(bold_info):
    prepped_bold_info = {}
    prepped_bold_info['event_tsv_content'] = load_xsv(bold_info['event_path'])
    prepped_bold_info['event_by_type'] = event_dict_list_to_event_by_type(
        prepped_bold_info['event_tsv_content']
    )
    prepped_bold_info['onset_array_by_type'] = {
        e_type: event_list_to_onset_array(
            events, bold_info['run_len_s'], bold_info['resolution_s']
        ) for e_type, events in prepped_bold_info['event_by_type'].items()
    }
    prepped_bold_info['regressors_by_type'] = {
        e_type: onset_array_convolve_with_basis(
            onset_array, bold_info['basis'], bold_info['TR_s'],
            bold_info['resolution_s']
        ) for e_type, onset_array in
        prepped_bold_info['onset_array_by_type'].items()
    }
    regressor_types = sorted(
        list(prepped_bold_info['regressors_by_type'].keys())
    )
    prepped_bold_info['regressor_types'] = regressor_types
    # all_stimuli_regressors = np.hstack([regressors_by_type for e_type,
    # regressors_by_type in prepped_bold_info['regressors_by_type'].items()])
    all_stimuli_regressors = np.hstack(
        [
            prepped_bold_info['regressors_by_type'][regressor_type]
            for regressor_type in regressor_types
        ]
    )
    timeseries_tsv_content = load_xsv(bold_info['timeseries_tsv_path'])
    prepped_bold_info['timeseries_tsv_raw'] = load_file(
        bold_info['timeseries_tsv_path']
    )
    confound_regressors = extract_confound_regressors(timeseries_tsv_content)
    motion_outliers = extract_confound_regressors(
        timeseries_tsv_content, USE_MOTION=False,
        USE_MOTION_DERIVATIVE1=False,  USE_MOTION_POW2=False,
        USE_MOTION_DERIVATIVE1_POW2=False, USE_CSF=False,
        USE_WM=False, USE_MOTION_OUTLIER=True
    ).astype(dtype=bool)
    prepped_bold_info['motion_outlier_index'] = motion_outliers.nonzero()[0]
    prepped_bold_info['all_regressors'] = np.hstack((
        all_stimuli_regressors,
        bold_info['dct_regressors'],
        confound_regressors
    ))
    return prepped_bold_info


def get_available_ram():
    mem_info = load_file('/proc/meminfo')
    mem_available = int(mem_info.split('MemAvailable:')[1].split('kB')[0])
    return mem_available


def get_cb_var(e_var, c_inv_xt_iv_x_ct):
    cb_var = (c_inv_xt_iv_x_ct.T * e_var).T
    return cb_var


def get_t_from_parts(cb, e_var, c_inv_xt_iv_x_ct):
    cb_var = get_cb_var(e_var, c_inv_xt_iv_x_ct)
    std_cb = np.sqrt(cb_var)
    # e_var_c_inv_xt_iv_x_ct = (c_inv_xt_iv_x_ct.T * e_var).T
    # sqrt_e_var_c_inv_xt_iv_x_ct = np.sqrt(e_var_c_inv_xt_iv_x_ct)
    return cb / std_cb


def get_c_inv_xt_iv_x_ct(c, inv_xt_iv_x):
    c_inv_xt_iv_x = matmul_1d_mat1_batch_2d_mat2(c, inv_xt_iv_x)
    c_inv_xt_iv_x_ct = matdot_2d_batch_1d_mat1_1d_mat2(c_inv_xt_iv_x, c.T)
    return c_inv_xt_iv_x_ct


def get_cb_with_derivative_boost(b, c, derivative_boost_masks):
    num_voxel = b.shape[0]
    num_contrast = c.shape[0]
    out = np.zeros((num_voxel, num_contrast), dtype=np.float32)
    for i, c_row, derivative_boost_mask in zip(
        np.arange(len(derivative_boost_masks)), c, derivative_boost_masks
    ):
        num_mask, num_relevant_regressor_col = derivative_boost_mask.shape
        derivative_boost_mask_padded = np.zeros(
            (num_mask, len(c_row)), dtype=bool
        )
        derivative_boost_mask_padded[:, :num_relevant_regressor_col] = \
            derivative_boost_mask
        for j in range(derivative_boost_mask.shape[0]):
            relevant_b = b[:, derivative_boost_mask_padded[j, :]]
            relevant_c = c_row[derivative_boost_mask_padded[j, :]]
            sign = np.sign(relevant_b[:, 0])
            b_derivative_boost = sign * np.sqrt(
                np.sum(relevant_b * relevant_b, axis=1)
            ) * np.sum(relevant_c)
            out[:, i] += b_derivative_boost
    return out


def pad_contrast(num_regressor, c):
    num_contrast, num_relevant_regressor = c.shape
    c_padded = np.zeros((num_contrast, num_regressor), dtype=np.float32)
    c_padded[:, :num_relevant_regressor] = c
    return c_padded


def get_t_map(e_var, inv_xt_iv_x, b, c, derivative_boost_masks=None):
    num_voxel, num_regressor = b.shape
    c_padded = pad_contrast(num_regressor, c)
    c_inv_xt_iv_x_ct = get_c_inv_xt_iv_x_ct(c_padded, inv_xt_iv_x)
    if derivative_boost_masks is None:
        cb = matmul_1d_mat1_batch_1d_mat2(c_padded, b)
    else:
        cb = get_cb_with_derivative_boost(b, c_padded, derivative_boost_masks)
    return get_t_from_parts(cb, e_var, c_inv_xt_iv_x_ct)


def get_t_map_without_derivative_boost(e_var, inv_xt_iv_x, b, c):
    cb = matmul_1d_mat1_batch_1d_mat2(c, b)
    c_inv_xt_iv_x_ct = get_c_inv_xt_iv_x_ct(c, inv_xt_iv_x)
    e_var_c_inv_xt_iv_x_ct = (c_inv_xt_iv_x_ct.T * e_var).T
    sqrt_e_var_c_inv_xt_iv_x_ct = np.sqrt(e_var_c_inv_xt_iv_x_ct)
    t = cb / sqrt_e_var_c_inv_xt_iv_x_ct
    return t


def prepare_bold_info(
    input_bold_path, input_event_path, atlas='HOCPAL', desc='th0',
    image_smoothing_fwhm_mm=2
):
    bold_image_handle = nib.load(input_bold_path)
    # Infer from the file name to get the templateflow template used by
    # fmriperep.
    bold_file_info = extract_bold_file_info_from_name(input_bold_path)
    bold_image_resolution_mm = bold_image_handle.header['pixdim'][1:4]
    TR_s = bold_image_handle.header['pixdim'][4]
    num_frames = bold_image_handle.header['dim'][4]
    run_len_s = num_frames * TR_s
    # regions = [7,17]# 7 Primary motor cortex, 17 somatosensory cortex
    regions = [
        {'region_name': 'left_mask_image', 'vals': [13]},
        {'region_name': 'right_mask_image', 'vals': [14]},
        {'region_name': 'mask_image', 'vals': [13, 14]}
    ]
    atlas_image_path = templateflow.api.get(
        template=bold_file_info['space'],
        resolution=bold_image_resolution_mm[0], atlas=atlas, desc=desc
    )
    atlas_image_handle = nib.load(atlas_image_path)
    atlas_image = atlas_image_handle.get_fdata(dtype=np.float32).astype(int)
    mask_image_dictionary = {
        region['region_name']: reduce(
            np.logical_or, [atlas_image == val for val in region['vals']]
        ) for region in regions
    }

    full_brain_mask_image_handle = nib.load(
        bold_file_info['brain_mask_file_path']
    )
    full_brain_mask_image = full_brain_mask_image_handle.get_fdata(
        dtype=np.float32
    ).astype(bool)
    mask_image_dictionary = {
        key: np.logical_and(mask, full_brain_mask_image)
        for key, mask in mask_image_dictionary.items()
    }

    mask_image = mask_image_dictionary['mask_image']

    # image_smoothing_kernel = None
    if image_smoothing_fwhm_mm > 0:
        image_smoothing_kernel = create_3d_gaussian_smoothing_kernel(
            image_smoothing_fwhm_mm, bold_image_resolution_mm
        )
    else:
        image_smoothing_kernel = None
    # According to the paper below, the AFNI's ARMA(1) no smoothing works best.
    # https://www.nature.com/articles/s41467-019-09230-w
    # Here, I am defaulting to AR(4) without smoothing
    ar_smoothing_kernel = None
    if (image_smoothing_kernel is None) and (ar_smoothing_kernel is None):
        expanded_mask = None
        expanded_mask_image_range = None
    else:
        if (image_smoothing_kernel is not None) and \
                (ar_smoothing_kernel is not None):
            image_smoothing_kernel_len = max(
                image_smoothing_kernel.shape + ar_smoothing_kernel.shape
            )
        elif image_smoothing_kernel is not None:
            image_smoothing_kernel_len = max(image_smoothing_kernel.shape)
        else:
            image_smoothing_kernel_len = max(ar_smoothing_kernel.shape)
        half_kernel_len = int(image_smoothing_kernel_len / 2)
        expanded_mask = grow_3d_mask(mask_image, half_kernel_len)
        expanded_mask_image_range = get_mask_range(expanded_mask)

    # Generate basis to use for all events
    # use a resoliution higher but devisable by TR_s and stimulus time
    resolution_s = 0.01
    bold, time_stamp = spm_hrf(resolution_s)
    dbold = spm_d_hrf(resolution_s)
    ddbold = spm_dd_hrf(resolution_s)
    bold_max = np.max(bold)
    bold, dbold, ddbold = bold/bold_max, dbold/bold_max, ddbold/bold_max
    basis = np.vstack((bold, dbold, ddbold)).T

    # Drift regressors
    frequency_cut_hz = 1 / 128  # SPM default
    TR_timestamp = np.arange(num_frames) * TR_s
    dct_regressors = dct_drift_basis(TR_timestamp, frequency_cut_hz)

    # load stimuli of the first run to use as a reference
    # sample_bold_stimuli_path = input_event_paths[0]
    # stimuli_events = load_xsv(sample_bold_stimuli_path)
    stimuli_events = load_xsv(input_event_path)
    letter_event_list = event_dict_list_to_letter_event_list(stimuli_events)
    event_by_type = event_dict_list_to_event_by_type(stimuli_events)

    # Generate regressors to use for all events
    onset_array_by_type = {
        e_type: event_list_to_onset_array(events, run_len_s, resolution_s)
        for e_type, events in event_by_type.items()
    }
    regressors_by_type = {
        e_type: onset_array_convolve_with_basis(
            onset_array, basis, TR_s, resolution_s
        ) for e_type, onset_array in onset_array_by_type.items()}
    regressor_names = []
    for e_type, regressors in regressors_by_type.items():
        for i in range(regressors.shape[1]):
            regressor_names.append(f'{e_type}_{i}')
    all_stimuli_regressors = np.hstack(list(regressors_by_type.values()))
    all_stimuli_regressors_type = list(regressors_by_type.keys())

    # Create the contrast matrix
    num_regressors_and_reg_sum = []
    num_regressor_sum = 0
    for e_type, regressors in regressors_by_type.items():
        num_regressors_and_reg_sum.append(
            (regressors.shape[1], num_regressor_sum)
        )
        num_regressor_sum += regressors.shape[1]
    total_num_regressors = all_stimuli_regressors.shape[1]

    contrasts = []
    # contrast matrix - stimulus vs background
    for num_regressors, reg_sum in num_regressors_and_reg_sum:
        contrast = np.zeros((1, total_num_regressors))
        contrast[0, reg_sum: reg_sum + num_regressors] = 1 / num_regressors
        contrasts.append(contrast)
    # contrast matrix - combinations of 2 stimulus contrast
    for r_1, r_2 in itertools.combinations(num_regressors_and_reg_sum, 2):
        contrast = np.zeros((1, total_num_regressors))
        contrast[0, r_1[1]:r_1[1]+r_1[0]] = 1 / r_1[0]
        contrast[0, r_2[1]:r_2[1]+r_2[0]] = -1 / r_2[0]
        contrasts.append(contrast)
    contrasts = np.vstack(contrasts)

    derivative_boost_masks = []
    for contrast in contrasts:
        derivative_boost_mask_pos = contrast > 0
        derivative_boost_mask_neg = contrast < 0
        if np.any(derivative_boost_mask_neg):
            derivative_boost_mask = np.stack(
                (derivative_boost_mask_pos, derivative_boost_mask_neg)
            )
        else:
            derivative_boost_mask = derivative_boost_mask_pos[np.newaxis, :]
        derivative_boost_masks.append(derivative_boost_mask)

    bold_info = extract_bold_file_info_from_name(input_bold_path)
    bold_info['event_path'] = input_event_path
    bold_info['run_len_s'] = run_len_s
    bold_info['resolution_s'] = resolution_s
    bold_info['basis'] = basis
    bold_info['TR_s'] = TR_s
    bold_info['dct_regressors'] = dct_regressors
    bold_info['full_brain_mask_image'] = full_brain_mask_image
    bold_info['mask_image'] = mask_image
    bold_info['expanded_mask'] = expanded_mask
    bold_info['expanded_mask_image_range'] = expanded_mask_image_range
    bold_info['image_smoothing_kernel'] = image_smoothing_kernel
    bold_info['ar_smoothing_kernel'] = ar_smoothing_kernel
    bold_info['contrasts'] = contrasts
    bold_info['derivative_boost_masks'] = derivative_boost_masks
    bold_info['mask_image_dictionary'] = mask_image_dictionary
    bold_info['calibration_mask_dictionary'] = {
        'b': 'mask_image',
        'f_l': 'mask_image',
        'f_r': 'mask_image',
        # 'h_l': 'right_mask_image',
        # 'h_r': 'left_mask_image',
        'h_l': 'mask_image',
        'h_r': 'mask_image',
        't': 'mask_image'
    }
    processed_bold_info = prep_bold_info(bold_info)
    print(
        f'Finished processing:\nBOLD path: {input_bold_path}\n'
        f'Event path: {input_event_path}\n-------------------------'
    )
    return bold_info, processed_bold_info


def generate_design_matrix(num_frames, stimulus_timestamps, sf_Hz):
    # use a resoliution higher but devisable by TR_s and stimulus time
    resolution_s = 0.01
    bold, time_stamp = spm_hrf(resolution_s)
    dbold = spm_d_hrf(resolution_s)
    ddbold = spm_dd_hrf(resolution_s)
    bold_max = np.max(bold)
    bold, dbold, ddbold = bold/bold_max, dbold/bold_max, ddbold/bold_max
    basis = np.vstack((bold, dbold, ddbold)).T
    TR_s = 1 / sf_Hz

    # Drift regressors
    frequency_cut_hz = 1 / 128  # SPM default
    TR_timestamp = np.arange(num_frames) * TR_s
    dct_regressors = dct_drift_basis(TR_timestamp, frequency_cut_hz)
    print(f'dct_regressors: {dct_regressors.shape}')

    # Design matrix
    high_res_num_frames = int(np.ceil(num_frames * TR_s / resolution_s))
    onset_matrix_high_res = np.zeros(
        (high_res_num_frames, len(stimulus_timestamps))
    )
    design_matrix = np.zeros((num_frames, 3 * len(stimulus_timestamps)))
    for s_i, stimulus in enumerate(stimulus_timestamps):
        for e_j, event_time in enumerate(stimulus):
            onset_matrix_high_res[
                int(np.ceil(event_time / resolution_s)), s_i
            ] = 1
        design_matrix[:, s_i * 3: s_i * 3 + 3] = \
            onset_array_convolve_with_basis(
                onset_matrix_high_res[:, s_i], basis, TR_s, resolution_s
            )[0:num_frames, :]

    return TR_timestamp, design_matrix, dct_regressors


def glm_with_ar_simplified(
    time_series, regressors, ar_order=20, use_ar_sse=False
):
    inv_xtx = np.linalg.pinv(regressors.T @ regressors)
    inv_xtx_xt = inv_xtx @ regressors.T
    b = matmul_1d_mat1_batch_1d_mat2(inv_xtx_xt, time_series.T)
    y_fit = matmul_1d_mat1_batch_1d_mat2(regressors, b)
    e = time_series.T - y_fit

    if (ar_order == 0):
        print(
            'Why would you even do AR=0? '
            'Be a good person and model the error autocorrelation!'
        )
        sse = e.T @ e
        return np.stack([inv_xtx] * time_series.shape[-1]), b, None, sse
    else:
        xcorr = batch_xcorr(e, ar_order)
        a = np.zeros((xcorr.shape[0], ar_order), np.float32)
        xcorr_not_zero_mask = np.all(xcorr != 0, axis=1)
        a[xcorr_not_zero_mask, :] = batch_AR_from_xcorr_using_Yule_Walker(
            xcorr[xcorr_not_zero_mask, :]
        )
        if use_ar_sse:
            # This is an alternative to the 'proper' way. Saves computation,
            # but is it the right way? This seem to inflate the sse by a lot?
            sse = batch_sse_from_xcorr_and_ar(xcorr, a)
            inv_xt_iv_x, b = batch_prewhiten_re_estimates_no_sse(
                a, regressors, time_series.T
            )
        else:
            inv_xt_iv_x, b, sse = batch_prewhiten_re_estimates(
                a, regressors, time_series.T, e
            )

    return inv_xt_iv_x, b, a, sse


def generate_design_matrix(num_frames, stimulus_timestamps, sf_Hz):
    # use a resoliution higher but devisable by TR_s and stimulus time
    resolution_s = 0.01
    bold, time_stamp = spm_hrf(resolution_s)
    dbold = spm_d_hrf(resolution_s)
    ddbold = spm_dd_hrf(resolution_s)
    bold_max = np.max(bold)
    bold, dbold, ddbold = bold/bold_max, dbold/bold_max, ddbold/bold_max
    basis = np.vstack((bold, dbold, ddbold)).T
    TR_s = 1 / sf_Hz

    # Drift regressors
    frequency_cut_hz = 1 / 128  # SPM default
    TR_timestamp = np.arange(num_frames) * TR_s
    dct_regressors = dct_drift_basis(TR_timestamp, frequency_cut_hz)
    # print(f'dct_regressors: {dct_regressors.shape}')

    # Design matrix
    high_res_num_frames = int(np.ceil(num_frames * TR_s / resolution_s))
    onset_matrix_high_res = np.zeros(
        (high_res_num_frames, len(stimulus_timestamps))
    )
    design_matrix = np.zeros((num_frames, 3 * len(stimulus_timestamps)))
    for s_i, stimulus in enumerate(stimulus_timestamps):
        for e_j, event_time in enumerate(stimulus):
            onset_matrix_high_res[
                int(np.ceil(event_time / resolution_s)), s_i
            ] = 1
        design_matrix[:, s_i * 3: s_i * 3 + 3] = \
            onset_array_convolve_with_basis(
                onset_matrix_high_res[:, s_i], basis, TR_s, resolution_s
            )[0:num_frames, :]

    return (
        TR_timestamp, design_matrix.astype(np.float32),
        dct_regressors.astype(np.float32)
    )
