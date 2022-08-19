import os
import csv
import json
import numpy as np
from scipy.signal import butter


def write_file(file_content, file_path, flag='w+'):
    with open(file_path, flag) as fp:
        fp.write(file_content)


def load_file(file_path, flag='r'):
    with open(file_path, flag) as fp:
        return fp.read()


def delete_file_if_exists(file_name):
    if os.path.exists(file_name):
        os.remove(file_name)


def load_json(json_file_path):
    with open(json_file_path) as fp:
        return json.load(fp)


def write_json(json_data, json_file_path, pretty=True):
    with open(json_file_path, 'w+') as fp:
        if pretty:
            fp.write(json.dumps(json_data, indent=2))
        else:
            fp.write(json.dumps(json_data))


def create_dir_if_not_exist(path, full_path=False):
    if not os.path.isdir(path):
        if full_path:
            os.makedirs(path)
        else:
            os.mkdir(path)


def delete_dir_if_exist(path):
    if os.path.isdir(path):
        shutil.rmtree(path)


def create_symlink(source, link_name):
    if os.path.islink(link_name):
        delete_file_if_exists(link_name)
    elif (os.path.exists(link_name)):
        raise ValueError(f'link_name: {link_name} is a file that exists')
    os.symlink(source, link_name)


# def load_tsv(tsv_file_path):
#     return [
#         dict(i) for i in
#         list(csv.DictReader(open(tsv_file_path), dialect='excel-tab'))
#     ]


def load_xsv(xsv_file_path, dialect='excel-tab'):
    # TODO: DictReader returns[{'key1':v1,'key2':v2,...},{'key':v1,...},...]
    # which seems quite very wasteful?
    # Maybe change it to {'key1':[val1,val2,...],'key2':[val1,val2,...],...}
    return [
        dict(i) for i in
        list(csv.DictReader(open(xsv_file_path), dialect=dialect))
    ]


def load_xsv_from_raw(raw_file_content, dialect='excel-tab'):
    return [
        dict(i) for i in
        list(csv.DictReader(raw_file_content, dialect=dialect))
    ]


def export_dict_list_to_xsv(file_path, dict_list, delimiter='\t'):
    columns = list(dict_list[0].keys())
    delete_file_if_exists(file_path)
    content = delimiter.join(columns) + '\n' + \
        '\n'.join([delimiter.join(
            [str(entry[col]) for col in columns]) for entry in dict_list]
        )
    with open(file_path, 'w+') as fp:
        fp.write(content)


def butterworth_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], analog=False, btype='band', output='sos')
    return sos


def butterworth_highpass(lowcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    sos = butter(order, low, analog=False, btype='highpass', output='sos')
    return sos


def butterworth_lowpass(highcut, fs, order=4):
    nyq = 0.5 * fs
    high = highcut / nyq
    sos = butter(order, high, analog=False, btype='lowpass', output='sos')
    return sos


def lognormalize(x, x_base=10):
    scale = np.log(x_base) / np.log(np.e)
    scaled_x = scale * x
    a = np.logaddexp.reduce(scaled_x)
    return np.exp(scaled_x - a)


def normalize_confusion_matrix(cm):
    return cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
