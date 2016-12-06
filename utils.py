import logging
import os
import pickle

import matplotlib.pyplot as plt

BASE_DATA_DIR = 'data'  # for experiments
DATA_DIR = 'data'
FIGURES_DIR = 'figures'
DPI = 120


def set_data_dir(dir):
    global DATA_DIR
    DATA_DIR = BASE_DATA_DIR + "/" + dir
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)


def get_latest_data_subdir(pattern=None, take=-1):
    def get_date(f):
        return os.stat(os.path.join(BASE_DATA_DIR, f)).st_mtime

    try:
        dirs = next(os.walk(BASE_DATA_DIR))[1]
    except StopIteration:
        return None

    if pattern is not None:
        dirs = (d for d in dirs if pattern in d)

    dirs = list(sorted(dirs, key=get_date))
    if len(dirs) == 0:
        return None

    return dirs[take]


def set_latest_data_subdir(pattern=None, take=-1):
    set_data_dir(get_latest_data_subdir(pattern=pattern, take=take))


def save_pickle(filename, save):
    try:
        f = open(filename, 'wb')
        pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
        f.close()
    except Exception as e:
        logging.error(f"Unable to save data to {filename}: {e}")
        raise


def serialize(name, item):
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    save = {name: item}
    save_pickle(DATA_DIR + '/' + name + '.pickle', save)


def deserialize(name):
    with open(DATA_DIR + '/' + name + '.pickle', 'rb') as f:
        save = pickle.load(f)
        item = save[name]
        del save

    return item


def deserialize_dict(name):
    with open(DATA_DIR + '/' + name + '.pickle', 'rb') as f:
        save = pickle.load(f)

    return save


def save_fig(file_name, clear=True):
    if not os.path.exists(FIGURES_DIR):
        os.makedirs(FIGURES_DIR)

    if file_name is not None:
        plt.savefig(FIGURES_DIR + '/' + file_name + '.png', dpi=DPI)

        if "--timestamp" in file_name:
            dir = FIGURES_DIR + "/final"
            if not os.path.exists(dir):
                os.makedirs(dir)

            file_name = file_name[:file_name.find("--timestamp")]
            plt.savefig(dir + '/' + file_name + '.png', dpi=DPI)

    if clear:
        plt.clf()
