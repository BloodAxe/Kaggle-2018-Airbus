import glob
import os


def find_in_dir(dirname):
    return [os.path.join(dirname, fname) for fname in sorted(os.listdir(dirname))]


def id_from_fname(fname):
    return os.path.splitext(os.path.basename(fname))[0]


def auto_file(filename, where='.') -> str:
    """
    Helper function to find a unique filename in subdirectory without specifying fill path to it
    :param where:
    :param filename:
    :return:
    """

    if os.path.isabs(filename):
        return filename

    prob = os.path.join(where, filename)
    if os.path.exists(prob) and os.path.isfile(prob):
        return prob

    files = list(glob.iglob(os.path.join(where, '**', filename), recursive=True))
    if len(files) == 0:
        raise FileNotFoundError('Given file could not be found with recursive search:' + filename)

    if len(files) > 1:
        raise FileNotFoundError('More than one file matches given filename. Please specify it explicitly' + filename)

    return files[0]
