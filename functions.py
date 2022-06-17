import os


def make_directory(path):
    try:
        os.mkdir(os.path.join(path))
    except FileExistsError:
        pass
