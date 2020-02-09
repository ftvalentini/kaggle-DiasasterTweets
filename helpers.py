import os

def create_dir(dir):
    """ create directory if not exists
    """
    if not os.path.exists(dir):
        os.makedirs(dir)
