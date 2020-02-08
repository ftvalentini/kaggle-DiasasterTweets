import os

def create_dir(dir):
    """ create directory if not exists
    """
    if not os.path.exists(dir):
        os.makedirs(dir)

def remove_bad_characters(string):
    """ removes weird characters from string
    such as: \x89Ûª, \x89Û¢åÊ, åÇ, etc.
    """
    clean_string = string.encode('ascii', errors='ignore').decode('utf-8')
    return clean_string
