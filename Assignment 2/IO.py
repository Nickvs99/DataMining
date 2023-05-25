import os

def get_next_path(path):
    """
    Finds the next free path in an sequentially named list of files

    e.g. path = 'file.txt':

    file-1.txt
    file-2.txt
    file-3.txt

    Runs in log(n) time where n is the number of existing files in sequence

    Credit: https://stackoverflow.com/a/47087513/12132063
    """
    
    path_pattern = get_path_pattern(path)

    i = 1

    # First do an exponential search
    while os.path.exists(path_pattern % i):
        i = i * 2

    # Result lies somewhere in the interval (i/2..i]
    # We call this interval (a..b] and narrow it down until a + 1 = b
    a, b = (i // 2, i)
    while a + 1 < b:
        c = (a + b) // 2 # interval midpoint
        a, b = (c, b) if os.path.exists(path_pattern % c) else (a, c)

    return path_pattern % b

def get_path_pattern(path):

    path_split = path.split(".")
    extension = path_split[-1]
    file_name = ".".join(path_split[:-1]) + "-%s"
    return ".".join([file_name, extension])
