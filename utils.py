import sys

def get_key_by_value(dict, value):
    return list(dict.keys())[list(dict.values()).index(value)]

def get_file_length(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

def print_in_line(text):
    sys.stdout.write("\r" + text)
    sys.stdout.flush()