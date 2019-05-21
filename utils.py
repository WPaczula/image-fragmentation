import sys

def get_key_by_value(dict, value):
    return list(dict.keys())[list(dict.values()).index(value)]

def print_in_line(text):
    sys.stdout.write("\r" + text)
    sys.stdout.flush()