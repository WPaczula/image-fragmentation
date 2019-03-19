import sys

def get_key_by_value(dict, value):
    return list(dict.keys())[list(dict.values()).index(value)]

def count_used_file_number(used_labels, file_name):
    count = 0
    with open(file_name) as f:
        for file_name in f:
            if file_name.split('/')[0] in used_labels:
                count += 1
    return count

def print_in_line(text):
    sys.stdout.write("\r" + text)
    sys.stdout.flush()