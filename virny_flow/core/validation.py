import argparse


def check_str_list_type(str_param):
    if '[' in str_param and ']' in str_param:
        return True
    return False


def has_unique_elements(lst):
    return len(lst) == len(set(lst))


def is_in_enum(val, enum_obj):
    enum_vals = [member.value for member in enum_obj]
    return val in enum_vals


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
