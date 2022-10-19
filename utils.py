import numpy as np


def none_or_int(v):
    none_names = ['None', 'Nan', 'Null', 'null']
    if v in none_names:
        return None
    else:
        return int(v)

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
