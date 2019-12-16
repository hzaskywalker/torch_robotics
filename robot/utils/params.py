import json
import numpy as np
import argparse


def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file
    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: (float(v) if isinstance(v, np.float) else v)
             for k, v in d.items()}
        json.dump(d, f, indent=4)


def dict2json(d):
    d = {k: (float(v) if isinstance(v, np.float) else v)
         for k, v in d.items()}
    return json.dumps(d, indent=4)


def parse_args(args, prefix=''):
    # prefix either end with . or ''
    from tools import Field
    prefixes = []
    l = len(prefix)
    out = Field()
    for i in args.__dict__:
        if i.startswith(prefix):
            if '.' in i[l:]:
                prefixes.append( i[l:].split('.')[0] )
            else:
                out[i[l:]] = args.__dict__[i]
    prefixes = list(set(prefixes))
    for i in prefixes:
        out[i] = parse_args(args, prefix + i + '.')
    return out

def flatten_json(j):
    out = {}
    for i in j:
        if isinstance(j[i], dict):
            t = flatten_json(j[i])
            for k in t:
                out['{}.{}'.format(i,k)] = t[k]
        else:
            out[i] = j[i]
    return out


def update_args(args, parser=None):
    if args.json is not None:
        with open(args.json, 'r') as f:
            j = flatten_json(json.load(f))
            for k, v in j.items():
                if k in args.__dict__:
                    args.__dict__[k] = v

    if parser is not None:
        import sys
        from tools import Field

        a = Field()
        parser._parse_known_args(sys.argv[1:], a)
        for i in a:
            if i != 'json':
                args.__dict__[i] = a[i]
    args = parse_args(args)
    return args

def get_args(parser=None):
    from tools import Field
    from tools.utils import seed
    from tools.trainer import get_parser
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument('--json', type=str, default=None)
    parser.add_argument('--seed', type=int, default=0)
    get_parser(parser)
    args = parser.parse_args()

    seed(args.seed)
    args = update_args(args, parser)

    args.json = dict2json(
        args.__dict__ if not isinstance(args, Field) else args)
    return args

