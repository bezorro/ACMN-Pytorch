import json
import shutil
import os
import argparse

def read_json(fname):
    file = open(fname, 'r')
    res  = json.load(file)
    file.close()
    return res

def write_json(obj, fname):
    file = open(fname, 'w')
    js   = json.dumps(obj)
    file.write(js)
    file.close()

def print_save(path, obj, to_screen = True):

    if to_screen : print(obj)
    if path == None : return

    if not os.path.isdir(path): os.makedirs(path)
    f = open(os.path.join(path, 'params.txt'), 'a')
    print(str(obj), file = f)
    f.close()

def forced_copydir(sdir, ddir):
    
    d_dir_path = os.path.dirname(ddir)
    if not os.path.isdir(d_dir_path) : os.makedirs(d_dir_path)
    if os.path.isdir(ddir) : shutil.rmtree(ddir)
    shutil.copytree(sdir, ddir)

def update_opt_remove_prefix(sopt, aopt, prefix = ''):

    aopt = dict(filter(lambda kv : kv[0].startswith(prefix), aopt.__dict__.items())) # remove prefix
    out_opt = { **sopt.__dict__, **aopt }

    return argparse.Namespace(**out_opt) # my_opt = { opt_prefix + '...' }