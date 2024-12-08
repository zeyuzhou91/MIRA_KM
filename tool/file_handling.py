"""
File handling

"""

import os
import glob
from pathlib import Path
import shutil


def delete_dir(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    return None
    
    
def empty_dir(dir_path):
    """
    Remove all content of the directory.
    """    
    
    content1 = glob.glob(os.path.join(dir_path, '*'))
    content2 = glob.glob(os.path.join(dir_path, '.*'))
    content = content1 + content2
    
    for f in content:
        if os.path.isfile(f) or os.path.islink(f):
            delete_file(f)
        elif os.path.isdir(f):
            delete_dir(f)
    return None    
    

    
def create_dir(dir_path):
    os.makedirs(dir_path, exist_ok=True)
    return None
    

def delete_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
    return None


def delete_file_list(file_list):
    for file in file_list:
        delete_file(file)
    return None


def copy_file(src_path, dst_path):
    if os.path.exists(src_path):
        shutil.copy2(src_path, dst_path)
    return None


def move_file(src_path, dst_path):
    if os.path.exists(src_path):
        shutil.move(src_path, dst_path)
    return None


def rename_file(src_path, dst_path):
    if os.path.exists(src_path):
        os.rename(src_path, dst_path)
    return None




