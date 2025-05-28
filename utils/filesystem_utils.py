"""
filesystem_utils.py

Utility functions for common file and directory operations.

Author: Zeyu Zhou
Date: 2025-05-20
"""

import os
import glob
import shutil
from pathlib import Path


def delete_dir(dir_path: str) -> None:
    """
    Delete a directory and all its contents.
    
    Parameters:
    -----------
    dir_path : str
        Path to the directory to delete.
    """
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
        
    return None


def empty_dir(dir_path: str) -> None:
    """
    Remove all contents (including hidden files) of a directory without deleting the directory itself.

    Parameters:
    -----------
    dir_path : str
        Path to the directory to empty.
    """
    contents = glob.glob(os.path.join(dir_path, '*')) + \
               glob.glob(os.path.join(dir_path, '.*'))

    for f in contents:
        if os.path.isfile(f) or os.path.islink(f):
            delete_file(f)
        elif os.path.isdir(f):
            delete_dir(f)
    
    return None


def create_dir(dir_path: str) -> None:
    """
    Create a directory, including any necessary parent directories.

    Parameters:
    -----------
    dir_path : str
        Path of the directory to create.
    """
    os.makedirs(dir_path, exist_ok=True)
    
    return None


def rename_dir(old_dir: str, new_dir: str) -> None:
    """
    Rename a directory.

    Parameters:
    -----------
    old_dir : str
        Path to the existing directory.
    new_dir : str
        New directory name or path.
    """
    if os.path.exists(old_dir):
        os.rename(old_dir, new_dir)
        
    return None    


def copy_dir(src_dir: str, dst_dir: str) -> None:
    """
    Copy an entire directory to a new location.

    Parameters:
    -----------
    src_dir : str
        Source directory path.
    dst_dir : str
        Destination directory path.
    """
    if os.path.exists(src_dir):
        shutil.copytree(src_dir, dst_dir)
        
    return None


def delete_file(file_path: str) -> None:
    """
    Delete a file.

    Parameters:
    -----------
    file_path : str
        Path to the file to delete.
    """
    if os.path.exists(file_path):
        os.remove(file_path)

    return None


def delete_file_list(file_list: list) -> None:
    """
    Delete multiple files.

    Parameters:
    -----------
    file_list : list
        List of file paths to delete.
    """
    for file in file_list:
        delete_file(file)
    
    return None


def copy_file(src_path: str, dst_path: str) -> None:
    """
    Copy a file to a new location.

    Parameters:
    -----------
    src_path : str
        Source file path.
    dst_path : str
        Destination file path.
    """
    if os.path.exists(src_path):
        shutil.copy2(src_path, dst_path)
        
    return None


def move_file(src_path: str, dst_path: str) -> None:
    """
    Move a file to a new location.

    Parameters:
    -----------
    src_path : str
        Source file path.
    dst_path : str
        Destination file path.
    """
    if os.path.exists(src_path):
        shutil.move(src_path, dst_path)

    return None


def rename_file(old_path: str, new_path: str) -> None:
    """
    Rename a file.

    Parameters:
    -----------
    old_path : str
        Original file path.
    new_path : str
        New file path.
    """
    if os.path.exists(old_path):
        os.rename(old_path, new_path)

    return None

