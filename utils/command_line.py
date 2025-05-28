"""
command_line.py

Utility for working with terminal commands within Python.

Author: Zeyu Zhou
Date: 2025-05-21
"""

import subprocess



def command_full_path(c: str):
    """
    Return the full path of a shell command by calling `which`.

    Parameters:
    - c: Command name as a string (e.g., 'python', 'flirt').

    Returns:
    - Full path to the executable as a string (including newline).
      Note: You may want to use `.strip()` to remove the newline.
    """
    
    quiry = ['which', c]
    result = subprocess.run(quiry, capture_output=True, text=True)
    c_full_path = result.stdout
    return c_full_path