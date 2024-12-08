"""
Terminal

"""


import subprocess


def command_full_path(c: str):
    """
    Return the full path of a command.
    """
    
    quiry = ['which', c]
    result = subprocess.run(quiry, capture_output=True, text=True)
    
    c_full_path = result.stdout
    
    return c_full_path
    
    


