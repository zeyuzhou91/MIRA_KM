"""
environment.py

Environment management for runtime variable tracking.

Provides:
- A container class `Environment` to store named variables initialized from a config file
- Methods to print, reset, and load variables from a list

Author: Zeyu Zhou
Date: 2025-05-22
"""

class Environment:
    
    def __init__(self, var_names_file):
        """
        Initialize the Environment by reading variable names from a file.

        Parameters:
        - var_names_file (str): Path to a file containing one variable name per line.
        """
        
        self.variables = {}
        
        self.read_var_names_from_file(var_names_file)
        
        
    def print_all(self):
        """
        Print all variables and their current values.
        """
        
        for (name, value) in self.variables.items():
            print(f'{name} = {value}')
        
        return None
    
    
    def reset_all(self):
        """
        Reset all variable values to an empty string.
        """
        
        for name in self.variables.keys():
            self.variables[name] = ''
            
        
    def read_var_names_from_file(self, file_path):
        """
        Read variable names from a text file.
        Each line should contain a single variable name.

        Parameters:
        - file_path (str): Path to the variable names file.

        Sets self.variables to a dict of {var_name: ''}.
        """
        
        try:
            with open(file_path, 'r') as file:
                for line in file:
                    # Strip whitespace and add non-empty lines to the array
                    var_name = line.strip()
                    if var_name:
                        self.variables[var_name] = ''
        except FileNotFoundError:
            print(f"Error: File not found at {file_path}")
        except IOError:
            print(f"Error: Could not read file at {file_path}")
        
        return None

    
    
    
    