"""
Environment
"""

class Environment:
    def __init__(self, var_names_file):
        
        self.variables = {}
        
        self.read_var_names_from_file(var_names_file)
        
        
    def print_all(self):
        
        for (name, value) in self.variables.items():
            print(f'{name} = {value}')
        
        return None
    
    
    def reset_all(self):
        
        for name in self.variables.keys():
            self.variables[name] = ''
            
        
    def read_var_names_from_file(self, file_path):
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

    
    
    
    