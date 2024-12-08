"""
Subject
"""

import pandas as pd
    
    
class Cohort:
    
    def __init__(self, info_dict: dict):
        
        self.info = info_dict
        
        
    @classmethod
    def from_excel(cls, filepath: str):
        """
        Read subject info from a xlsx file. 
        """
        
        info_dict = {}
        
        df = pd.read_excel(filepath)
                
        headers = df.columns.tolist()
        #print(headers)
        
        subjects = df[headers[0]].tolist()
        #print(subjects)
        
        headers = headers[1:]  # remove the first header
        
        for (index, subject) in enumerate(subjects):
            info_dict[subject] = {}
            
            for header in headers:
                info_dict[subject][header] = df[header].iloc[index]
                    
        return cls(info_dict)
        
    
    def print_info(self):
        
        for (subject, subj_info) in self.info.items():
            print("=================")
            print(subject)
            print("=================")
            
            for (header, value) in subj_info.items():
                print(f'{header}: {value}, {type(value)}')
            
            print("")
            
        return None
    
    
    def add_header(self, h: str | list[str]):
        
        if isinstance(h, str):
        
            for (subj, subj_info) in self.info.items():
                subj_info[h] = None
        
        elif isinstance(h, list):
            
            for (subject, subj_info) in self.info.items():
                for header in h:
                    subj_info[header] = None
        
        return None
    



    def assign_value(self, 
                    subj: str,
                    header: str,
                    value: float):
        
        self.info[subj][header] = value
            
        return None


    def get_value(self, 
                  subj: str,
                  header: str):
        
        
        v = self.info[subj][header] 
            
        return v


    
    def write_to_excel(self, filepath: str):
        """
        Write info to an excel file. 
        """
                
        df = pd.DataFrame(self.info).T  # Transpose to get subjects as rows
        
        df.to_excel(filepath, engine='openpyxl')
        
        return None
    