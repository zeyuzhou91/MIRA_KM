"""
subject.py

Manages subject-level metadata for a PET study cohort.
Supports reading/writing from Excel, editing subject-level fields,
filtering subjects based on metadata, and exporting the dataset.

Author: Zeyu Zhou
Date: 2025-05-21
"""

from typing import Dict, List, Any
import pandas as pd
    
    
class SubjectRegistry:
    """
    Manages metadata for a cohort of subjects.

    Attributes:
    - info (dict): Nested dictionary of subject metadata {subject: {field: value}}
    """
    
    def __init__(self, info_dict: dict):
        
        self.info = info_dict
        
        
    @classmethod
    def from_excel(cls, filepath: str):
        """
        Create SubjectRegistry from an Excel file.
        The first column is treated as subject ID; the rest as metadata.

        Parameters:
        - filepath (str): Path to the Excel file

        Returns:
        - SubjectRegistry object
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
        """Prints detailed metadata for all subjects."""
        
        for (subject, subj_info) in self.info.items():
            print("=================")
            print(subject)
            print("=================")
            
            for (header, value) in subj_info.items():
                print(f'{header}: {value}, {type(value)}')
            
            print("")
            
        return None
    
    
    def add_header(self, h: str | list[str]):
        """
        Add a new metadata field (header) for all subjects and initialize to None.

        Parameters:
        - h (str or list[str]): Header(s) to add
        """
        
        if isinstance(h, str):
        
            for (subj, subj_info) in self.info.items():
                subj_info[h] = None
        
        elif isinstance(h, list):
            
            for (subject, subj_info) in self.info.items():
                for header in h:
                    subj_info[header] = None
        
        return None
 
    
    def add_subj_header_value(self, 
                              subj: str,
                              header: str,
                              value: float):
        """
        Assign a value to a specific header for a subject.
        """
        
        self.info[subj][header] = value
            
        return None
    


    def assign_value(self, 
                    subj: str,
                    header: str,
                    value: float):
        """
        Assign a value to a specific header for a subject.
        
        Alias for add_subj_header_value
        """
        
        self.info[subj][header] = value
            
        return None


    def get_value(self, 
                  subj: str,
                  header: str):
        """
        Retrieve value for a given subject and header.
        """        
        
        v = self.info[subj][header] 
            
        return v


    def all_values_of_header(self, header: str):
        """
        Retrieve all values of a specific header across subjects.

        Returns:
        - List of values
        """
        
        all_values = []
        
        for (subj, subj_info) in self.info.items():
            v = self.info[subj][header]
            all_values.append(v)
            
        return all_values
    
    
    def all_subj_names(self):
        """
        Return a list of all subject IDs.
        
        ZEYU: maybe a better name?
        """
        
        return list(self.info.keys())
        
    
    def write_to_excel(self, filepath: str):
        """
        Write the subject metadata to an Excel file.

        Parameters:
        - filepath (str): Output Excel path
        """
                
        df = pd.DataFrame(self.info).T  # Transpose to get subjects as rows
        
        df.index.name = "directory"  # Set the name of the index
        
        df.to_excel(filepath, engine='openpyxl')
        
        return None


    def filter_subjects(self, criteria: Dict[str, Any]) -> List[str]:
        """
        Filters subjects based on specified criteria.
    
        Parameters:
        - criteria (dict): A dictionary where keys are headers (column names) and 
          values are the values to filter by. Supports exact matches.
    
        Returns:
        - List of subject names that match all the given criteria.
        
        Example Usage:
            
            criteria = {"Age": 30, "Gender": "Male"}
            filtered = cohort.filter_subjects(criteria)
            print(filtered)  # Outputs subjects that are 30 years old and male
        """
        
        filtered_subjects = []
    
        for subj, subj_info in self.info.items():
            match = all(subj_info.get(header) == value for header, value in criteria.items())
            if match:
                filtered_subjects.append(subj)
    
        return filtered_subjects
    