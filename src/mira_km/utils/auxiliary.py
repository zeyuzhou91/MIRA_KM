"""
aux.py

Auxiliary utility functions and legacy class stubs for data I/O, 
especially for reading and writing CSV files with headers and units, 
and for kinetic model unit mappings.

Author: Zeyu Zhou
Date: 2025-05-15
"""

import os
import csv
import matplotlib.pyplot as plt
import numpy as np
import shutil
from numpy.typing import NDArray
# from ..core import ROI, Environment


# To remove Environment
# class Environment:
#     def __init__(self):
        
#         self.root_dir = ''  # root directory path
#         self.subj_dir = ''  # directory path of the subject
#         self.MRI_dir = ''   # directory path for MRI images and segmentations
#         self.PET_dir = ''   # directory path for PET images
#         self.AIF_dir = ''   # directory path for AIF data
#         self.km_dir = ''    # directory path for kinetic modeling (temp)
#         self.masks_dir = '' # directory path for masks 
#         self.seg_path = ''  # path of the segmentation file
#         self.mr2pet_lta_path = '' # path of the MR to PET linear transformation file
#         self.pet2mr_lta_path = '' # path of the PET to MR linear transformation file
#         self.framedurationfile_path = '' # path of the frame durations file
#         self.framemidpointfile_path = '' # path of the frame mid-points file 
#         self.AIF_ptac_path = ''
#         self.AIF_pif_path = ''
#         self.AIF_p2wb_ratio_path = ''


# To remove ROI
# class ROI:
#     def __init__(self, name):
#         self.name = name     # string
#         self.ID = None       # integer
#         self.num_frames = None   # positive integer, 1 for single frame
#         self.num_voxels = None  # integer
#         self.vol_ml = None   # volume in [mL]
        
#         self.avg_intensity = []  # list of decimals, average intensity of the voxels (mean of voxels) 
#         self.tot_intensity = []  # list of decimals, total intensity of the voxels (sum of voxels)
#         self.concentration = []     # list of decimals, average concentration in [Bq/mL], for dynamic imaging          

#         self.onetcm_params = {'K1': None, 'k2': None, 'VD': None}  
#         self.twotcm_params = {'K1': None, 'k2': None, 'k3': None, 'k4':None, 'VND':None, 'VT':None, 'VS':None, 'BPND':None}              


# To remove FrameSchedule
# class FrameSchedule:
#     def __init__(self, durations=None, mid_points=None):
#         self.durations = []   # list of numbers
#         self.start_points = []       # list of numbers
#         self.mid_points = []         # list of numbers
        
#         if mid_points == None:
#             self.calculate_attributes_from_durations(durations)
#         elif durations == None:
#             self.calculate_attributes_from_midpoints(mid_points)
        
#     def calculate_attributes_from_durations(self, durations):
#         self.durations = durations
        
#         cur_start = 0.0
#         for duration in self.durations:
            
#             self.start_points.append(cur_start)
            
#             nxt_start = cur_start + duration
            
#             mid = (cur_start + nxt_start) / 2.0
#             self.mid_points.append(mid)
            
#             cur_start = nxt_start
    
#         return None

#     def calculate_attributes_from_midpoints(self, mid_points):
#         self.mid_points = mid_points
        
#         cur_start = 0.0
#         for mid in self.mid_points:
            
#             self.start_points.append(cur_start)
            
#             duration = 2 * (mid - cur_start)
#             self.durations.append(duration)
            
#             nxt_start = cur_start + duration
            
#             cur_start = nxt_start
    
#         return None
            


# To remove TimeCurve
# Timed curve of a quantity
# class TimeCurve:
#     def __init__(self, name):
#         self.name = name   # string, name of the quantity, or name of the tissue 
#         self.t_data = []     # list of floats, time points
#         self.y_data = []   # list of floats, values of quantity
#         self.t_unit = ''     # string
#         self.y_unit = ''  # string, unit of measured quantity
#         self.fitfunc = None  # function, the fitting function
#         self.fitparams = None   # numpy.ndarray, parameters of the fitting function

#     def plot(self, xlim=None, ylim=None):
#         """
#         Plot the data and fitted curve. 
#         """
        
#         plt.figure()
#         if self.t_data != [] and self.y_data != []: 
#             plt.scatter(self.t_data, self.y_data, c='blue', label='data')
#         if self.t_data != [] and self.fitfunc != None and self.fitparams.all() != None:
#             tmax = np.max(self.t_data)
#             tfit = np.linspace(0, tmax*1.1, 1000)
#             yfit = self.fitfunc(tfit, *self.fitparams)
#             plt.plot(tfit, yfit, c='red', label='fit')
#         plt.xlabel(f'Time ({self.t_unit})')
#         if self.y_unit == 'unitless':
#             plt.ylabel(f'{self.name}')
#         else:
#             plt.ylabel(f'{self.name} ({self.y_unit})')
#         if xlim == None:
#             pass
#         else:
#             plt.xlim(xlim)
#         if ylim == None:
#             pass
#         else:
#             plt.ylim(ylim)
#         plt.legend()
#         plt.show()
        
#         return None
    
#     def print_fitparams(self):
#         """
#         Print the fitting parameters. 
#         """
        
#         print('Fitting parameters:\n')
#         print(self.fitparams)
#         return None
        

# To remove plot_timecurve
# def plot_timecurve(tc):
    
        
#     tmax = np.max(tc.t_data)
#     tfit = np.linspace(0, tmax*1.1, 1000)
#     yfit = tc.fitfunc(tfit, *tc.fitparams)
    
#     plt.figure()
#     plt.scatter(tc.t_data, tc.y_data, c='blue', label='data')
#     plt.plot(tfit, yfit, c='red', label='fit')
#     plt.xlabel(f'Time ({tc.t_unit})')
#     plt.ylabel(f'{tc.name} ({tc.y_unit})')
#     plt.legend()
#     plt.show()
    
#     return None
    

# To remove delete_folder_contents (implenmented in file_handling)
# def delete_folder_contents(folder):
    
#     for filename in os.listdir(folder):
#         file_path = os.path.join(folder, filename)
#         try:
#             if os.path.isfile(file_path) or os.path.islink(file_path):
#                 os.unlink(file_path)
#             elif os.path.isdir(file_path):
#                 shutil.rmtree(file_path)
#         except Exception as e:
#             print('Failed to delete %s. Reason: %s' % (file_path, e))
        
#     return None


# To move extract_file_name to file.py
# def extract_file_name(file_path):
#     """
#     Extract the file base name and extension from its path. 
    
#     Example: /home/documents/mask_cerebellum.nii.gz
#     Return: mask_cerebellum and .nii.gz

#     Parameters
#     ----------
#     file_path : string, file path
#         The full path of the file.

#     Returns
#     -------
#     basename : string
#         The base name of the file. 
#     extension: string
#         The extension of the file. 
#     """

#     # os.path.basename returns the full name of the file, including the extension
#     fullname = os.path.basename(file_path)
    
#     # split fullname from the first occurrence of '.'
#     # e.g.: mask_cerebellum.nii.gz => mask_cerebellum and nii.gz
#     basename, extension = fullname.split('.', 1)
#     extension = '.' + extension
    
#     return basename, extension
    


# ==========================
# CSV WRITE FUNCTIONS
# ==========================

def write_to_csv_onecol(arr: NDArray, header: str, unit: str, csvfile_path: str) -> None:
    """
    Write a 1D array to a CSV file with a single column, including a header and unit row.

    Parameters
    ----------
    arr : NDArray
        The 1D array to write.
    header : str
        Column name.
    unit : str
        Unit of the column data.
    csvfile_path : str
        File path to save the CSV.
    """
    with open(csvfile_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow([header])
        csv_writer.writerow([unit])
        for x in arr:
            csv_writer.writerow([x])
            
    return None


def write_to_csv_twocols(arr1, header1, unit1, arr2, header2, unit2, csvfile_path):
    """
    Write two 1D arrays to a CSV file as two columns, including headers and units.
    """
    with open(csvfile_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow([header1, header2])
        csv_writer.writerow([unit1, unit2])
        for x, y in zip(arr1, arr2):
            csv_writer.writerow([x, y])
            
    return None
    

def write_to_csv_threecols(arr1, header1, unit1, 
                           arr2, header2, unit2, 
                           arr3, header3, unit3,
                           csvfile_path):
    """
    Write three 1D arrays to a CSV file as three columns, including headers and units.
    """
    with open(csvfile_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow([header1, header2, header3])
        csv_writer.writerow([unit1, unit2, unit3])
        for x, y, z in zip(arr1, arr2, arr3):
            csv_writer.writerow([x, y, z])
            
    return None


def write_to_csv_multicols(arr: NDArray, headers: list[str], units: list[str], csvfile_path: str):
    """
    Write an N x M array to a CSV file with multiple columns, including headers and units.

    Parameters
    ----------
    arr : NDArray
        Array with shape (N, M).
    headers : list of str
        Column names (length N).
    units : list of str
        Units of each column (length N).
    csvfile_path : str
        File path to save the CSV.
    """
    with open(csvfile_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(headers)
        csv_writer.writerow(units)
        N, M = arr.shape
        for i in range(M):
            csv_writer.writerow(arr[:, i])
    
    return None



# ==========================
# CSV READ FUNCTIONS
# ==========================

def read_from_csv_onecol(filepath: str) -> (list[float], str, str):
    """
    Read a single-column CSV file with header and unit rows.

    Returns
    -------
    data : list of float
    header : str
    unit : str
    """
    with open(filepath, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        header = next(csv_reader)[0]
        unit = next(csv_reader)[0]
        data = [float(row[0]) for row in csv_reader]
    return data, header, unit



def read_from_csv_twocols(filepath: str) -> (list[float], str, str, list[float], str, str):
    """
    Read a two-column CSV file with header and unit rows.

    Returns
    -------
    data1, header1, unit1, data2, header2, unit2
    """
    data1, data2 = [], []
    with open(filepath, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        header1, header2 = next(csv_reader)
        unit1, unit2 = next(csv_reader)
        for row in csv_reader:
            data1.append(float(row[0]))
            data2.append(float(row[1]))
    return data1, header1, unit1, data2, header2, unit2    

     

def read_from_csv_threecols(filepath: str) -> (list[float], str, str, 
                                               list[float], str, str,
                                               list[float], str, str):
    """
    Read a three-column CSV file with header and unit rows.

    Returns
    -------
    data1, header1, unit1, data2, header2, unit2, data3, header3, unit3
    """
    data1, data2, data3 = [], [], []
    with open(filepath, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        header1, header2, header3 = next(csv_reader)
        unit1, unit2, unit3 = next(csv_reader)
        for row in csv_reader:
            data1.append(float(row[0]))
            data2.append(float(row[1]))
            data3.append(float(row[2]))
    return data1, header1, unit1, data2, header2, unit2, data3, header3, unit3



def read_from_csv_fourcols(filepath: str) -> (list[float], list[float], list[float], list[float]):
    """
    Read a four-column CSV file. Header and unit rows are ignored in the return.

    Returns
    -------
    data1, data2, data3, data4 : list of float
    """
    data1, data2, data3, data4 = [], [], [], []
    with open(filepath, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        _ = next(csv_reader)  # skip headers
        _ = next(csv_reader)  # skip units
        for row in csv_reader:
            data1.append(float(row[0]))
            data2.append(float(row[1]))
            data3.append(float(row[2]))
            data4.append(float(row[3]))
    return data1, data2, data3, data4




# ==========================
# MODEL UNIT DICTIONARY
# ==========================

def model_unit_table(model_name: str) -> dict[str, str]:
    """
    Return a dictionary of parameter units for a given kinetic model.

    Parameters
    ----------
    model_name : str
        Name of the model, e.g., '1TCM', '2TCM', 'Logan', 'RTM', 'SRTM'.

    Returns
    -------
    unit_table : dict
        Mapping from parameter names to their units.
        
    """
    
    if model_name == '1TCM':
        return {'K1': 'mL/min/mL', 
                'k2': '/min', 
                'VB': 'unitless', 
                'VD': 'unitless'}
    
    elif model_name == '2TCM':
        return {'K1': 'mL/min/mL', 
                'k2': '/min', 
                'k3': '/min', 
                'k4': '/min',
                'VB': 'unitless', 
                'VND': 'unitless', 
                'VS': 'unitless',
                'VT': 'unitless', 
                'BPND': 'unitless'}
    
    elif model_name == 'Logan':
        return {'slope': 'unitless', 
                'intercept': 'min', 
                'tstart': 'min'}
    
    elif model_name == 'RTM':
        return {'R1': 'unitless', 
                'k2': '/min', 
                'k3': '/min',
                'k4': '/min', 
                'BPND': 'unitless'}
    
    elif model_name == 'SRTM':
        return {'R1': 'unitless', 
                'k2': '/min', 
                'BPND': 'unitless'}
    
    else:
        return {}




            
# if __name__ == "__main__":
    

    
#     pass
    
    
    
    