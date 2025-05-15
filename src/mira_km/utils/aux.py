"""
Auxiliary classes and functions.
"""

import os
import csv
import matplotlib.pyplot as plt
import numpy as np
import shutil
from numpy.typing import NDArray
# from ..core import ROI, Environment



class Environment:
    def __init__(self):
        
        self.root_dir = ''  # root directory path
        self.subj_dir = ''  # directory path of the subject
        self.MRI_dir = ''   # directory path for MRI images and segmentations
        self.PET_dir = ''   # directory path for PET images
        self.AIF_dir = ''   # directory path for AIF data
        self.km_dir = ''    # directory path for kinetic modeling (temp)
        self.masks_dir = '' # directory path for masks 
        self.seg_path = ''  # path of the segmentation file
        self.mr2pet_lta_path = '' # path of the MR to PET linear transformation file
        self.pet2mr_lta_path = '' # path of the PET to MR linear transformation file
        self.framedurationfile_path = '' # path of the frame durations file
        self.framemidpointfile_path = '' # path of the frame mid-points file 
        self.AIF_ptac_path = ''
        self.AIF_pif_path = ''
        self.AIF_p2wb_ratio_path = ''


class ROI:
    def __init__(self, name):
        self.name = name     # string
        self.ID = None       # integer
        self.num_frames = None   # positive integer, 1 for single frame
        self.num_voxels = None  # integer
        self.vol_ml = None   # volume in [mL]
        
        self.avg_intensity = []  # list of decimals, average intensity of the voxels (mean of voxels) 
        self.tot_intensity = []  # list of decimals, total intensity of the voxels (sum of voxels)
        self.concentration = []     # list of decimals, average concentration in [Bq/mL], for dynamic imaging          

        self.onetcm_params = {'K1': None, 'k2': None, 'VD': None}  
        self.twotcm_params = {'K1': None, 'k2': None, 'k3': None, 'k4':None, 'VND':None, 'VT':None, 'VS':None, 'BPND':None}              


class FrameSchedule:
    def __init__(self, durations=None, mid_points=None):
        self.durations = []   # list of numbers
        self.start_points = []       # list of numbers
        self.mid_points = []         # list of numbers
        
        if mid_points == None:
            self.calculate_attributes_from_durations(durations)
        elif durations == None:
            self.calculate_attributes_from_midpoints(mid_points)
        
    def calculate_attributes_from_durations(self, durations):
        self.durations = durations
        
        cur_start = 0.0
        for duration in self.durations:
            
            self.start_points.append(cur_start)
            
            nxt_start = cur_start + duration
            
            mid = (cur_start + nxt_start) / 2.0
            self.mid_points.append(mid)
            
            cur_start = nxt_start
    
        return None

    def calculate_attributes_from_midpoints(self, mid_points):
        self.mid_points = mid_points
        
        cur_start = 0.0
        for mid in self.mid_points:
            
            self.start_points.append(cur_start)
            
            duration = 2 * (mid - cur_start)
            self.durations.append(duration)
            
            nxt_start = cur_start + duration
            
            cur_start = nxt_start
    
        return None
            

# Timed curve of a quantity
class TimeCurve:
    def __init__(self, name):
        self.name = name   # string, name of the quantity, or name of the tissue 
        self.t_data = []     # list of floats, time points
        self.y_data = []   # list of floats, values of quantity
        self.t_unit = ''     # string
        self.y_unit = ''  # string, unit of measured quantity
        self.fitfunc = None  # function, the fitting function
        self.fitparams = None   # numpy.ndarray, parameters of the fitting function

    def plot(self, xlim=None, ylim=None):
        """
        Plot the data and fitted curve. 
        """
        
        plt.figure()
        if self.t_data != [] and self.y_data != []: 
            plt.scatter(self.t_data, self.y_data, c='blue', label='data')
        if self.t_data != [] and self.fitfunc != None and self.fitparams.all() != None:
            tmax = np.max(self.t_data)
            tfit = np.linspace(0, tmax*1.1, 1000)
            yfit = self.fitfunc(tfit, *self.fitparams)
            plt.plot(tfit, yfit, c='red', label='fit')
        plt.xlabel(f'Time ({self.t_unit})')
        if self.y_unit == 'unitless':
            plt.ylabel(f'{self.name}')
        else:
            plt.ylabel(f'{self.name} ({self.y_unit})')
        if xlim == None:
            pass
        else:
            plt.xlim(xlim)
        if ylim == None:
            pass
        else:
            plt.ylim(ylim)
        plt.legend()
        plt.show()
        
        return None
    
    def print_fitparams(self):
        """
        Print the fitting parameters. 
        """
        
        print('Fitting parameters:\n')
        print(self.fitparams)
        return None
        

def plot_timecurve(tc):
    
        
    tmax = np.max(tc.t_data)
    tfit = np.linspace(0, tmax*1.1, 1000)
    yfit = tc.fitfunc(tfit, *tc.fitparams)
    
    plt.figure()
    plt.scatter(tc.t_data, tc.y_data, c='blue', label='data')
    plt.plot(tfit, yfit, c='red', label='fit')
    plt.xlabel(f'Time ({tc.t_unit})')
    plt.ylabel(f'{tc.name} ({tc.y_unit})')
    plt.legend()
    plt.show()
    
    return None
    


def delete_folder_contents(folder):
    
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
        
    return None



def extract_file_name(file_path):
    """
    Extract the file base name and extension from its path. 
    
    Example: /home/documents/mask_cerebellum.nii.gz
    Return: mask_cerebellum and .nii.gz

    Parameters
    ----------
    file_path : string, file path
        The full path of the file.

    Returns
    -------
    basename : string
        The base name of the file. 
    extension: string
        The extension of the file. 
    """

    # os.path.basename returns the full name of the file, including the extension
    fullname = os.path.basename(file_path)
    
    # split fullname from the first occurrence of '.'
    # e.g.: mask_cerebellum.nii.gz => mask_cerebellum and nii.gz
    basename, extension = fullname.split('.', 1)
    extension = '.' + extension
    
    return basename, extension
    


def write_to_csv_onecol(arr: NDArray, header: str, unit: str, csvfile_path: str) -> None:
    """
    Write a 1D array to a CSV file as a column. 

    Parameters
    ----------
    csvfile_path : file path, ending in .csv
    """
    
    with open(csvfile_path, 'w', newline='') as csvfile:
        # Create a CSV writer object
        csv_writer = csv.writer(csvfile)

        # Write the header
        csv_writer.writerow([header])
        
        # Write the unit
        csv_writer.writerow([unit])

        # Write the data
        for x in arr:
            csv_writer.writerow([x])
    
    return None

    

def write_to_csv_twocols(arr1, header1, unit1, arr2, header2, unit2, csvfile_path):
    """
    Write two 1D arrays to a CSV file as two columns.  

    Parameters
    ----------
    header1 : string
    arr1 : list of numbers
    header2 : string
    arr2 : list of numbers
    csvfile_path : string, file path, ending in .csv

    Returns
    -------
    None.

    """
    
    # Writing to CSV file
    with open(csvfile_path, 'w', newline='') as csvfile:
        # Create a CSV writer object
        csv_writer = csv.writer(csvfile)
    
        # Write the header
        csv_writer.writerow([header1, header2])
        
        # Write the unit
        csv_writer.writerow([unit1, unit2])
    
        # Write the data
        for x, y in zip(arr1, arr2):
            csv_writer.writerow([x, y])
    
    return None


def write_to_csv_threecols(arr1, header1, unit1, 
                           arr2, header2, unit2, 
                           arr3, header3, unit3,
                           csvfile_path):
    """
    Write two 1D arrays to a CSV file as three columns.  
    """
    
    # Writing to CSV file
    with open(csvfile_path, 'w', newline='') as csvfile:
        # Create a CSV writer object
        csv_writer = csv.writer(csvfile)
    
        # Write the header
        csv_writer.writerow([header1, header2, header3])
        
        # Write the unit
        csv_writer.writerow([unit1, unit2, unit3])
    
        # Write the data
        for x, y, z in zip(arr1, arr2, arr3):
            csv_writer.writerow([x, y, z])
    
    return None


def write_to_csv_multicols(arr: NDArray, headers: list[str], units: list[str], csvfile_path: str):
    """
    Write the arrays to a CSV file.  

    Parameters
    ----------
    arr: dimension N x m
    headers: length N
    units: length N
    csvfile_path : string, file path, ending in .csv

    Returns
    -------
    None.

    """
        
    with open(csvfile_path, 'w', newline='') as csvfile:
        # Create a CSV writer object
        csv_writer = csv.writer(csvfile)

        # Write the header
        csv_writer.writerow(headers)
        
        # Write the unit
        csv_writer.writerow(units)

        # Write the data
        # Basically, we need to write in the transpose of arr
        N, m = arr.shape
        for i in range(m):
            csv_writer.writerow(arr[:, i])
    
    return None



def read_from_csv_onecol(filepath: str) -> (list[float], str, str):
    """
    Read a CSV file with one column, output the header, unit, and the data.

    Parameters
    ----------
    filepath : file path ending in .csv. Only has one column, 1st row is header,
                    2nd row is unit. 
    """
    
    with open(filepath, 'r') as csvfile:
        # Create a CSV reader object
        csv_reader = csv.reader(csvfile)

        # Read the header
        header = next(csv_reader)[0]
        
        # Read the unit
        unit = next(csv_reader)[0]

        # Read the data into a list
        data = [float(row[0]) for row in csv_reader]

    return data, header, unit
    


def read_from_csv_twocols(filepath: str) -> (list[float], str, str, list[float], str, str):
    """
    Read a CSV file with two columns, output the data. 

    Parameters
    ----------
    filepath : file path, ending in .csv. Has two columns, 1st row is headers,
        2nd row is units. 
    """
    
    data1 = []
    data2 = []

    # Read from CSV file
    with open(filepath, 'r') as csvfile:
        # Create a CSV reader object
        csv_reader = csv.reader(csvfile)

        # Read the header row
        headers = next(csv_reader)

        # Extract header names
        header1, header2 = headers[0], headers[1]
        
        # Read the unit row
        units = next(csv_reader)
        
        # Extract the units
        unit1, unit2 = units[0], units[1]
        

        # Read the data into arrays
        for row in csv_reader:
            data1.append(float(row[0]))
            data2.append(float(row[1]))

    return data1, header1, unit1, data2, header2, unit2     


def read_from_csv_threecols(filepath: str) -> (list[float], str, str, 
                                               list[float], str, str,
                                               list[float], str, str):
    """
    Read a CSV file with three columns, output the data. 

    Parameters
    ----------
    filepath : file path, ending in .csv. Has three columns, 1st row is headers,
        2nd row is units. 
    """
    
    data1 = []
    data2 = []
    data3 = []

    # Read from CSV file
    with open(filepath, 'r') as csvfile:
        # Create a CSV reader object
        csv_reader = csv.reader(csvfile)

        # Read the header row
        headers = next(csv_reader)

        # Extract header names
        header1, header2, header3 = headers[0], headers[1], headers[2]
        
        # Read the unit row
        units = next(csv_reader)
        
        # Extract the units
        unit1, unit2, unit3 = units[0], units[1], units[2]
        

        # Read the data into arrays
        for row in csv_reader:
            data1.append(float(row[0]))
            data2.append(float(row[1]))
            data3.append(float(row[2]))

    return data1, header1, unit1, data2, header2, unit2, data3, header3, unit3


def read_from_csv_fourcols(filepath: str) -> (list[float], list[float], list[float], list[float]):
    """
    Read a CSV file with  four columns, output the data. 

    Parameters
    ----------
    filepath : file path, ending in .csv. Has four columns, 1st row is headers,
        2nd row is units. 
    """
    
    data1 = []
    data2 = []
    data3 = []
    data4 = []

    # Read from CSV file
    with open(filepath, 'r') as csvfile:
        # Create a CSV reader object
        csv_reader = csv.reader(csvfile)

        # Read the header row
        headers = next(csv_reader)

        # # Extract header names
        # header1, header2, header3, header4 = headers[0], headers[1]
        
        # Read the unit row
        units = next(csv_reader)
        
        # # Extract the units
        # unit1, unit2, unit3, unit4 = units[0], units[1]
        

        # Read the data into arrays
        for row in csv_reader:
            data1.append(float(row[0]))
            data2.append(float(row[1]))
            data3.append(float(row[2]))
            data4.append(float(row[3]))

    return data1, data2, data3, data4




def model_unit_table(model_name: str) -> dict[str, str]:
    """
    Given the model name, return the unit table of the model. 
    """
    
    unit_table = {}
    
    if model_name == '1TCM':
        unit_table = {'K1': 'mL/min/mL',
                      'k2': '/min',
                      'VB': 'unitless',
                      'VD': 'unitless'}
    
    elif model_name == '2TCM':
        unit_table = {'K1': 'mL/min/mL',
                      'k2': '/min',
                      'k3': '/min',
                      'k4': '/min',
                      'VB': 'unitless',
                      'VND': 'unitless',
                      'VS': 'unitless',
                      'VT': 'unitless',
                      'BPND': 'unitless'}
        
    elif model_name == 'Logan':
        unit_table = {'slope': 'unitless',
                      'intercept': 'min',
                      'tstart': 'min'}
    
    elif model_name == 'RTM':
        unit_table = {'R1': 'unitless',
                      'k2': '/min',
                      'k3': '/min',
                      'BPND': 'unitless',
                      'k4': '/min'}    

    elif model_name == 'SRTM':
        unit_table = {'R1': 'unitless',
                      'k2': '/min',
                      'BPND': 'unitless'}
    
    return unit_table






            
if __name__ == "__main__":
    
    # test
    
    # file_path = "/home/documents/mask_cerebellum.nii.gz"
    
    # base, extension = extract_file_name(file_path)
    # print(base)
    # print(extension)
    
    # env = Environment()
    # env.MRI_dir = '/zeyu/documents/Data/FEPPA/F301/Freesurfer/bert1/mri'
    # env.PET_dir = '/zeyu/documents/Data/FEPPA/F301/PET'
    # print(env.MRI_dir)
    # print(env.PET_dir)
    
    # header = 'concentration'
    # arr = [10, 20, 30, 40, 50]
    # write_to_csv_onecol(header, arr, '/Users/zeyuzhou/Documents/kinetic_modeling_test/FEPPA_20190523_AA_31002478/test.csv')
    
    
    # dur = [1, 2, 5, 6, 10]
    # FS = FrameSchedule(dur)
    # print(FS.durations)
    # print(FS.start_points)
    # print(FS.mid_points)
    
    # col1_data, col2_data = read_from_csv_twocols('/Users/zeyuzhou/Documents/kinetic_modeling_test/FEPPA_20190523_AA_31002478/AIF/arterial_plasma_tac.csv')
    # print(col1_data)
    # print(col2_data)
    
    
    f = np.array([1,1,2,1])
    t = np.array([1,2,4,8])
    intf = discrete_integrate(f, t)
    
    print('intf =', intf)
    
    pass
    
    
    
    