"""
arterial.py

Functions and classes for processing arterial input function (AIF) and blood-related data
in dynamic PET studies. This includes fitting input functions, modeling blood activity,
and estimating plasma/blood concentrations using standard models.

Includes:
- TimedPoints: Class for managing time-series data with fitting and plotting utilities
- BloodInput: Class for estimating CP and CB from metabolite data or blood TAC
- Fitting models: linear_2exp, linear_3exp, zero_linear_3exp, Hill, twoExp, etc.


Author: Zeyu Zhou
Date: 2025-05-22
"""

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from collections.abc import Callable
from numpy.typing import NDArray


from .utils import auxiliary as aux
from .frameschedule import FrameSchedule
from .tac import TAC


# An array of numbers with time stamps
class TimedPoints:
    def __init__(self, 
                 name: str, 
                 data: NDArray, 
                 unit: str, 
                 t: NDArray, 
                 t_unit: str):
        
        self.name = name   # name of the quantity
        
        self.t = t     # time stamps
        self.data = data     # values of quantity
        
        self.t_unit = t_unit     # unit of time
        self.unit = unit    # unit of measured quantity
        
        self.f_to_fit = None   # Callable: time+params -> float
        self.f_fitted = None   # Callable: time -> float
        self.fitfuncname = None  # str, name of the fit function
        self.fit_params = None   # numpy.ndarray, parameters of the fitting function
        
    
    @classmethod
    def from_file(cls, filepath: str, name: str):
        """
        Initialize a TimedPoints instance from a two-column CSV file.
        Assumes columns are: time and value.
        """
        
        t, t_header, t_unit, data, header, unit = aux.read_from_csv_twocols(filepath)
        t = np.array(t)
        data = np.array(data)
        
        return cls(name, data, unit, t, t_unit)
        

    def plot(self, xlim: list[float] = None, ylim: list[float] = None) -> None:
        """
        Plot the data and fitted function. 
        """
        
        plt.figure()
        if self.t != [] and self.data != []: 
            plt.scatter(self.t, self.data, c='blue', label='data')
        if self.t != [] and self.f_fitted != None and self.fit_params.all() != None:
            tmax = np.max(self.t)
            tfit = np.linspace(0, tmax*1.1, 1000)
            yfit = self.f_fitted(tfit)
            plt.plot(tfit, yfit, c='red', label='fit')
        plt.xlabel(f't ({self.t_unit})')
        plt.ylabel(f'{self.y_unit}')
        plt.title(f'{self.name}')
        
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
    
    
    def fit(self,
            f: Callable[..., float],
            fname: str,
            bounds: tuple | None = None, 
            ) -> None:
        """
        Fit the points to a model function using non-linear least squares. 
        Stores the fitted function and parameters in the object. 

        Parameters
        ----------
        f : function to be fitted, first argument of f must be time, the remaining arguments are parameters 
        fname : name of f 
        bounds : (optional) bounds on the parameters of f, see
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html for format
        
        """
        
        self.fitfuncname = fname
        self.f_to_fit = f
        
        if bounds is None:
            self.fit_params, _ = curve_fit(f, self.t, self.data)
        else:
            self.fit_params, _ = curve_fit(f, self.t, self.data, bounds=bounds)
            
        def f_fitted(t):
            return self.f_to_fit(t, *self.fit_params)
        self.f_fitted = f_fitted
        
        return None
    
    
    def print_fitparams(self):
        """
        Print the fitted parameters. 
        """
        
        print('Fitting parameters:\n')
        print(self.fitparams)
        return None




class BloodInput:
    
    # CP: plasma concentration
    # CB: blood concentration
    
    def __init__(self,
                 CP: Callable[..., float],
                 CB: Callable[..., float],
                 t: NDArray,
                 unit: str,
                 t_unit: str):
        
        self.t = t
        self.unit = unit
        self.t_unit = t_unit
        
        self.CP = CP
        self.CB = CB
        
        
    @classmethod
    def from_metabolite_files(cls,
                              ptac_path: str,
                              pif_path: str,
                              p2wb_ratio_path: str,
                              t: NDArray,
                              ) -> None:
        """
        Construct a BloodInput object from metabolite data CSVs.
        Fits intact fraction and plasma-to-blood conversion.
        """
        
        # plasma concentration tac (before metabolite correction)
        ptac = TimedPoints.from_file(ptac_path, 'Plasma Activity Concentration')
        
        if ptac.unit == 'kBq/mL':
            pass
        elif ptac.unit == 'Bq/mL':
            ptac.data = ptac.data / 1000.0
            ptac.unit = 'kBq/mL'
        
        ptac.fit(f=zero_linear_3exp, fname='zero_linear_3exp')
        
        # plasma intact fraction
        pif = TimedPoints.from_file(pif_path, 'Plasma Intact Fraction')
        pif.fit(f=Hill, fname='Hill', bounds=Hill_bounds())
        
        
        # wb2plasma concentration ratio
        p2wb = TimedPoints.from_file(p2wb_ratio_path, 'Ratio of plasma-to-wholeblood activity concentration')
        
        wb2p = TimedPoints(name = 'Ratio of wholeblood-to-plasma activity concentration',
                           data = 1.0 / p2wb.data,
                           unit = p2wb.unit,
                           t = p2wb.t,
                           t_unit = p2wb.t_unit)
        wb2p.fit(f=twoExp, fname='twoExp', bounds=twoExp_bounds())
        
        
        CP = lambda t: ptac.f_fitted(t) * pif.f_fitted(t)
        CB = lambda t: ptac.f_fitted(t) * wb2p.f_fitted(t)
        
        
        return cls(CP = CP,
                   CB = CB,
                   t = t,
                   unit = ptac.unit,
                   t_unit = ptac.t_unit)
    
        
    @classmethod
    def from_blood_tac(cls, 
                       tac: TAC | str,
                       fs: FrameSchedule,
                       p0: tuple | None = None):
        """
        Construct a BloodInput object by fitting a blood TAC file or TAC object.
        Used when plasma data is not available.
        """
        
        
        if isinstance(tac, TAC):
            blood_tac = tac
        else:
            blood_tac = TAC.from_file(fs = fs, 
                                      tacfile_path = tac)
        
        # bounds = ((0, 0, 0, 0, 0, 0, 0),
        #           (np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf))
        
        # bounds = ((0, 0, 0, 0, 0),
        #           (np.inf, np.inf, np.inf, np.inf, np.inf))
        
        # bounds = None
        
        # fitted_params = blood_tac.fit(model = linear_3exp,
        #                               p0 = p0,
        #                               bounds = bounds)
        
        # print("ICA fitting:")
        
        # a, Tpk, A1, lamb1, A2, lamb2, lamb3 = fitted_params
        # print(f"a = {a}")
        # print(f"Tpk = {Tpk}")
        # print(f"A1 = {A1}")
        # print(f"lamb1 = {lamb1}")
        # print(f"A2 = {A2}")
        # print(f"lamb2 = {lamb2}")
        # print(f"lamb3 = {lamb3}")
        
        
        # a, Tpk, A1, lamb1, lamb2 = fitted_params
        # print(f"a = {a}")
        # print(f"Tpk = {Tpk}")
        # print(f"A1 = {A1}")
        # print(f"lamb1 = {lamb1}")
        # print(f"lamb2 = {lamb2}")
        
        #blood_tac.plot_with_fitting()
        
        CB = blood_tac.fitted_func
        
        return cls(CP = None,
                   CB = CB,
                   t = blood_tac.t,
                   unit = blood_tac.unit,
                   t_unit = blood_tac.t_unit)
    
    

    def plot(self, trange: list[float]) -> None:
        """
        len(trange) = 2
        """

        ts = np.linspace(trange[0], trange[1], 1000)
        
        if self.CP is not None:
            cps = self.CP(ts)
                
            plt.figure(1)
            plt.plot(ts, cps)
            plt.xlabel(f't ({self.t_unit})')
            plt.ylabel(f'{self.unit}')
            plt.title(r'$C_P(t)$)')
            plt.show()
        
        if self.CB is not None:
            cbs = self.CB(ts)
        
            plt.figure(2)
            plt.plot(ts, cbs)
            plt.xlabel(f't ({self.t_unit})')
            plt.ylabel(f'{self.unit}')
            plt.title(r'$C_B(t)$)')
            plt.show()

        return None


def linear_2exp(t: float | np.ndarray, 
                a, Tpk, A1, lamb1, lamb2
                ) -> float | np.ndarray:
    """
    Piecewise function for blood activity modeling:
    - Linear ramp up to time Tpk
    - Sum of two exponentials after Tpk
    
    f(t) = a * t,                if t < Tpk
         = A1 * exp(-lamb1*(t-Tpk)) + A2 * exp(-lamb2*(t-Tpk)), if t ≥ Tpk
    
    Where A2 = a * Tpk - A1 (ensures continuity)
    """
    
    A2 = a * Tpk - A1
    
    return ( 
        ((t>=0)&(t<Tpk)) * (a * t) + 
        (t>=Tpk) * (A1 * np.exp(-lamb1*(t-Tpk)) + A2 * np.exp(-lamb2*(t-Tpk))))



def linear_3exp(t: float | np.ndarray, 
                a, Tpk, A1, lamb1, A2, lamb2, lamb3,
                ) -> float | np.ndarray:
    """
    Piecewise function for blood activity modeling:
    - Linear ramp up to time Tpk
    - Sum of three exponentials after Tpk
    
    f(t) = a * t, if t < Tpk
         = A1 * exp(-lamb1*(t-Tpk)) + A2 * exp(-lamb2*(t-Tpk)) + A3 * exp(-lamb3*(t-Tpk)), if t ≥ Tpk
    
    Where A3 = a * Tpk - (A1 + A2) (ensures continuity)
    """
    
    A3 = a * Tpk - (A1 + A2)
    return ( 
        ((t>=0)&(t<Tpk)) * (a * t) + 
        (t>=Tpk) * (A1*np.exp(-lamb1*(t-Tpk)) + A2*np.exp(-lamb2*(t-Tpk)) + A3*np.exp(-lamb3*(t-Tpk))))



def zero_linear_3exp(t: float | np.ndarray, 
                     a, b, Tpk, A1, lamb1, A2, lamb2, lamb3,
                     ) -> float | np.ndarray:
    """
    Extended model that includes a delay and intercept before linear ramp:
    - Flat zero until -b/a
    - Linear ramp from -b/a to Tpk
    - Three exponentials after Tpk
        
    f(t) = 0      if  t < -b/a
         = a * t + b  if -b/a <= t < Tpk
         = A1 * exp{-lamb1*(t-Tpk)} + A2 * exp{-lamb2*(t-Tpk)} + A3 * exp{-lamb3*(t-Tpk)}  if t >= Tpk
    
    The parameters should satisfy:
        a * Tpk + b = A1 + A2 + A3
    
    Source: http://www.turkupetcentre.net/petanalysis/input_fitting_exp.html
    """
    
    A3 = a * Tpk + b - (A1 + A2)
    return (
        ((t>=0)&(t<-b/a)) * 0  +  
        ((t>=-b/a)&(t<Tpk)) * (a * t + b) + 
        (t>=Tpk) * (A1*np.exp(-lamb1*(t-Tpk)) + A2*np.exp(-lamb2*(t-Tpk)) + A3*np.exp(-lamb3*(t-Tpk))) )



def Hill(t: float | np.ndarray, a, b, c) -> float | np.ndarray:
    """
    Hill-type sigmoid function:
    
    f(t) = 1 - (1-a)t^b/(c+t^b)
    
    Constraints:
        0 <= a <= 1
        b >= 1
        c > 0
        
    This is a decreasing curve with values between 0 and 1. 
    Eventually, it stays at constant level a
    
    Source: http://www.turkupetcentre.net/petanalysis/input_parent_fitting_hill.html#:~:text=Hill%20type%20functions%20have%20been,of%20parent%20radiotracer%20in%20plasma.&text=%2C%20where%200%20%E2%89%A4%20a%20%E2%89%A4,parameter%20a%20the%20final%20level.
    
    There is a version of the Hill function with two more parameters d and e
    (see Source above), maybe useful for other situations. 
    """
    
    return (1 - (1-a)*t**b/(c+t**b))


def Hill_bounds() -> tuple:
    """
    Returns bounds on the parameters a, b, c of the Hill function. 
    """
    
    # See details in the Hill function description
    bounds = ([0, 1,      0,    ],
              [1, np.inf, np.inf])
    
    return bounds
                


def twoExp(t, r1, r2, r3, r4) -> float:
    """
    Sum of two exponentials used to fit plasma-to-blood or blood-to-plasma ratio:

    f(t) = r1 * exp(-r3*t) + r2 * (1 - exp(-r4*t))

    Both r3 and r4 represent decay rates.
    
    Constraints for all parameters: (0, +inf)
    
    Source: http://www.turkupetcentre.net/petanalysis/input_blood-to-plasma_fitting.html
    """
    
    return r1 * np.exp(-r3*t) + r2 * (1-np.exp(-r4*t))



def twoExp_bounds() -> tuple:
    """
    Returns bounds on the parameters r1, r2, r3, r4 of the twoExp function. 
    """
    
    bounds = ([0,      0,      0,    0],
              [np.inf, np.inf, np.inf, np.inf])
    
    return bounds


def oneExp(t, rmin, rmax, rate) -> float:
    """
    Single exponential decay model:

    f(t) = rmin + (rmax - rmin) * exp(-rate * t)

    When t = 0, f(t) = rmax
    When t = +inf, f(t) = rmin    
    
    Not currently in use but available for potential alternative fits.
    """

    return rmin + (rmax-rmin)*np.exp(-rate*t)


def oneExp_bounds() -> tuple:
    """
    Returns bounds on the parameters rmin, rmax, rate of the oneExp function. 
    """
    
    bounds = ([0,      0,      0],
              [np.inf, np.inf, np.inf])
    
    return bounds

    
    



        
        
    
    
    