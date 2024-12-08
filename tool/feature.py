"""
Feature extraction functions. 
"""



import matlab.engine




def matlab_approxcanny(
        thresh: float,
        sigma: float, 
        infilepath: str,
        outfilepath: str,
        matlab_dir: str) -> None:
    """
    Apply the edge3-approxcanny algorithm in Matlab for edge detection. 
    
    thresh: high sensitivity threshold of the Canny algorithm. the low sensitivity threshold is
              set as 0.4*thresh
    sigma: standard deviation of the Gaussian smoothing filter. 
    matlab_dir: directory where the matlab function is located. 
    """

    eng = matlab.engine.start_matlab()
    eng.addpath(matlab_dir)
    eng.approxcanny(infilepath, outfilepath, thresh, sigma, nargout=0)
    
    return None
    


            
        
            
            
            
            