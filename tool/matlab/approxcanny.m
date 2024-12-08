%   approxcanny: gerenate the edges of a given image using Approximate Canny's 
%           edge detection algorithm. 
%
%   Usage: approxcanny(infilepath, outfilepath, thresh, sigma);
%   
%   infilepoath - file path of the input image, must be .nii or .nii.gz
%   outfilepath - file path of the output image of edges, must be .nii or
%                 .nii.gz
%   thresh - numeric scalar, high sensitivity threshold. the low sensitivity threshold is
%            set as 0.4*thresh
%            Edges with gradient values above the high threshold are automatically
%            included, those below the low threshold are automatically
%            excluded, while those in the middle are included only if they
%            are connected to included edges -- hysteresis thresholding. 
%   sigma - numeric scalar, standard deviation of the Gaussian smoothing filter. 
%           For large sigma, smoothing is more aggresive and it is more
%           likely for edges of fine details to be left out. 
%           For small sigma, smoothing is less visible, more edges can be
%           detected, but noisy edges may also be included. 
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [] = approxcanny(infilepath, outfilepath, thresh, sigma)

    in_nii = load_untouch_nii(infilepath);

    img = in_nii.img;
    
    % class(edges) = logical
    edges = edge3(img, 'approxcanny', thresh, sigma);

    % montage(edges);

    % change class of edges to single
    edges = single(edges);
    
    % create output nii file
    out_nii = in_nii;
    out_nii.img = edges;

    save_untouch_nii(out_nii, outfilepath);

end


