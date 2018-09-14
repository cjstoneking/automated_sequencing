#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
This is based on PyElastix by Almar Klein - but fairly heavily modified
(streamlined for our particular use case)

PyElastix details below:
    
# Copyright (c) 2010-2016, Almar Klein
# This code is subject to the MIT license

PyElastix - Python wrapper for the Elastix nonrigid registration toolkit

This Python module wraps the Elastix registration toolkit. For it to
work, the Elastix command line application needs to be installed on
your computer. You can obtain a copy at http://elastix.isi.uu.nl/.
Further, this module depends on numpy.

https://github.com/almarklein/pyelastix
"""

import os
import shutil
import subprocess
import numpy as np
import re


def _make_temp_folders(working_directory):
    if not os.path.isdir(os.path.join(working_directory, 'elastix_temp_files')):
        os.mkdir(os.path.join(working_directory, 'elastix_temp_files'))
    if not os.path.isdir(os.path.join(working_directory, 'elastix_temp_files', 'input')):
        os.mkdir(os.path.join(working_directory, 'elastix_temp_files', 'input'))
    if not os.path.isdir(os.path.join(working_directory, 'elastix_temp_files', 'output')):
        os.mkdir(os.path.join(working_directory, 'elastix_temp_files', 'output'))
        
def _clean_temp_folders(working_directory):
    if os.path.isdir(os.path.join(working_directory, 'elastix_temp_files')):
        shutil.rmtree(os.path.join(working_directory, 'elastix_temp_files'))
        
def _get_dtype_maps():
    """ Get dictionaries to map numpy data types to ITK types and the 
    other way around.
    """
    
    # Define pairs
    tmp = [ (np.float32, 'MET_FLOAT'),  (np.float64, 'MET_DOUBLE'),
            (np.uint8, 'MET_UCHAR'),    (np.int8, 'MET_CHAR'),
            (np.uint16, 'MET_USHORT'),  (np.int16, 'MET_SHORT'),
            (np.uint32, 'MET_UINT'),    (np.int32, 'MET_INT'),
            (np.uint64, 'MET_ULONG'),   (np.int64, 'MET_LONG') ]
    
    # Create dictionaries
    map1, map2 = {}, {}
    for np_type, itk_type in tmp:
        map1[np_type.__name__] = itk_type
        map2[itk_type] = np_type.__name__
    
    # Done
    return map1, map2

DTYPE_NP2ITK, DTYPE_ITK2NP = _get_dtype_maps()

class Image(np.ndarray):
    
    def __new__(cls, array):
        try:
            ob = array.view(cls)
        except AttributeError:  # pragma: no cover
            # Just return the original; no metadata on the array in Pypy!
            return array
        return ob
    
def _write_image_data(im, id, working_directory):
    """ Write a numpy array to disk in the form of a .raw and .mhd file.
    The id is the image sequence number (1 or 2).
    """
    im = im* (1.0/3000)
    # Create text
    lines = [   "ObjectType = Image",
                "NDims = <ndim>",
                "BinaryData = True",
                "BinaryDataByteOrderMSB = False",
                "CompressedData = False",
                #"TransformMatrix = <transmatrix>",
                "Offset = <origin>",
                "CenterOfRotation = <centrot>",
                "ElementSpacing = <sampling>",
                "DimSize = <shape>",
                "ElementType = <dtype>",
                "ElementDataFile = <fname>",
                "" ]
    text = '\n'.join(lines)
    
    # Determine file names
    tempdir= os.path.join(working_directory, 'elastix_temp_files', 'input')
    fname_raw_ = 'im%i.raw' % id
    fname_raw = os.path.join(tempdir, fname_raw_)
    fname_mhd = os.path.join(tempdir, 'im%i.mhd' % id)
    
    # Get shape, sampling and origin
    shape = im.shape
    if hasattr(im, 'sampling'): sampling = im.sampling
    else: sampling = [1 for s in im.shape]
    if hasattr(im, 'origin'): origin = im.origin
    else: origin = [0 for s in im.shape]
    
    # Make all shape stuff in x-y-z order and make it string
    shape = ' '.join([str(s) for s in reversed(shape)])
    sampling = ' '.join([str(s) for s in reversed(sampling)])
    origin = ' '.join([str(s) for s in reversed(origin)])
    
    # Get data type
    dtype_itk = DTYPE_NP2ITK.get(im.dtype.name, None)
    if dtype_itk is None:
        raise ValueError('Cannot convert data of this type: '+ str(im.dtype))
    
    # Set mhd text
    text = text.replace('<fname>', fname_raw)
    text = text.replace('<ndim>', str(im.ndim))
    text = text.replace('<shape>', shape)
    text = text.replace('<sampling>', sampling)
    text = text.replace('<origin>', origin)
    text = text.replace('<dtype>', dtype_itk)
    text = text.replace('<centrot>', ' '.join(['0' for s in im.shape]))
    if im.ndim==2:
        text = text.replace('<transmatrix>', '1 0 0 1')
    elif im.ndim==3:
        text = text.replace('<transmatrix>', '1 0 0 0 1 0 0 0 1')
    elif im.ndim==4:
        pass # ???
    
    # Write data file
    f = open(fname_raw, 'wb')
    try:
        f.write(im.data)
    finally:
        f.close()
    
    # Write mhd file
    f = open(fname_mhd, 'wb')
    try:
        f.write(text.encode('utf-8'))
    finally:
        f.close()
    


def _read_image_data(working_directory, fname):

    des = open(os.path.join(working_directory, fname), 'r').read()
    
    # Get data filename and load raw data
    match = re.findall('ElementDataFile = (.+?)\n', des)
    p = os.path.join(working_directory, match[0])
    data = open(p, 'rb').read()
    
    # Determine dtype
    match = re.findall('ElementType = (.+?)\n', des)
    dtype_itk = match[0].upper().strip()
    dtype = DTYPE_ITK2NP.get(dtype_itk, None)
    if dtype is None:
        raise RuntimeError('Unknown ElementType: ' + dtype_itk)
    
    # Create numpy array
    a = np.frombuffer(data, dtype=dtype)
    
    # Determine shape, sampling and origin of the data
    match = re.findall('DimSize = (.+?)\n', des)
    shape = [int(i) for i in match[0].split(' ')]
    #
    match = re.findall('ElementSpacing = (.+?)\n', des)
    sampling = [float(i) for i in match[0].split(' ')]
    #
    match = re.findall('Offset = (.+?)\n', des)
    origin = [float(i) for i in match[0].split(' ')]
    
    # Reverse shape stuff to make z-y-x order
    shape = [s for s in reversed(shape)]
    sampling = [s for s in reversed(sampling)]
    origin = [s for s in reversed(origin)]
    
    # Take vectors/colours into account
    N = np.prod(shape)
    if N != a.size:
        extraDim = int( a.size / N )
        shape = tuple(shape) + (extraDim,)
        sampling = tuple(sampling) + (1.0,)
        origin = tuple(origin) + (0,)
    
    # Check shape
    N = np.prod(shape)
    if N != a.size:
        raise RuntimeError('Cannot apply shape to data.')
    else:
        a.shape = shape
        a = Image(a)
        a.sampling = sampling
        a.origin = origin
    return a


def _write_parameter_file(params, working_directory):

    # Get path
    path = os.path.join(working_directory, 'elastix_temp_files', 'input', 'params.txt')
    
    # Define helper function
    def valToStr(val):
        if val in [True, False]:
            return '"%s"' % str(val).lower()
        elif isinstance(val, int):
            return str(val)
        elif isinstance(val, float):
            tmp = str(val)
            if not '.' in tmp:
                tmp += '.0'
            return tmp
        elif isinstance(val, str):
            return '"%s"' % val
    
    # Compile text
    text = ''
    for key in params:
        val = params[key]
        # Make a string of the values
        if isinstance(val, (list, tuple)):
            vals = [valToStr(v) for v in val]
            val_ = ' '.join(vals)
        else:
            val_ = valToStr(val)
        # Create line and add
        line = '(%s %s)' % (key, val_)
        text += line + '\n'
    
    # Write text
    f = open(path, 'wb')
    try:
        f.write(text.encode('utf-8'))
    finally:
        f.close()
    

class Parameters:
    """ Struct object to represent the parameters for the Elastix
    registration toolkit. Sets of parameters can be combined by
    addition. (When adding `p1 + p2`, any parameters present in both
    objects will take the value that the parameter has in `p2`.)
    
    Use `get_default_params()` to get a Parameters struct with sensible
    default values.
    """
    
    def as_dict(self):
        """ Returns the parameters as a dictionary. 
        """
        tmp = {}
        tmp.update(self.__dict__)
        return tmp
    
    def __repr__(self):
        return '<Parameters instance with %i parameters>' % len(self.__dict__)
    
    def __str__(self):
        
        # Get alignment value
        c = 0
        for key in self.__dict__:
            c = max(c, len(key))
        
        # How many chars left (to print on less than 80 lines)
        charsLeft = 79 - (c+6)
        
        s = '<%i parameters>\n' % len(self.__dict__)
        for key in self.__dict__.keys():
            valuestr = repr(self.__dict__[key])
            if len(valuestr) > charsLeft:
                valuestr = valuestr[:charsLeft-3] + '...'
            s += key.rjust(c+4) + ": %s\n" % (valuestr)
        return s
    
    def __add__(self, other):
        p = Parameters()
        p.__dict__.update(self.__dict__)
        p.__dict__.update(other.__dict__)
        return p


def _get_fixed_params(im):
    """ Parameters that the user has no influence on. Mostly chosen
    bases on the input images.
    """
    
    p = Parameters()
    
    if not isinstance(im, np.ndarray):
        return p
    
    # Dimension of the inputs
    p.FixedImageDimension = im.ndim
    p.MovingImageDimension = im.ndim
    
    # Always write result, so I can verify
    p.WriteResultImage = True
    
    # How to write the result
    tmp = DTYPE_NP2ITK[im.dtype.name]
    p.ResultImagePixelType = tmp.split('_')[-1].lower()
    p.ResultImageFormat = "mhd"
    
    # Done
    return p


def get_advanced_params():
    """ Get `Parameters` struct with parameters that most users do not
    want to think about.
    """
    
    p = Parameters()
    
    # Internal format used during the registration process
    p.FixedInternalImagePixelType = "float"
    p.MovingInternalImagePixelType = "float"
    
    # Image direction
    p.UseDirectionCosines = True
    
    # In almost all cases you'd want multi resolution
    p.Registration = 'MultiResolutionRegistration'
    
    # Pyramid options
    # *RecursiveImagePyramid downsamples the images
    # *SmoothingImagePyramid does not downsample
    p.FixedImagePyramid = "FixedRecursiveImagePyramid"
    p.MovingImagePyramid = "MovingRecursiveImagePyramid"
    
    # Whether transforms are combined by composition or by addition.
    # It does not influence the results very much.
    p.HowToCombineTransforms = "Compose"
    
    # For out of range pixels
    p.DefaultPixelValue = 0
    
    # Interpolator used during interpolation and its order
    # 1 means linear interpolation, 3 means cubic.
    p.Interpolator = "BSplineInterpolator"
    p.BSplineInterpolationOrder = 1
    
    # Interpolator used during interpolation of final level, and its order
    p.ResampleInterpolator = "FinalBSplineInterpolator"
    p.FinalBSplineInterpolationOrder = 3
    
    # According to the manual, there is currently only one resampler
    p.Resampler = "DefaultResampler"
    
    # Done
    return p


def get_default_params(type='BSPLINE'):
    """ get_default_params(type='BSPLINE')
    
    Get `Parameters` struct with parameters that users may want to tweak.
    The given `type` specifies the type of allowed transform, and can
    be 'RIGID', 'AFFINE', 'BSPLINE'.
    
    For detail on what parameters are available and how they should be used,
    we refer to the Elastix documentation. Here is a description of the
    most common parameters:
    
    * Transform (str):
        Can be 'BSplineTransform', 'EulerTransform', or
        'AffineTransform'. The transformation to apply. Chosen based on `type`.
    * FinalGridSpacingInPhysicalUnits (int):
        When using the BSplineTransform, the final spacing of the grid.
        This controls the smoothness of the final deformation.
    * AutomaticScalesEstimation (bool):
        When using a rigid or affine transform. Scales the affine matrix
        elements compared to the translations, to make sure they are in
        the same range. In general, it's best to use automatic scales
        estimation.
    * AutomaticTransformInitialization (bool):
        When using a rigid or affine transform. Automatically guess an
        initial translation by aligning the geometric centers of the 
        fixed and moving.
    * NumberOfResolutions (int):
        Most registration algorithms adopt a multiresolution approach
        to direct the solution towards a global optimum and to speed
        up the process. This parameter specifies the number of scales
        to apply the registration at. (default 4)
    * MaximumNumberOfIterations (int):
        Maximum number of iterations in each resolution level.
        200-2000 works usually fine for nonrigid registration.
        The more, the better, but the longer computation time.
        This is an important parameter! (default 500).
    """
    
    # Init
    p = Parameters()
    type = type.upper()
    
    
    # ===== Metric to use =====
    p.Metric = 'AdvancedMattesMutualInformation'
    
    # Number of grey level bins in each resolution level,
    # for the mutual information. 16 or 32 usually works fine.
    # sets default value for NumberOf[Fixed/Moving]HistogramBins
    p.NumberOfHistogramBins = 32
    
    # Taking samples for mutual information
    p.ImageSampler = 'RandomCoordinate'
    p.NumberOfSpatialSamples = 2048
    p.NewSamplesEveryIteration = True
    
    
    # ====== Transform to use ======
    
    # The number of levels in the image pyramid
    p.NumberOfResolutions = 4
    
    if type in ['B', 'BSPLINE', 'B-SPLINE']:
        
        # Bspline transform
        p.Transform = 'BSplineTransform'
        
        # The final grid spacing (at the smallest level)
        p.FinalGridSpacingInPhysicalUnits = 16
    
    if type in ['RIGID', 'EULER', 'AFFINE']:
        
        # Affine or Euler transform
        if type in ['RIGID', 'EULER']:
            p.Transform = 'EulerTransform'
        else:
            p.Transform = 'AffineTransform'
        
        # Scales the affine matrix elements compared to the translations, 
        # to make sure they are in the same range. In general, it's best to
        # use automatic scales estimation.
        p.AutomaticScalesEstimation = True
        
        # Automatically guess an initial translation by aligning the
        # geometric centers of the fixed and moving.
        p.AutomaticTransformInitialization = True
    
    
    # ===== Optimizer to use =====
    p.Optimizer = 'AdaptiveStochasticGradientDescent'
    
    # Maximum number of iterations in each resolution level:
    # 200-2000 works usually fine for nonrigid registration.
    # The more, the better, but the longer computation time.
    # This is an important parameter!
    p.MaximumNumberOfIterations = 500
    
    # The step size of the optimizer, in mm. By default the voxel size is used.
    # which usually works well. In case of unusual high-resolution images
    # (eg histology) it is necessary to increase this value a bit, to the size
    # of the "smallest visible structure" in the image:
    #p.MaximumStepLength = 1.0 Default uses voxel spaceing
    
    # Another optional parameter for the AdaptiveStochasticGradientDescent
    #p.SigmoidInitialTime = 4.0
    
    
    # ===== Also interesting parameters =====
    
    #p.FinalGridSpacingInVoxels = 16
    #p.GridSpacingSchedule = [4.0, 4.0, 2.0, 1.0]
    #p.ImagePyramidSchedule = [8 8  4 4  2 2  1 1]
    #p.ErodeMask = "false"
    
    # Done
    return p


def _compile_params(params, im1):
    """ Compile the params dictionary:
    * Combine parameters from different sources
    * Perform checks to prevent non-compatible parameters
    * Extend parameters that need a list with one element per dimension
    """
    
    # Compile parameters
    p = _get_fixed_params(im1) + get_advanced_params()
    p = p + params
    params = p.as_dict()
    
    # Check parameter dimensions
    if isinstance(im1, np.ndarray):
        lt = (list, tuple)
        for key in [    'FinalGridSpacingInPhysicalUnits',
                        'FinalGridSpacingInVoxels' ]:
            if key in params.keys() and not isinstance(params[key], lt):
                params[key] = [params[key]] * im1.ndim
    
    # Check parameter removal
    if 'FinalGridSpacingInVoxels' in params:
        if 'FinalGridSpacingInPhysicalUnits' in params:
            params.pop('FinalGridSpacingInPhysicalUnits')
    
    # Done
    return params


def register(im1, im2, params, working_directory, exact_params=False):
    """ register(im1, im2, params, exact_params=False, verbose=1)
    
    Perform the registration of `im1` to `im2`, using the given 
    parameters. Returns `(im1_deformed, field)`, where `field` is a
    tuple with arrays describing the deformation for each dimension
    (x-y-z order, in world units).
    
    Parameters:
    
    * im1 (ndarray or file location):
        The moving image (the one to deform).
    * im2 (ndarray or file location):
        The static (reference) image.
    * params (dict or Parameters):
        The parameters of the registration. Default parameters can be
        obtained using the `get_default_params()` method. Note that any
        parameter known to Elastix can be added to the parameter
        struct, which enables tuning the registration in great detail.
        See `get_default_params()` and the Elastix docs for more info.
    * exact_params (bool):
        If True, use the exact given parameters. If False (default)
        will process the parameters, checking for incompatible
        parameters, extending values to lists if a value needs to be
        given for each dimension.
    * verbose (int):
        Verbosity level. If 0, will not print any progress. If 1, will
        print the progress only. If 2, will print the full output
        produced by the Elastix executable. Note that error messages
        produced by Elastix will be printed regardless of the verbose
        level.
    """
    
    _clean_temp_folders(working_directory)
    _make_temp_folders(working_directory)
    
    # Reference image
    refIm = im1
    if isinstance(im1, (tuple,list)):
        refIm = im1[0]
    
    # Check parameters
    if not exact_params:
        params = _compile_params(params, refIm)
    if isinstance(params, Parameters):
        params = params.as_dict()
    
    # Groupwise?
    if im2 is None:
        # todo: also allow using a constraint on the "last dimension"
        if not isinstance(im1, (tuple,list)):
            raise ValueError('im2 is None, but im1 is not a list.')
        #
        ims = im1
        ndim = ims[0].ndim
        # Create new image that is a combination of all images
        N = len(ims)
        new_shape = (N,) + ims[0].shape
        im1 = np.zeros(new_shape, ims[0].dtype)
        for i in range(N):
            im1[i] = ims[i]
        # Set parameters
        #params['UseCyclicTransform'] = True # to be chosen by user
        params['FixedImageDimension'] = im1.ndim
        params['MovingImageDimension'] = im1.ndim
        params['FixedImagePyramid'] = 'FixedSmoothingImagePyramid'
        params['MovingImagePyramid'] = 'MovingSmoothingImagePyramid'
        params['Metric'] = 'VarianceOverLastDimensionMetric'
        params['Transform'] = 'BSplineStackTransform'
        params['Interpolator'] = 'ReducedDimensionBSplineInterpolator'
        params['SampleLastDimensionRandomly'] = True
        params['NumSamplesLastDimension'] = 5
        params['SubtractMean'] = True
        # No smoothing along that dimenson
        pyramidsamples = []
        for i in range(params['NumberOfResolutions']):
            pyramidsamples.extend( [0]+[2**i]*ndim )
        pyramidsamples.reverse()
        params['ImagePyramidSchedule'] = pyramidsamples
    
    #write input images 
    _write_image_data(im1, 1, working_directory)
    _write_image_data(im2, 2, working_directory)
    
    #write parameter file
    _write_parameter_file(params, working_directory)
    
    # Get path of trafo param file
    #path_trafo_params = os.path.join(tempdir, 'TransformParameters.0.txt')
    
    # Register

    os.chdir(working_directory)
    subprocess.call(['sudo', 'bash', 'elastix_wrapper_script.sh'])

    #if elastix worked, it will have output a registered image
    try:
        registered_image = _read_image_data(os.path.join(working_directory,'elastix_temp_files', 'output'), 'result.0.mhd')
    except IOError as err:
        error_msg = "Elastix-based image registration failed"
        #if we can't find an image, assume this is because elastix did not produce one
        raise RuntimeError(error_msg)
    
    _clean_temp_folders(working_directory)

    return registered_image