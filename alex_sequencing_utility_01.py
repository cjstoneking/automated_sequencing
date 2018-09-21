## UTILITY FUNCTIONS

from __future__ import division

############################## I/O ###################################

def save_tile_data(tile_dir,data_fname,dict_to_save,verbose=False,do_overwrite=False):
    # Save tile data (any variable, inside a dict) as variables in an hdf5
    # tile_dir      :  
    # fname         :  
    # dict_to_save  : 
    # sub_dir       : 
    # verbose       : 
    # do_overwrite  : True/False - Make new file?
    import h5py
    import os
    import numpy as np
    import pickle

    if verbose:
        import sys
        import pprint as pp
    
    # Recursive dict assignment
    def dict_to_hdf5(this_group,this_dict):
        for key,val in this_dict.items():
            if verbose: 
                print '\nWriting %s (size %g bytes) as' % (key,sys.getsizeof(val))
                #pp.pprint(val)

            # If value is a dict, make a subgroup for dict members
            if type(val) == type({}):
                if key not in this_group:
                    # Make a new group for this key
                    new_group = this_group.create_group(key)
                else:
                    # Already exists
                    new_group = this_group[key]
                new_group = dict_to_hdf5(new_group,val)
            else:
                # Ensure that the variable doesn't already exist
                try:
                    del this_group[key]
                except:
                    pass
                # Dump as a pickled bytestring
                if type(val) is np.ndarray:
                    if verbose: print '\tSaving numpy array  < %s > as raw array' % (key)
                    # Dump ndarrays raw
                    try:
                        this_group.create_dataset(key,data=val)
                    except RuntimeError:
                        # Might already exist
                        this_group.create_dataset(key,data=val)

                else:
                    s = pickle.dumps(val)
                    if verbose:
                        print '\tPICKLING :: ',key,'\n<<<<\n',s[:200].replace('\n', ' \\n '),' \n>>>...etc'
                    this_group.create_dataset(key,data=np.array(s))

                    # OLD VERSION
                    # Dumps a pickled string.  I think this is limited by total size
                    # If I'm honest here I don't know why I'm transforming this to a string representation (April 8, 2018)
                    # : this_group.create_dataset(key,data=np.array(s).astype('|S9'))




        return this_group
    
    # Open file and recusively asssign dicts
    full_path = os.path.join(tile_dir,data_fname)
    if do_overwrite:
        with h5py.File(full_path, 'w') as f:
            dict_to_hdf5(f,dict_to_save)
    else:
        with h5py.File(full_path, 'a') as f:
            dict_to_hdf5(f,dict_to_save)

def load_hdf5_data(tile_dir,data_fname,variable_to_load=None,sub_dir=None,verbose=False):
    """
    Load tile data from hdf5 file.  This data is saved as one file per tile, with variables represented hierarchically.

    Depending on "variable_to_load", you can request either a single variable (which will return its value) or a 
    dict of multiple values (which correspond to HDF5 groups).

    *All* data is pickled before saving, to get rid of issues with datatypes etc.

    __INPUTS__
    tile_dir         :: 
    tile_label       :: Tile label of format '001_001'. This can be a *full file* path if you want, but can also just be a tile label. Will search for files ending in 001_001.hdf5 or simialr.
    variable_to_load :: can be a single string 'data_array' or nested path '/params/parameter_three'.  If None or empty, returns the full content of the HDF5 as a dict.
    sub_dir          :: if supplied, sub_dir is appended onto tile_dir
    verbose          :: prints some info during processing

    __OUTPUTS__
    output           :: The requested data, either in its original format (if a single HDF5 dataset), or 
    """

    # Load tile data from the hdf5 struct
    # Note that this data is stored as a list, and at least as of 2.23.2018, had fields:
    # d = {
    #             'focus_images':        focus_images,
    #             'focus_images_rgb':   focus_images_rgb,
    #             'focus_params':        focus_params,
    #             'best_slice_inds':     best_slice_inds,
    #             'merged_rgb_image':   merged_rgb_image,
    #             'focus_image_circles': focus_image_circles,
    #             'plot_params':         plot_params,
    #             'save_name_mosaic':    save_name_mosaic,
    #             'save_name_focus':     save_name_focus,
    #             'save_name_rgb':       save_name_rgb,
    #             'tile_full_path':      tile_full_path
    # FROM MICROMANAGER TIFF
    #  data['image'] = image
    # data['n_slices']   = metadata['slices']
    # data['n_channels'] = metadata['channels']
    # data['n_rows']     = micromanager_metadata['Height']
    # data['n_cols']     = micromanager_metadata['Width']
    # data['order']      = metadata['order']
    # data['ChNames']    = micromanager_metadata['Summary']['ChNames']
    # data['metadata']   = metadata
    # data['micromanager_metadata']  = micromanager_metadata
    #             }

    import h5py
    import os
    import numpy as np
    import re
    import pickle

    def hdf5_object_to_dict(hdf5_object):
        # Loads an hdf5 dataset into a nested dict.
        # Input: an hdf5 dataset (or group)
        # Output: a dict with the same structure
        #print('\n In hdf5_to_dict')
        #print hdf5_object
        #print hdf5_object.keys()
        import numpy as np
        
        type_str = str(type(hdf5_object))
        if verbose: 
            print('\nAdvancing into dict')
            print('- type_str is ' + type_str)
        if any([ t in type_str for t in ['h5py._hl.group.Group','h5py._hl.files.File']]):
            # If object is a file or group, iterate over keys
            #print 'found group, making dict with keys',hdf5_object.keys()
            output = {}
            if verbose: print('- Opening into dict')
            for key in hdf5_object:
                if verbose: print('- Returning key: ' + key)
                # if it's a group, nest it into a dict
                output[key] = hdf5_object_to_dict(hdf5_object[key])
        else:
            # Simple dataset, just return value
            #print 'simple dataset, returning value'
            # BUG : Possibly need an eval() here, as hdf5_object.value is a string.
            if verbose: 
                print('- type(hdf5_object.value) is' + str(type(hdf5_object.value)))
        
            if type(hdf5_object.value) is np.ndarray:
                output = hdf5_object.value # Value is present at this point.
            else:
                try:
                    output = pickle.loads(hdf5_object.value)
                    if verbose: print('- pickle.loads succeeded')
                except Exception as e:
                    if verbose: print('- *** pickle.loads failed, len is' + str(len(output)))
                    print 'message :: ' + e.message
                    #print 'args :: ' + e.args
                    pass
        return output    

    ## MAIN
    if sub_dir is not None:
        tile_dir = os.path.join(tile_dir,sub_dir)

    full_path = os.path.join(tile_dir,data_fname)    
    if verbose: 
        print '\tLoading <%s> from <%s>' % (variable_to_load,data_fname),
    with h5py.File(full_path, "r") as f:
        if variable_to_load is None or variable_to_load == '' or variable_to_load == '/':
            # Return h5py dataset object as a dict
            output = hdf5_object_to_dict(f)
        else:
            # Return a given variable
            output = hdf5_object_to_dict(f[variable_to_load])
            
    if verbose: print ' ... done loading.'
    return output
        
def import_micromanager_tiff(full_path,n_channels=4,dimension_order=None):

    """
    Import micromanager tiff, returning a dict containing the data
    Returns image as 
    """

    from tifffile import TiffFile
    import numpy as np
    import os

    print '\tLoading : ',os.path.split(full_path)[1]
    with TiffFile(full_path) as tif:
        image = tif.asarray()
        metadata = tif.imagej_metadata
        micromanager_metadata = tif.micromanager_metadata  # None in stitched images
        
    # Reorder dimensions -  largely used for stitched tiles
    if dimension_order is not None:
        assert( len(dimension_order) == len(image.shape), 'Can''t reshape dimensions because dimension_order (%s) doesn''t match image.shape (%s)' % (str(dimension_order),str(image.shape)))
        image = np.transpose(image,dimension_order)


    # Image pages divisible by n_channels
    if image.shape[0]/n_channels != image.shape[0]//n_channels:
        raise ValueError('Image stack isn''t easily divisible by n_channels  :: %s' % (str(image.shape)))
    
    n_slices = image.shape[0]//n_channels
   
   # CLEANUP - this is only useful for raw imagej images (not stitched files)
    if 'order' in metadata.keys():
        # Deinterleaves if order iterates as c,z
        if metadata['order'] == 'zct':
            print('\tDecoding as zct (really zc) --> converting to C,Z,Y,X')
            image = np.array([ image[idx::n_slices,:,:] for idx in range(n_slices)])
            image = np.swapaxes(image,0,1)

        elif metadata['order'] == 'czt':
            print('\tDecoding as czt')
            image = np.array([ image[idx::n_channels,:,:] for idx in range(n_channels)])
    else:
        metadata['order'] = None

    data = dict()
    data['image'] = image
    data['n_slices']   = metadata['slices']
    data['n_channels'] = metadata['channels']
    data['n_rows']     = image.shape[2]
    data['n_cols']     = image.shape[3]
    data['order']      = metadata['order']
    data['metadata']   = metadata
    if micromanager_metadata is not None:
        data['ChNames']    = micromanager_metadata['Summary']['ChNames']
        data['micromanager_metadata']  = micromanager_metadata
    else:
        data['ChNames'] = ['Ch1','Ch2','Ch3','Ch4']


    return data

############################## IMAGE TRANSFORMATIONS ###################################


def remove_background(data,params=None,verbose=False):
    """
    Remove background.

    tile_data is of shape C,Z,Y,X  e.g (4, 21, 730, 960)

    params['method'] can be a string or list of methods that will be run in the order given:

        subtract_min          - acts pixelwise, subtracts min value across channels (i.e. dimmest channel values)
        subtract_squared_min - 
        subtract_median      -   
        subtract_gaussian - subtracts a 3d Gaussian average from each channel
        rectify - set negative values to 0

        ['subtract_gaussian','rectify']
    """
 
    import numpy as np
    import scipy

    # Bound to range 0..1
    data = data - np.min(data)
    data = data / np.max(data)

    if verbose: print('Subtracting background with pipeline: %s' % (params['pipeline']))
    for method in params['pipeline']:

        if method is 'subtract_min':
            if verbose: print('... running subtract_min')
            data = data - np.min(data,axis=0)        

        if method is 'subtract_squared_min':
            if verbose: print('... running subtract_squared_min')
            data = data - np.min(data,axis=0) ** 2   

        if method is 'subtract_median':
            if verbose: print('... running subtract_median')
            data = data - np.sqrt(np.median(data,axis=0))        

        if method is 'gaussian_hp':
            if verbose: print('... running gaussian_hp with sigmas %s' % (str(params['hp_sigmas'])))
            for cc in range(data.shape[0]):
                data[cc,:,:,:] = data[cc,:,:,:] - scipy.ndimage.filters.gaussian_filter( data[cc,:,:,:], params['hp_sigmas'], order=0, output=None, mode='reflect', cval=0.0, truncate=4.0)

        if method is 'gaussian_lp':
            if verbose: print('... running gaussian_lp with sigmas %s' % (str(params['lp_sigmas'])))
            for cc in range(data.shape[0]):
                data[cc,:,:,:] = scipy.ndimage.filters.gaussian_filter( data[cc,:,:,:], params['lp_sigmas'], order=0, output=None, mode='reflect', cval=0.0, truncate=4.0)

        if method is 'rectify':
            if verbose: print('... rectifying')
            data[data<0] = 0

        if method is 'channel_minmax':
            if verbose: print('... normalizing channel intensity with min/max %s' % (str(params['channel_minmax_percentiles'])))
            for cc in range(data.shape[0]):
                data_cc = data[cc,:,:,:]
                data_cc = data_cc - np.percentile(data_cc,  params['channel_minmax_percentiles'][0])
                data_cc = data_cc / np.percentile(data_cc,  params['channel_minmax_percentiles'][1])
                data_cc[data_cc < 0] = 0
                data_cc[data_cc > 1] = 1
                data[cc,:,:,:] = data_cc




    # Bound to range 0..1
    data = data - np.min(data)
    data = data / np.max(data)

    return data


def get_edf_images(data,sobel_hp_filter_size=None,medfilt_kernel=None,argmax_footprint_size=None):
    """
    Get Extended Depth of Focus images from an image stack using a sobel filter. For each pixel, we examine the z-plane to find the best focus point,
    defined by the maximum "edge" from a sobel filter.  We keep image values from the optimal z-plane across pixels to reconstruct the full image.

    Expects data to be of size [z,y,x]

    """
    import numpy as np
    from scipy import ndimage,signal

    def sobel_edf(image,sobel_hp_filter_size=None):
            # Given an image stack (z,y,x), return a a 2d image with the pixel values taken from z with strongest edges (for each pixel).
            # This borrows heavily from LuisPedro's blog post
            # at metarabbit.wordpress.com/2013/0812/extended-depth-of-field-in-python-using-mahotas
            sobel_image = np.zeros_like(image)
            for z in range(image.shape[0]):
                img = image[z,:,:]
                if sobel_hp_filter_size is not None:
                    img = ndimage.gaussian_filter(img, sobel_hp_filter_size)
                # Sobel filter for edge detection
                Gx = [[ 1, 0, -1],[ 2, 0, -2],[1,0,-1]]
                Gy = [[ 1, 2,  1],[ 0, 0,  0],[-1,-2,-1]]
                img_x = signal.convolve2d(img, Gx,mode='same')
                img_y = signal.convolve2d(img, Gy,mode='same')
                sobel_image[z,:,:] = np.sqrt(img_x**2 + img_y**2)

            if argmax_footprint_size is not None:
                from scipy.ndimage.filters import maximum_filter 
                sobel_image == maximum_filter(sobel_image,footprint=np.ones(argmax_footprint_size))

            # This is a 2d matrix where each xy coordinate is the z coordinate that had maximum value in sobel_image
            sobel_argmax = np.argmax(sobel_image,axis=0)


            # Aply sobel_argmax to the original image.
            z,h,w = image.shape
            tmp_image = image.reshape((z,-1)) # image is now (z,nr_pixels)
            tmp_image = tmp_image.transpose() # image is now (nr_pixels,z)
            image_edf = tmp_image[np.arange(len(tmp_image)),sobel_argmax.ravel()] # select the rightpixel at each location
            image_edf = image_edf.reshape((h,w)) # reshape to get final result

            return image_edf

    # MAIN  - get_edf_images 
    n_channels = data.shape[0]
    n_slices   = data.shape[1]
    focus_images = []

    # Median filter for cleanup
    for cc in range(n_channels):
        focus_image = sobel_edf(data[cc,:,:,:],sobel_hp_filter_size=sobel_hp_filter_size)
        if medfilt_kernel is not None:
            focus_image =  signal.medfilt(focus_image,medfilt_kernel)
        focus_images += [focus_image]

    return focus_images  

def get_focus_images(data,focus_params,cmaps):

    """
    focus_params=plot_params['focus_params'],cmaps=DATA['cmaps'])


     Picks the best focus planes of a data matrix in shape C,Z,Y,X
     Assumes 4 channels
     picks the best focus plane by a convolved variance measure

     focus_params is a dict of form:
        { 'offset'       : 2,
          'do_filter'          : True,
          'hp_filter_size'     : 3,
          'ceil_percentile'    : 99.9,
          'center_bias'        : 1.03,
          'intensity_transform': 'sqrt' }



     Returns
       focus_images        - a list of 2d arrays corresponding to focus image for each channel. 
       focus_images_rgb   - array of size [rows, cols, 4] corresponding to rgb with given cmaps
       best_slice_inds     - list, indices of best slices for each channel
    
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import ndimage,signal


    # If in shape P,C,Z,Y,X, plot the first position by default.
    if len(data.shape) == 5:
        if data.shape[0] == 1:
            data = data[0,:,:,:,:]
            # Note data is in shape C,Z,Y,X

    # Info
    n_channels = data.shape[0]
    n_columns  = n_channels
    n_slices   = data.shape[1]

    focus_images = []
    best_slice_inds = []

    for cc in range(n_channels):
        
        # Normalize to channel min/max
        channel_img = data[cc,:,:,:]
        channel_img = channel_img - np.min(np.ravel(channel_img))
        channel_img = channel_img / np.percentile(np.ravel(channel_img),ceil_percentile)
        #channel_img = np.minimum(channel_img,1)
        
        if intensity_transform is not None:
            if intensity_transform == 'sqrt':
                channel_img = np.sqrt(channel_img)
            else:
                raise ValueError("Didn't recognize intensity transform %s" % (intensity_transform))

        if hp_filter_size is not None:
            # Run a high-pass filter on each image.
            for zz in range(data.shape[1]):
                lowpass = ndimage.gaussian_filter(channel_img[zz,:,:], hp_filter_size)
                channel_img[zz,:,:] = channel_img[zz,:,:] - lowpass
            channel_img = channel_img - np.min(np.ravel(channel_img))
            channel_img = channel_img / np.max(np.ravel(channel_img))

        # Find focus plane
        # Here we take a bunch of subimages, and calculate the variance within each subimage.
        # Then we find the slice with the highest median variance across subimages.
        # slice_variance = [np.var(np.ravel(data[cc,zz,:,:])) for zz in range(data.shape[1])]
        slice_variance = []
        for zz in range(data.shape[1]):

            # Process sub-images
            subimg_variance = []
            grid_size = 1;
            for rrr in range(grid_size):
                for ccc in range(grid_size):
                    # Subimage
                    subimg = channel_img[zz,
                                         (rrr)*data.shape[2]//grid_size:(rrr+1)*data.shape[2]//grid_size,
                                         (ccc)*data.shape[3]//grid_size:(ccc+1)*data.shape[3]//grid_size]

                    # # High-pass filters for edge detection
                    # #f = [[0, -1, 0],[-1, 4, -1],[0, -1, 0]]
                    #f = [[-1, -1, -1],[-1, 8, -1],[-1, -1, -1]]
                    #subimg = signal.convolve2d(subimg, f)

                    # Sobel filter for edge detection - this seems to work better.
                    Gx = [[ 1, 0, -1],[ 2, 0, -2],[1,0,-1]]
                    Gy = [[ 1, 2,  1],[ 0, 0,  0],[-1,-2,-1]]
                    subimg_x = signal.convolve2d(subimg, Gx)
                    subimg_y = signal.convolve2d(subimg, Gy)
                    subimg = np.sqrt(subimg_x**2 + subimg_y**2)

                    # # Canny - not tested
                    # if canny_params is None:
                    #     canny_params = dict()
                    #     canny_params['sigma'] = 2
                    #     canny_params['low_threshold'] = 10
                    #     canny_params['high_threshold'] = 30
                    # subimg = canny(subimg, sigma=canny_params['sigma'], low_threshold=canny_params['low_threshold'], high_threshold=canny_params['high_threshold'])


                    # Variance of subimage
                    subimg_variance += [ np.var(np.ravel( subimg   )) ]

                    # 3-point running average
                    subimg_variance = np.convolve(subimg_variance,np.array([.2,.2,.2,.2,.2]))
                    subimg_variance = subimg_variance[2:-2]

            slice_variance += [np.median(subimg_variance)]            

        # Focus on the center with a linear bias
        bias_vector = np.concatenate( (np.linspace(1,center_bias,np.floor(n_slices/2)),np.linspace(center_bias,1,np.ceil(n_slices/2))),axis=0)
        slice_variance = slice_variance * bias_vector

        # Choose best slice
        best_slice_inds += [np.argmax(slice_variance)]
        print 'Channel %.0f :: best slice at index %.0f' % (cc,best_slice_inds[-1])
        
        if (best_slice_inds[-1] < n_slices/4) | (best_slice_inds[-1] > n_slices*3/4):
            print '*** WARNING ***  : detected focus plane is near edge of stack.'
            #print 'Slice values with center bias %.3f' % (center_bias)
            #for val in  slice_variance: print '%.3g' % (val)


        # Subselect image with best focus
        if offset == 0:
            # Best slice
            focus_img = channel_img[best_slice_inds[-1],:,:]
        else:
            # Max projection of best slice +/- offset
            z_inds = range(best_slice_inds[-1]-offset,best_slice_inds[-1]+offset)
            focus_img = channel_img[z_inds,:,:]
            focus_img = np.max(focus_img,axis=0)

        # # Append to images lists (i.e. one element of each list for each channel)
        focus_images += [focus_img.astype(np.float16)]
        
    return focus_images , best_slice_inds

def normalize_image_list(focus_images,method='collective',min_pc=0,max_pc=100):
    """
    Normalize intensity of focus_images to the [0..1] range.
    
    method : collective --> normalize all together 
           : individual --> normalize each individually.

    """
    import numpy as np

    n_channels = len(focus_images)

    if method == 'collective':
        # Normalize all channels together
        min_intensity = min([np.percentile(image,min_pc) for image in focus_images])
        max_intensity = max([np.percentile(image,max_pc) for image in focus_images])
        for cc in range(n_channels):
            focus_images[cc] = focus_images[cc] - min_intensity
            focus_images[cc] = focus_images[cc] / (max_intensity - min_intensity)

    if method == 'individual':
        # Normalize every channel separately.
        for cc in range(n_channels):
            focus_images[cc] = rescale_image_by_percentile(focus_images[cc],min_pc,max_pc,do_floor=True)

    return focus_images

def rescale_image_by_percentile(image,min_pc=0,max_pc=100,do_floor=True):
    # Rescales image by minimum/maximum percentiles (min_pc,max_pc)
    # Returns an image in the range [0..1]
    import numpy as np
    vmin = np.percentile(image,min_pc)
    vmax = np.percentile(image,max_pc)
    image = (image- vmin) / (vmax-vmin)
    if do_floor:
        image[image < 0] = 0
    return image



########## MAKE RGB IMAGES #################3

def calculate_brightness(r,g,b,gamma=1):
    # Here we are using the brightness calculation per https://en.wikipedia.org/wiki/Luma_(video)
    #Y = 0.2126*(r**gamma) + 0.7152*(g**gamma) + 0.0722*(b**gamma)
    Y = 1*(r**gamma) + 1*(g**gamma) + 1*(b**gamma)
    return Y

def make_rgb_image(image,cmap,normalize_colormap=True,min_pc=0,max_pc=100,gamma=1,):
    """
    Convert a 2d image to a 2+1d RGB image.
    cmap should be a matplotlib colormap
    """
    import numpy as np
    import warnings


    # Convert holoviews cmap into an rgb colormap (e.g. [0.1,0.2,0.3])
    rgb_cmap = np.array(cmap(255))[:3]

    # Normalize brightness of colormap
    if normalize_colormap:
        rgb_cmap =  rgb_cmap / calculate_brightness(rgb_cmap[0],rgb_cmap[1],rgb_cmap[2],gamma)

    # Set image range
    image = rescale_image_by_percentile(image,min_pc,max_pc,do_floor=True)

    # Add color channels to make shape (rows,cols,1)
    image = np.expand_dims(image,-1)

    # Move colormap into a shape that can be broadcast (1,1,3)
    cmap = np.reshape(rgb_cmap,(1,1,3))

    # Gamma and type transformation
    rgb_image = (image**gamma)*cmap.astype(np.float32)

    return rgb_image


def merge_rgb_images(rgb_images,transform=None,additivity='simple',ceil=1):
    """
    Merge RGB images according to a given equation.
    
    additivitiy
        simple = sum of each channel
        SMS = sqrt(sum(intensities across images)) for each channel

    This will work for both RGB and rgb, but will ignore the A channel (i.e. will fill with ones)

    """

    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import signal


    # Takes in rgb image in a list, and merges them to generate a composite
    n_images = len(rgb_images)
    output_image = np.ones_like(rgb_images[0],dtype=np.float16)
    for rgb in range(3):
        
        # This is an array of size N, where N is the number of images to merge.
        channel_images = np.array([image[:,:,rgb] for image in rgb_images])

        # How should we merge RGB rgb_images?  
        if additivity == 'SMS':
            # Merge as sqrt( mean( intensities**2 ) )
            output_image[:,:,rgb] = np.sqrt(np.mean(channel_images**2,axis=0))
        elif additivity == 'simple':
            # Merge as simple sum
            output_image[:,:,rgb] = np.sum(channel_images,axis=0)

    if ceil is not None:
        output_image[output_image > ceil] = ceil;

    return output_image

################################ PLOTTING ###########################################

def plot_focus_images(focus_images, best_slice_inds=None, offset=None, cmaps=None, channel_names=['Ch1','Ch2','Ch3','Ch4'], suptitle=None, min_pc=0,max_pc=100, gamma=None, do_plot=True,do_save = False, save_filename=None):
      
    
    # Plots an image mosaic data from get_focus_plane
    import matplotlib.pyplot as plt
    import os
    import numpy as np

    # Figure
    fig=plt.figure(dpi=600)
    plt.style.use('dark_background')
    
    # Info
    n_channels = len(focus_images)
    n_columns  = np.ceil(n_channels/2)

    if cmaps is None:
        print 'Colormaps not defined - defaulting to gray.'
        cmaps = [[1,1,1]]*4
                     
    for cc in range(n_channels):
        
        # Convert to float and compress into [0..1]
        focus_img = np.float32(focus_images[cc])
        focus_img = focus_img - np.min(focus_img)
        focus_img = focus_img / np.max(focus_img)
        
        # Set background color to black
        cmap = cmaps[cc]
        cmap.set_under(color=u'k')
        

        # min_pc and min_pc are defined as percentiles, and *applied separately for each channel*.
        # Can either be
        # min_pc = 50 # Scalar
        # min_pc = [50] # Short list/tuple
        # min_pc = [50,40,50,60] # One entry for each channel
        # Set intensity floor.  Allows either scalars, length-1 arrays [these are applied to all channels], or length=n_channels arrays (applied to each channel separately)
        if min_pc is None:
            vmin = None;
        elif type(min_pc) is float or type(min_pc) is int or np.isscalar(min_pc):
            vmin = np.percentile(np.ravel(focus_img),min_pc)
        elif len(min_pc) == 1:
            vmin = np.percentile(np.ravel(focus_img),min_pc[0])          
        else:
            vmin = np.percentile(np.ravel(focus_img),min_pc[cc])


        # Set intensity floor - allow either a scalar or a list of value for each channel.
        if type(min_pc) is list:
            this_min_pc = min_pc[cc]
        else:
            this_min_pc = min_pc

        if type(max_pc) is list:
            this_max_pc = max_pc[cc]
        else:
            this_max_pc = max_pc

        # Rescale image to 0..1 via percentiles, perform gamma conversion
        focus_img = rescale_image_by_percentile(focus_img,this_min_pc,this_max_pc,do_floor=True)
        if gamma is not None:
            focus_img = focus_img**gamma

        # Plotting
        fig.add_subplot(2,n_columns , cc+1)
        plt.imshow(focus_img,cmap=cmap)

        # Hide axes
        cur_axes = plt.gca()
        cur_axes.axes.get_xaxis().set_ticks([])
        cur_axes.axes.get_yaxis().set_ticks([])

        if best_slice_inds is not None:
            this_title =  "Channel: %s   Slice: %.0f+/-%.0f" % (channel_names[cc],best_slice_inds[cc],offset)
        else:
            this_title =  "Channel: %s" % (channel_names[cc])
        plt.title(this_title)

    plt.subplots_adjust(wspace=0,hspace=0)
    if suptitle is not None:
        plt.suptitle(suptitle)

    if do_save:
        plt.savefig(save_filename,facecolor='k')

    if do_plot:
        plt.show()
    else:
        plt.close(fig)


def plot_mosaic(data,z_inds=None,z_inds_to_plot=5, n_channels=None, cmaps=None, channel_names=['Ch1','Ch2','Ch3','Ch4'], suptitle=None, gamma=None,do_plot=True,do_save=False,save_filename=None):
        
    # Plots an image mosaic of a data matrix in shape C,Z,Y,X
    # Assumes 4 channels

    import matplotlib.pyplot as plt
    import numpy as np
    import os


    # If in shape P,C,Z,Y,X, plot the first position by default.
    if len(data.shape) == 5:
        if data.shape[0] == 1:
            data = data[0,:,:,:,:]
    
    # Channels
    if n_channels is None:
         n_channels = data.shape[0]
    n_rows    = n_channels
  
    # Columns = n-zstep to plot
    if z_inds is None:
        n_slices = data.shape[1]
        n_columns = min(n_slices,z_inds_to_plot)
        z_inds    = np.linspace(n_slices//4,n_slices-n_slices//4,n_columns,dtype=int)
    else:
        n_columns = len(z_inds)
    
    # Initialize fig.
    w=25
    h = np.ceil(25/n_columns*4-3)
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(w, h))
    
    if cmaps is None:
        cmaps = [plt.get_cmap('Blues').reversed(),
                 plt.get_cmap('Greens').reversed(),
                 plt.get_cmap('Oranges').reversed(),
                 plt.get_cmap('Reds').reversed()]    
    [cmap.set_under(color=u'k') for cmap in cmaps]

    for cc in range(n_channels):
        
        # Normalize channel image to min/max
        channel_img = data[cc,:,:,:]
        channel_img = channel_img - np.min(np.ravel(channel_img))
        channel_img = channel_img / np.max(np.ravel(channel_img))
        
        if gamma is not None:
            channel_img = channel_img**gamma

        for z in range(n_columns):
            img = channel_img[z_inds[z],:,:]
            fig.add_subplot(n_rows, n_columns, n_columns*(cc)+z+1)
            plt.imshow(img,cmap=cmaps[cc])

            # Hide axes
            cur_axes = plt.gca()
            cur_axes.axes.get_xaxis().set_ticks([])
            cur_axes.axes.get_yaxis().set_ticks([])
            plt.yticks = None

            # Title and labels
            if cc == 0:
                plt.title('Slice ' + str(z_inds[z]+1))
            if z == 0:
                plt.ylabel(channel_names[cc])

    plt.subplots_adjust(wspace=0,hspace=0)
    if suptitle is not None:
        plt.suptitle(suptitle)

    if save_filename is not None:
        plt.savefig(save_filename,facecolor='k')

    if do_plot:
        plt.show()
    else:
        plt.close(fig)


def plot_rgb_image(merged_image, circles=None,cmaps=None, suptitle=None, do_plot=True, do_save=False, save_filename=None):

    import matplotlib.pyplot as plt
    import numpy as np
    import os

    # Plot rgb image after normalizing each channel
    image = np.float32(merged_image)

    fig=plt.figure(dpi=600)
    plt.style.use('dark_background')
    plt.imshow(image)

    # Plot circles
    if circles is not None and circles:

        def plot_circle(x,y,radius,scale=2,color='w'):
            circle1=plt.Circle((x,y),radius*scale,color=color,fill=False,linewidth=0.5)
            plt.gcf().gca().add_artist(circle1)

        if type(circles[0]) is list:
            # Here we have nested tuples for each channel: 
            # as a list of lists of tuples 
            # i.e. [ circles1, circles2, circles3, circles4 ]
            # Where circles1 = [ (x,y,r), (x,y,r), ... ]

            # Make sure we have a colormap
            if cmaps is None:
                cmaps = [plt.get_cmap('Blues'),
                         plt.get_cmap('Greens'),
                         plt.get_cmap('Oranges'),
                         plt.get_cmap('Reds')]

            for channel_circles,cmap in zip(circles,cmaps):
                for circle in channel_circles:
                    color = cmap(255)
                    plot_circle(circle[0],circle[1],circle[2],color=color)

        else:
            # Here we have simple circles as a list of tuples.= [ (x,y,r), (x,y,r), ... ]
            for circle in circles:
                plot_circle(circle[0],circle[1],circle[2])

    if suptitle is not None:
        plt.suptitle(suptitle)

    if do_save:
        plt.savefig(save_filename,facecolor='k')

    if do_plot:
        plt.show()
    else:
        plt.close(fig)

def find_circles(input_image,params,plot_final=True,plot_all=False):

    """

    Find circles in a 1-channel image, based off of a canny / hough transform.
    Params are defaults for 20x widefield imaging, will have to be readjusted later.
    
    Largely borrowed from: http://scikit-image.org/docs/dev/auto_examples/edges/plot_circular_elliptical_hough_transform.html

    Returns: circles = zip(cx,cy,radii)

    """

    #Default parameters
    try:
        canny_params = params['canny_params']
    except:
        print('Setting canny_params to defaults')
        canny_params = dict()
        canny_params['sigma'] = 2
        canny_params['low_threshold'] = 10
        canny_params['high_threshold'] = 30
        
    try:
        hough_params = params['hough_params']
    except:
        print('Setting hough_params to defaults')
        hough_params = dict()
        hough_params['radius_low']  = 15
        hough_params['radius_high'] = 20
        hough_params['radius_n']    = 5
    
    try:
        peak_params = params['peak_params']
    except:
        print('Setting peak_params to defaults')
        peak_params= dict()
        peak_params['total_num_peaks'] = 50
        peak_params['min_xdistance']   = hough_params['radius_high']*3
        peak_params['min_ydistance']   = hough_params['radius_high']*3


    
    from skimage import data, color
    from skimage.transform import hough_circle, hough_circle_peaks
    from skimage.feature import canny
    from skimage.draw import circle_perimeter
    from skimage.util import img_as_ubyte
    import numpy as np
    import matplotlib.pyplot as plt

    image = img_as_ubyte(input_image)
    n_rows = image.shape[0]
    n_cols = image.shape[1]


    
   
    # Load picture and detect edges
    #image = img_as_ubyte(data.coins()[160:230, 70:270])
    edges = canny(image, sigma=canny_params['sigma'], low_threshold=canny_params['low_threshold'], high_threshold=canny_params['high_threshold'])

    
    # Detect two radii
    hough_radii = np.arange( hough_params['radius_low'],hough_params['radius_high'],hough_params['radius_n'] )
    hough_res = hough_circle(edges, hough_radii)


    # Select the most prominent N circles
    accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,
                                               total_num_peaks = peak_params['total_num_peaks'],
                                               min_xdistance   = peak_params['min_xdistance'],
                                               min_ydistance   = peak_params['min_ydistance'])
    circles = zip(cx,cy,radii)


    if plot_all:
        # Plot Edges
        fig,ax=plt.subplots(dpi=600)
        plt.imshow(edges[:,:])
        # Plot hough transform (should see peaks)
        fig,ax=plt.subplots(dpi=600)
        plt.imshow(np.sum(hough_res,axis=0))

    if plot_final or plot_all:
        fig, ax = plt.subplots(ncols=1, nrows=1, dpi=600)
        ax.imshow(input_image)

        # Plot circles
        def plot_circle(x,y,radius,color='w',scale=2):
            circle1=plt.Circle((x,y),radius*scale,color=color,fill=False)
            plt.gcf().gca().add_artist(circle1)

        for circle in circles:
            plot_circle(circle[0],circle[1],circle[2])

        plt.show()
        
    return circles


def plot_stitched_figures(sequence_data,focus_images,merged_rgb_image,DATA,pos,plot_params,figures_to_plot):
    """
    Convenience function for plotting a bunch of figures

    sequence_data is matrix format
    focus_images is a list of 2d images (one for each channel)

    """

    import os

    # SaveDir
    save_folder = os.path.join(DATA[pos]['path'], plot_params['save_dir'] )
    save_path = os.path.join(save_folder,DATA[pos]['fname_stub'])
    try:    os.mkdir( save_folder )
    except: pass

    if 'plot_mosaic' in figures_to_plot:
        # Plot 0 - Mosaic of each channel
        plot_mosaic( sequence_data,
                         z_inds         = None,
                         z_inds_to_plot = 5, 
                         n_channels     = None, 
                         cmaps          = DATA['cmaps'], 
                         channel_names  = DATA['channel_names'], 
                         gamma          = plot_params['gamma'],
                         suptitle       = plot_params['suptitle'],
                         do_plot        = plot_params['do_plot_figures'],
                         do_save        = plot_params['do_save_figures'],
                         save_filename  = save_path + '_Mosaic')

    if 'plot_focus_images' in figures_to_plot:
        # Plot 1- 2d image of each channel
        suptitle = 'Focus by Sobel/EDF'
        plot_focus_images( focus_images,
                               min_pc   = plot_params['min_pc'],
                               max_pc   = plot_params['max_pc'],
                               cmaps    = DATA['cmaps'],
                               gamma    = plot_params['gamma'],
                               suptitle = plot_params['suptitle'],
                               do_plot  = plot_params['do_plot_figures'],
                               do_save  = plot_params['do_save_figures'],
                               save_filename = save_path + '_Focus')

    if 'plot_rgb_image' in figures_to_plot:
        # Plot 2 - RGB image
        plot_rgb_image( merged_rgb_image,
                            do_plot       = plot_params['do_plot_figures'],
                            do_save       = plot_params['do_save_figures'],
                            save_filename = save_path + '_RGB')

    if plot_rgb_image in figures_to_plot:
        # Plot 3 - RGB with circle around potential cells
        plot_rgb_image( merged_rgb_image,
                            circles       = focus_image_circles,
                            cmaps         = DATA['cmaps'],
                            do_plot       = plot_params['do_plot_figures'],
                            do_save       = plot_params['do_save_figures'],
                            save_filename = save_path + '_RGBCircles')
    


############################## TILE STITCHING ###################################

def register_images(img1,img2 , overlap_fraction=.05, overlap_multiple=2, trim_fraction=0.1 ,  lp_filter_size = 3, hp_filter_size = 1, adjacency=None, verbose=0, plot_figures=1 ):

    # Assume log-scaled (or similar) float32 images of equal size.
    # Expect img1 / img2 to be of size [channel,rows,cols] - this is 2d only, no Z.
    # Adjacency of img1/img2 is either left_right or top_bottom
    # verbose == 1 is text, verbose ==2 is figures
    # Returns 
    #   final_offset  -- list (row,column)
    #   peak_score  - scalar measure of peak quality (max/median of xcorr image)
    #   xcorr_sum     - 2D array showing xcorr_sum

    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import signal
    import matplotlib.patches as patches
    from scipy import ndimage,signal

    
    if verbose:
        print 'img1.shape ' + str(img1.shape)
        print 'img2.shape ' + str(img2.shape)
    assert len(img1.shape)==3, 'Expect img1  to be of size [channel,rows,cols]'
    assert len(img2.shape)==3, 'Expect img2 to be of size [channel,rows,cols]'
    assert img1.shape == img2.shape, 'Expect img1 and img2 to be of the same size'
    
    n_channels = img1.shape[0]
    out_channels = None
                     
        
    # TRIM IMAGES APPROPRIATELY
    # For left_right, we trim the right edge of the LEFT image to 2x "overlap_fraction", and the left edge of the RIGHT image to 1x overlap_fraction.
    #    We also trim the rows of img2 symmetrically by inset_fraction in case the top/bottom edges are slightly off
    # For top_bottom, we trim the bottom edge of the TOP image to 2x "overlap_fraction", and the top of the BOTTOM image to 1x overlap_fraction.
    #    We also trim the columns of img2 symmetrically by inset_fraction in case the top/bottom edges are slightly off.
    if adjacency == 'left_right':
        # Here we do the RIGHT edge of img1 and the LEFT edge of img2
        img1_rows = img1.shape[1]
        img1_cols = img1.shape[2]
        img1_first_row = 0
        img1_last_row  = img1_rows
        img1_first_col = int(np.floor((1-overlap_multiple*overlap_fraction)*img1_cols))
        img1_last_col  = img1_cols
        img1_trimmed = img1[ :, img1_first_row:img1_last_row, img1_first_col:img1_last_col]

        img2_rows = img2.shape[1]
        img2_cols = img2.shape[2]
        img2_first_row = int(np.floor(trim_fraction/2 * img2_rows)) # smaller
        img2_last_row  = int(np.ceil((1-trim_fraction/2) * img2_rows)) # smaller
        img2_first_col = 0
        img2_last_col  = int(np.ceil((overlap_fraction)*img2_cols))
        img2_trimmed = img2[ :, img2_first_row:img2_last_row, img2_first_col:img2_last_col]

        if verbose: print 'Expected subimage offset: row %.0f, col %.0f' % ( img2_first_row, img1_trimmed.shape[2] - img1_cols*overlap_fraction )

    elif adjacency == 'top_bottom':
        # Here we do the BOTTOM edge of img1 and the TOP edge of img2
        # We also trim the rows of img2 by inset_fraction in case the top/bottom edges are slightly off
        img1_rows = img1.shape[1]
        img1_cols = img1.shape[2]
        img1_first_row = int(np.floor((1-overlap_multiple*overlap_fraction)*img1_rows))
        img1_last_row  = img1_rows
        img1_first_col = 0
        img1_last_col  = img1_cols
        img1_trimmed = img1[ :, img1_first_row:img1_last_row, img1_first_col:img1_last_col]

        img2_rows = img2.shape[1]
        img2_cols = img2.shape[2]
        img2_first_row = 0
        img2_last_row  = int(np.ceil((overlap_fraction)*img2_rows))
        img2_first_col = int(np.floor(trim_fraction/2 * img2_cols)) # smaller
        img2_last_col  = int(np.ceil((1-trim_fraction/2) * img2_cols)) # smaller
        img2_trimmed = img2[ :, img2_first_row:img2_last_row, img2_first_col:img2_last_col]
            
        if verbose: print 'Expected subimage offset: row %.0f, col %.0f' % (  img1_trimmed.shape[1] - img1_rows*overlap_fraction , img2_first_col)

    else:
        raise ValueError("Input 'adjacency' must be either 'left_right or 'top_bottom'")
    
    if verbose:
        print 'img1_trimmed.shape ' + str(img1_trimmed.shape)
        print 'img2_trimmed.shape ' + str(img2_trimmed.shape)
  
        
    # RUN XCORRELATION FOR EACH CHANNEL
    for channel in range(n_channels):
        if verbose: print 'channel ' + str(channel)

        # data1 / dat2 are single-channel
        chan1 = img1_trimmed[channel,:,:]
        chan2 = img2_trimmed[channel,:,:]

        # Bandpass filter
        chan1 = ndimage.gaussian_filter(chan1, hp_filter_size) - ndimage.gaussian_filter(chan1, lp_filter_size)
        chan1 = chan1 - np.min(chan1)
        chan1 = chan1 / np.max(chan1)
        
        chan2 = ndimage.gaussian_filter(chan2, hp_filter_size) - ndimage.gaussian_filter(chan2, lp_filter_size)
        chan2 = chan2 - np.min(chan2)
        chan2 = chan2 / np.max(chan2)

        if verbose: 
            print 'chan1.shape ' + str(chan1.shape)
            print 'chan2.shape ' + str(chan2.shape)
        
        # Calculate cross-correlation
        xcorr  = signal.correlate2d(chan1,chan2,mode='valid')

        # Make xcorr_all, which is of size (Channels,Rows,Columns)
        try:
            xcorr_all = np.concatenate((xcorr_all,np.expand_dims(xcorr, axis=0)),axis=0)
        except:
            # Initialize as (Channels,Rows,Columns)
            xcorr_all = np.expand_dims(xcorr, axis=0)
    
    
    # Sum of XCORR across channnels
    xcorr_sum = np.sum(xcorr_all,axis=0)

    #FIND INDICES OF MAXIMUM VALUE IN xcorr_sum 
    xcorr_offset = np.int16(np.unravel_index(xcorr_sum.argmax(), xcorr_sum.shape))

    # CALCULATE PEAK-SCORE
    # ALTERNATIVE IF THERE ARE MULTIPLE PEAKS
    # from skimage.feature import peak_local_max
    # peaks = peak_local_max(xcorr_sum,num_peaks=2,min_distance = 5)
    # print xcorr_sum.shape
    # print 'peaks :: '
    # print peaks.shape
    # print peaks
    # print peaks[0]
    # print peaks[1]
    # peak_score = xcorr_sum[peaks[0][0],peaks[0][1]] /xcorr_sum[peaks[1][0],peaks[1][1]]

    # Peak_score is (max -median) / (max-min).  Range 0..1  This only really makes sense if correlate2d returns the 'valid' region, since that's all non-zero
    peak_score =  (np.max(xcorr_sum) - np.median(xcorr_sum)) / (np.max(xcorr_sum) - np.min(xcorr_sum))

    if verbose: 
        print 'peak_score :: %.3g' % (peak_score)

    # CALCULATE FINAL OFFSET (origin of img2 in the coordinates of img1)
    if adjacency == 'left_right':
        final_offset = xcorr_offset + np.array((-img2_first_row, img1_first_col))
    elif adjacency == 'top_bottom':
        final_offset = xcorr_offset + np.array((img1_first_row, -img2_first_col))


    
    if verbose:
        print 'Observed subimage offset (max xcorr across channels) row, col ::  ' + str(xcorr_offset)
        print 'Observed final offset (max xcorr across channels) row, col ::  ' + str(final_offset)

    ## PLOTTING ##
    if plot_figures:
        fig=plt.figure(dpi=300)
        plt.style.use('dark_background')

        img1_sum = np.sum(img1,axis=0)
        #img1_sum = img1_sum/np.max(img1_sum)
        img2_sum = np.sum(img2,axis=0)
        #img2_sum = img2_sum/np.max(img2_sum)

        # Expand dynamic range, to the same degree for each image
        max_val = np.max([np.max(img1_sum),np.max(img2_sum)])
        min_val = np.min([np.min(img1_sum),np.min(img2_sum)])
        img1_sum = (img1_sum - min_val) / (max_val - min_val)
        img2_sum = (img2_sum - min_val) / (max_val - min_val)


        def add_rect(first_row,last_row,first_col,last_col):
            ax.add_patch(patches.Rectangle((first_col,first_row),last_col-first_col,last_row-first_row,fill=False,color='r',linewidth=1))

    if plot_figures > 1:
        # Plot img1
        ax = fig.add_subplot(3,2,1)
        plt.imshow(img1_sum,aspect='equal')
        plt.grid(color='w',linestyle='dashed',linewidth=0.5)
        add_rect(img1_first_row,img1_last_row,img1_first_col,img1_last_col)

        # Plot img2
        ax = fig.add_subplot(3,2,2)
        plt.imshow(img2_sum,aspect='equal')
        plt.grid(color='w',linestyle='dashed',linewidth=0.5)  
        add_rect(img2_first_row,img2_last_row,img2_first_col,img2_last_col)

        # Plot img1_trimmed
        ax = fig.add_subplot(3,2,3)
        plt.imshow(np.sum(img1_trimmed,axis=0),aspect='equal')
        plt.grid(color='w',linestyle='dashed',linewidth=0.5)

        # Plot img2_trimmed
        fig.add_subplot(3,2,4)
        plt.imshow(np.sum(img2_trimmed,axis=0),aspect='equal')
        plt.grid(color='w',linestyle='dashed',linewidth=0.5)  # Aligned result

        # Plot xcorr_sum
        ax = fig.add_subplot(3,2,5)
        plt.imshow(xcorr_sum)
        plt.plot(xcorr_offset[1],xcorr_offset[0],'rx')
        plt.grid(color='w',linestyle='dashed',linewidth=0.5)
        plt.title('Peak score: %.3f' % (peak_score))
    
    if plot_figures:
        # Plot img1/img2 as red/green overlap
        fig=plt.figure(dpi=300)
        plt.style.use('dark_background')
        if adjacency == 'left_right':
            full_rows = img1_rows + np.abs(final_offset[0]) # can be negative
            full_cols = img1_cols + img2_cols - final_offset[1]
            full_matrix = np.zeros((np.abs(final_offset[0])+img2_rows,np.abs(final_offset[1])+img2_cols,3)) # RGB image
            if verbose: print 'full_matrix.shape ' + str(full_matrix.shape)
            if final_offset[0] >= 0: 
                # Positive vertical offset (img2 below img1)
                full_matrix[ 0:img1_rows, 0:img1_cols,0] = img1_sum # Red
                full_matrix[ final_offset[0] : final_offset[0]+img2_rows , final_offset[1] : final_offset[1]+img2_cols , 1] = img2_sum # Green
            else: 
                # Negative vertical offset (img2 above img1), i.e. final_offset[0] is negative
                full_matrix[ -final_offset[0] : -final_offset[0] +img1_rows,               0 : img1_cols,                  0] = img1_sum # Red
                full_matrix[               0 : img2_rows                 , final_offset[1] : final_offset[1]+img2_cols , 1] = img2_sum # Green
            full_matrix[:,:,2] = full_matrix[:,:,0] # Blue

        
            fig=plt.figure(dpi=600)

            ax = fig.add_subplot(2,2,1)
            plt.imshow(np.sum(img1,axis=0))
            #plt.title(img1_label)

            ax = fig.add_subplot(2,2,2)
            plt.imshow(np.sum(img2,axis=0))
            #plt.title(img2_label)
        
            ax = fig.add_subplot(2,1,2)
            plt.imshow(np.sqrt(full_matrix/np.max(full_matrix)))
            fig.suptitle('Overlap (adjacency = %s)' % (adjacency))


        elif adjacency == 'top_bottom':
            full_cols = img1_cols + np.abs(final_offset[1]) # can be negative
            full_rows = img1_rows + img2_rows - final_offset[0]
            full_matrix = np.zeros((np.abs(final_offset[0])+img2_rows,np.abs(final_offset[1])+img2_cols,3)) # RGB image            
            if verbose: print 'full_matrix.shape ' + str(full_matrix.shape)
            if final_offset[1] >=0:
                # Positive horizonal offset (img2 to the right of img1)
                full_matrix[ 0:img1_rows, 0:img1_cols,0] = img1_sum # RED
                full_matrix[ final_offset[0] : final_offset[0]+img2_rows , final_offset[1] : final_offset[1]+img2_cols , 1] = img2_sum # GREEN
            else:
                # Negative horizontal offset (img2 to the left of img1)
                full_matrix[ 0:img1_rows, -final_offset[1]:-final_offset[1]+img1_cols,0] = img1_sum # RED
                full_matrix[ final_offset[0] : final_offset[0]+img2_rows ,  0:img2_cols , 1] = img2_sum # GREEN
            full_matrix[:,:,2] = full_matrix[:,:,0] # Blue

            fig=plt.figure(dpi=600)

            ax = fig.add_subplot(2,2,1)
            plt.imshow(np.sum(img1,axis=0))
            #plt.title(img1_label)

            ax = fig.add_subplot(2,2,3)
            plt.imshow(np.sum(img2,axis=0))
            #plt.title(img2_label)
            
            ax = fig.add_subplot(1,2,2)
            plt.imshow(np.sqrt(full_matrix/np.max(full_matrix)))
            fig.suptitle('Overlap (adjacency = %s)' % (adjacency))

            

    return final_offset, peak_score, xcorr_sum

def test_register_image(this_data,adjacency = 'left_right',overlap=0.05,focus_slice=10,verbose=1):

    import numpy as np
    import matplotlib.pyplot as plt

    # Test Script for registering images
    # Expects this_data = data['full_image],
    
    this_data = np.sqrt(np.float32(this_data))
    n_rows = this_data.shape[3]
    n_cols = this_data.shape[4]

    #adjacency = 'left_right'
    #adjacency = 'top_bottom'

    if adjacency == 'left_right':

        img1_right_column = int(n_cols/2 + n_cols*overlap)
        img2_left_column  = int(n_cols/2 - n_cols*overlap)
        if verbose:
            print 'img1 right column: %.0f' % (img1_right_column)
            print 'img2 left column: %.0f' % (img2_left_column)
        
        img1 = this_data[0,:,focus_slice, :, :img1_right_column ]
        img2_row_offset = 0
        img2_col_offset = img2_left_column
        img2 = this_data[0,:,focus_slice,img2_row_offset:, img2_col_offset: ]

        if verbose: print 'test_register_image :: left_right :: n_rows*overlap %.0f' % (n_cols*overlap)

    elif adjacency == 'top_bottom':

        img1 = this_data[0,:,focus_slice, :int(n_rows/2+n_rows*overlap), : ]
        img2_row_offset = int(np.ceil(n_rows/2 - n_rows*overlap))
        img2_col_offset = 0
        img2 = this_data[0,:,focus_slice,img2_row_offset:, img2_col_offset: ]
        if verbose:  print 'test_register_image :: top_bottom :: n_cols*overlap %.0f' % (n_cols*overlap)


    if verbose:
        print 'test_register_image :: this_data.shape ' + str(this_data.shape)
        print 'test_register_image :: img1.shape ' + str(img1.shape)
        print 'test_register_image :: img2.shape %s ' % (str(img2.shape))
        print 'test_register_image :: expected offset :: ' + str((img2_row_offset,img2_col_offset)) + ' ***\n'

    ## PLOT ##
    fig=plt.figure(dpi=600)
    plt.style.use('dark_background')
    # Plot img1_trimmed
    ax = fig.add_subplot(2,2,1)
    plt.imshow(np.sum(img1,axis=0),aspect='equal')
    plt.grid(color='w',linestyle='dashed',linewidth=0.5)
    if adjacency == 'left_right': plt.vlines(n_cols/2-1,plt.ylim()[0],plt.ylim()[1],colors='r')
    if adjacency == 'top_bottom': plt.hlines(n_rows/2-1,plt.xlim()[0],plt.xlim()[1],colors='r')

    # Plot img2_trimmed
    fig.add_subplot(2,2,2)
    plt.imshow(np.sum(img2,axis=0),aspect='equal')
    plt.grid(color='w',linestyle='dashed',linewidth=0.5)  
    if adjacency == 'left_right': plt.vlines(n_cols*overlap,plt.ylim()[0],plt.ylim()[1],colors='r')
    if adjacency == 'top_bottom': plt.hlines(n_rows*overlap,plt.xlim()[0],plt.xlim()[1],colors='r')
    fig.suptitle('Original Images')

    max_inds, peak_score, xcorr_sum = register_images(img1,img2 , overlap_fraction=0.1, trim_fraction=0.1 ,adjacency=adjacency,verbose=verbose )

def align_3x3_tiles(tile_dir,hdf5_dir,tile_labels,left_right_params=None, top_bottom_params=None):

    # Perform a stitching registration of 3x3 tiles
    # Each edge is stitched using register_images()
    # With variable overlap parameters


    import numpy as np
    import matplotlib.pyplot as plt
    import os

    print('Aligning 3x3 tiles in \n\t%s' % (os.path.join(os.path.basename(os.path.dirname(tile_dir)),hdf5_dir)))

    # Default parameters for left/right edges
    if left_right_params == None:
        left_right_params = {
            'adjacency'         : 'left_right',
            'overlap_fraction'  : 0.1,
            'trim_fraction'     : 0.1,
            'focus_slice'       : 10,
            'lp_filter_size'    : 3,
            'hp_filter_size'    : 1,
            'overlap_multiple'  : 5 }

   # Default parameters for top/bottom edges
    if top_bottom_params == None:
        top_bottom_params = {
            'adjacency'         : 'top_bottom',
            'overlap_fraction'  : 0.05,
            'trim_fraction'     : 0.2,
            'focus_slice'       : 10,
            'lp_filter_size'    : 3,
            'hp_filter_size'    : 1 ,
            'overlap_multiple'  : 5 }


            
    rows = len(tile_labels)
    cols = len(tile_labels[0])

    # tile_offsets is in tile_row, tile_col, img_row_col
    print '\n Aligning left/right'
    left_right_offsets = np.zeros((rows,cols,2))
    for row,tile_row in enumerate(tile_labels):
        for col in range(len(tile_row)):
            
            # First column gets offset of 0,0
            if col == 0:
                left_right_offsets[row,col,:] = [0,0]
                continue
            
            # Else we align to previous column image
                
            img1_label = tile_labels[row][col-1]
            img2_label =  tile_labels[row][col]

            
            # Get focus images, which are of size (channels, rows, columns)
            img1 = np.asarray(load_hdf5_data(tile_dir,img1_label,'focus_images',sub_dir=hdf5_dir))
            img2 = np.asarray(load_hdf5_data(tile_dir,img2_label,'focus_images',sub_dir=hdf5_dir))
            
            print('\tPerforming left/right alignment <%s> and <%s> ' % (img1_label,img2_label))
            
            # Normalize intensity
            img1 = np.sqrt(np.float32(img1))
            img2 = np.sqrt(np.float32(img2))
            max_val = np.max([np.max(img1),np.max(img2)])
            min_val = np.min([np.min(img1),np.min(img2)])
            img1 = (img1 - min_val) / (max_val - min_val)
            img2 = (img2 - min_val) / (max_val - min_val)

            max_inds, peak_score, _ = register_images(img1,img2, 
                                                          overlap_fraction=left_right_params['overlap_fraction'], 
                                                          overlap_multiple=left_right_params['overlap_multiple'], 
                                                          trim_fraction=left_right_params['trim_fraction'],
                                                          lp_filter_size=left_right_params['lp_filter_size'],
                                                          hp_filter_size=left_right_params['hp_filter_size'],
                                                          adjacency=left_right_params['adjacency'],
                                                          verbose=0, plot_figures=1)
            
            # KEY OUTPUT
            left_right_offsets[row,col,:] = max_inds
            
            if peak_score > 0.7:
                print '\tAligned %s (left) with %s (right) :: peak score %.3f *\n' % (img1_label,img2_label,peak_score)
            else:
                print '\tAligned %s (left) with %s (right) :: peak score %.3f\n' % (img1_label,img2_label,peak_score)
            
            
    print '\n Aligning top/bottom'
    top_bottom_offsets = np.zeros((rows,cols,2))
    for row,tile_row in enumerate(tile_labels):
        for col in range(len(tile_row)):
            
            # First column gets offset of 0,0
            if row == 0:
                top_bottom_offsets[row,col,:] = [0,0]
                continue
            
            # Else we align to previous column image
                
            img1_label = tile_labels[row-1][col]
            img2_label = tile_labels[row][col]

            # Get focus images, which are of size (channels, rows, columns)
            img1 = np.asarray(load_hdf5_data(tile_dir,img1_label,'focus_images',sub_dir=hdf5_dir))
            img2 = np.asarray(load_hdf5_data(tile_dir,img2_label,'focus_images',sub_dir=hdf5_dir))
            
            print('\tPerforming top/bottom alignment of <%s> and <%s> ' % (img1_label,img2_label))
            
            # Normalize intensity
            img1 = np.float32(img1)
            img2 = np.float32(img2)
            max_val = np.max([np.max(img1),np.max(img2)])
            min_val = np.min([np.min(img1),np.min(img2)])
            img1 = (img1 - min_val) / (max_val - min_val)
            img2 = (img2 - min_val) / (max_val - min_val)
            #img1 = np.sqrt(img1)
            #img2 = np.sqrt(img2)
            
    
            # PERFORM REGISTRATION
            max_inds, peak_score, _ = register_images(img1,img2, 
                                                          overlap_fraction=top_bottom_params['overlap_fraction'], 
                                                          overlap_multiple=top_bottom_params['overlap_multiple'], 
                                                          trim_fraction=top_bottom_params['trim_fraction'],
                                                          lp_filter_size=top_bottom_params['lp_filter_size'],
                                                          hp_filter_size=top_bottom_params['hp_filter_size'],
                                                          adjacency=top_bottom_params['adjacency'],
                                                          verbose=0, plot_figures=1)            
            top_bottom_offsets[row,col,:] = max_inds
            
            if peak_score > 0.7:
                print 'Aligned %s (top) with %s (bottom) :: peak score %.3f \t *' % (img1_label,img2_label,peak_score)
            else:
                print 'Aligned %s (top) with %s (bottom) :: peak score %.3f' % (img1_label,img2_label,peak_score)

    # Convert offsets to center-relative (row, col, offset_row, offset_col), relative to center tile at 0,0
    center_relative_offsets = convert_registration_offsets(left_right_offsets, top_bottom_offsets)
    return center_relative_offsets  

def convert_registration_offsets(left_right_offsets, top_bottom_offsets):

    # Our original registration pipeline output a bunch of left-right registrations,
    # as well as a bunch of top/bottom registrations.  This converts these into overall
    # coordinate offsets from the *center* tile.

    # Inputs: left_right_offsets, top_bottom_offsets 
    #   Each is a 4d matrix with (row, col, offset_row, offset_col) that represents the origin (top-left location) of img2 in the coordinates of img1
    #   Left/right offsets in order left/right (so offsets are large and positive)
    #   Top/bottom offsets are similar (large and positive since we're in a top-left coordinate system).

    # Outputs: center_relative_offsets (4d matrix as above)

    # Sample output:
    #     left_right_offsets
    #     array([[[  0    ,   0   ],          [  4    ,   812 ],          [  4    ,   703 ]], 
    #            [[  0    ,   0   ],          [  5    ,   731 ],          [  3    ,   788 ]], 
    #            [[  0    ,   0   ],          [  3    ,   814 ],          [  4    ,   705 ]]])


    #     top_bottom_offsets
    #     array([[[  0    ,   0   ],          [  0    ,   0   ],          [  0    ,   0   ]], 
    #           [[  618  ,  -9   ],          [  620  ,  -89  ],          [  619  ,  -3   ]], 
    #           [[  611  ,  -4   ],          [  608  ,   79  ],          [  610  ,  -4   ]]])

    #     center_relative_offsets
    #     array([[[ -624  ,  -723 ],          [ -620  ,   89  ],          [ -616  ,   792 ]], 
    #           [[ -5    ,  -731 ],          [  0    ,   0   ],          [  3    ,   788 ]], 
    #           [[  605  ,  -735 ],          [  608  ,   79  ],          [  612  ,   784 ]]])



    import numpy as np
    import pprint as pp

    np.set_printoptions(edgeitems=120)
    np.core.arrayprint._line_width = 80
    np.set_printoptions( threshold=100, edgeitems=10, linewidth=140,
        formatter = dict( float = lambda x: " % .03g\t" % x ))  # float arrays %.3g


    def print_array(arr):

        tmp = pp.pformat(arr)
        tmp = tmp.replace('],\n','],\t')
        #tmp = tmp.replace('\t','  ')
        print(tmp+'\n')
        
    print 'left_right_offsets'
    print_array(left_right_offsets)
    print
    print 'top_bottom_offsets'
    print_array(top_bottom_offsets)

    def ndprint(a, format_string ='{0:.2f}'):
        print [format_string.format(v,i) for i,v in enumerate(a)]

    def ndprint(arr):
        for subarr in arr:
            print(subarr)
            print 'hi'

    center_relative_offsets = np.zeros_like(left_right_offsets)

    # Various algorithms for stitching

    ##SIMPLE - i.e. everything -->  Minor seams, not obvious whether they're important.
    center_relative_offsets[:,0,:] += -1 * left_right_offsets[:,1,:] # top row
    center_relative_offsets[:,2,:] += +1 * left_right_offsets[:,2,:] # bottom row
    # Fill in T/B offsets (rows)
    center_relative_offsets[0,:,:] += -1 * top_bottom_offsets[1,1,:] # left column
    center_relative_offsets[2,:,:] += +1 * top_bottom_offsets[2,1,:] # right column


    # # L/R is columns only, T/B is rows only
    # # Fill in L/R offsets - columns only
    # center_relative_offsets[:,0,1] += -1 * left_right_offsets[:,1,1]
    # center_relative_offsets[:,2,1] += +1 * left_right_offsets[:,2,1]
    # # Fill in T/B offsets (rows) - rows only
    # center_relative_offsets[0,:,0] += -1 * top_bottom_offsets[1,1,0]
    # center_relative_offsets[2,:,0] += +1 * top_bottom_offsets[2,1,0]


    # # Fill in L/R offsets - columns only
    # center_relative_offsets[:,0,1] += -1 * left_right_offsets[:,1,1]
    # center_relative_offsets[:,2,1] += +1 * left_right_offsets[:,2,1]
    # # Fill in L/R offsets for columns - average for each row
    # center_relative_offsets[:,0,0] += -1 * np.mean(left_right_offsets[:,1,0])
    # center_relative_offsets[:,2,0] += -1 * np.mean(left_right_offsets[:,2,0])
    # # Fill in T/B offsets (rows) - rows only
    # center_relative_offsets[0,:,0] += -1 * top_bottom_offsets[1,1,0]
    # center_relative_offsets[2,:,0] += +1 * top_bottom_offsets[2,1,0]
    # # Fill in T/B offsets (columns) - average 
    # center_relative_offsets[0,:,1] += -1 * np.mean(top_bottom_offsets[1,:,1])
    # center_relative_offsets[2,:,1] += +1 * np.mean(top_bottom_offsets[2,:,1])

    print 'center_relative_offsets'
    print_array(center_relative_offsets)
    print('Done')

    return center_relative_offsets



###############
