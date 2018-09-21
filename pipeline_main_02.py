#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Main function for running the pipeline
"""

import alex_sequencing_utility_01 as seq
import cjs_util_01 as util


from skimage.data import imread
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import subprocess
from skimage.feature import peak_local_max
from matplotlib.colors import LinearSegmentedColormap
import time

#Experiment-specific parameters:
data_path = '/Users/cstoneki/Documents/data/michael_KL_data/slide_23'
n_channels = 5


#Other parameters:
do_overwrite_stitching = 0
do_overwrite_hdf5_file = 0
    
#------------------------------------------------------------------------------
#Settings that we probably don't want to modify

if n_channels==4:
    channel_names = ['G','T','A','C']
    channel_colors = [
                  [0,1,1],  # C
                  [1,0,1],  # M
                  [1,1,0],  # Y
                  [1,1,1] ] # K

elif n_channels==5:
    channel_names = ['GFP', 'G', 'T', 'A', 'C']
    channel_colors = [
                  [0,1,0],  # G
                  [0,1,1],  # C
                  [1,0,1],  # M
                  [1,1,0],  # Y
                  [1,1,1] ] # K



#------------------------------------------------------------------------------

def announce(string=None):
    if string is None:
        string = 'Last run at'
    print string + ' :: ' + time.asctime(time.localtime() )
announce()        

def plot_img_bw(img):
    tmp = np.zeros([img.shape[0], img.shape[1], 3])
    for i in range(3):
        tmp[:,:,i] = img
    plt.imshow(tmp)

def parse_data_tree(start_dir):
    #Alex code
    ##### 0 - Make DATA struct   #####
    """
    Here we generated a nested dict with appropriate information for our analysis.
    
    DATA
        start_dir
        colormaps
        tile_labels
        position_dirs
        positions    - name of the directory for each position
        *positions
            path
            image_names
            tiles
            
            *tile   - name of each tile (ie., name of the image file with .ome.tif stripped)
                path
                fname_stub
                image_file
                data_file
                label
                
    """
    
    DATA = dict()
    DATA['start_dir'] = start_dir
    DATA['n_channels'] = n_channels
    DATA['image_regex'] = '\d{3}_\d{3}\.ome\.tif$'
    DATA['channel_names'] = channel_names
    DATA['cmaps'] = [LinearSegmentedColormap.from_list( 'cmap', ((0,0,0),list(color)), N=255 ) for color in channel_colors]
    DATA['tile_labels'] =  [['002_002', '001_002', '000_002'],  
                            ['002_001', '001_001', '000_001'],
                            ['002_000', '001_000', '000_000']]
    
    # Positions are folders that match a regex like '3x3 Seq_\d*$'
    DATA['positions'] = [tile_dir for tile_dir in sorted(os.listdir(DATA['start_dir'])) if re.search('2x2 Seq_\d*$|3x3 Seq_\d*$',tile_dir) is not None]
    # Sort DATA['positions']
    def get_position(a):
        # Returns an INT version of the number after the last '_' in a string
        return int(a[len(a) - a[::-1].find('_'):])
    DATA['positions'] = sorted(DATA['positions'],key=get_position)
    
    for pos in DATA['positions']:
        print pos + ': ',
        DATA[pos] = dict()
        DATA[pos]['path'] = os.path.join(DATA['start_dir'],pos)
        DATA[pos]['image_names'] = [fname for fname in os.listdir(DATA[pos]['path']) if re.search(DATA['image_regex'],fname) is not None]
        DATA[pos]['tiles']       = [path[:-8] for path in DATA[pos]['image_names']]
        DATA[pos]['fname_stub']  = DATA[pos]['tiles'][0][:-7] + 'stitched'
        DATA[pos]['stitched_image_fname']  = DATA[pos]['fname_stub'] + '.ome.tif'
        DATA[pos]['data_fname']  = DATA[pos]['fname_stub'] + '.hdf5'
        
        for tile in DATA[pos]['tiles']:
            print tile[-7:],
            DATA[pos][tile] = dict()
            DATA[pos][tile]['path']        = DATA[pos]['path']
            DATA[pos][tile]['fname_stub']  = tile
            DATA[pos][tile]['image_fname'] = DATA[pos][tile]['fname_stub'] + '.ome.tif'
            DATA[pos][tile]['data_fname']  = DATA[pos][tile]['fname_stub'] + '.hdf5'
            DATA[pos][tile]['label']       = DATA[pos][tile]['fname_stub'][-7:]
        print ''
    # # For reading you can iterate like this
    # for pp in DATA['positions']:
    #     for tt in DATA[pp]['tiles']:
    #         tile = DATA[pp][tt]    
    #         print tile['label']


    announce('Set up DATA')    
    return DATA



def run_stitching(DATA):

    
    for pos in DATA['positions']:
        path = DATA[pos]['path']
        for tile in [tile for tile in DATA[pos]['tiles'] if '001_001' in tile]: # Pick a single file
    
            image_fname = DATA[pos][tile]['image_fname']
            input_file  = os.path.join(path, image_fname)
            assert os.path.isfile(input_file), 'Couldn''t find file %s' % (input_file)
            output_file = os.path.join(path,image_fname.replace('001_001','stitched'))
            
            # Set up stitching macro
            cmd1 = 'run("Grid/Collection stitching", "type=[Positions from file] order=[Defined by image metadata] browse=[%s] multi_series_file=[%s] fusion_method=[Linear Blending] regression_threshold=0.30 max/avg_displacement_threshold=2.50 absolute_displacement_threshold=3.50 add_tiles_as_rois increase_overlap=0 invert_x invert_y display_fusion computation_parameters=[Save memory (but be slower)] image_output=[Fuse and display]");' % (input_file,input_file)
            cmd2 = 'run("Save", "save=[%s]");' % (output_file)
            cmd3 = 'close();'
            f = open(os.path.join(path,'stitching_macro.ijm'),'w')
            f.write(cmd1)
            f.write(cmd2)
            f.write(cmd3)
            f.close()
    
            # Assemble final bash command
            fiji_path = '/Applications/Fiji.app/Contents/MacOS/ImageJ-macosx'
            macro_path = os.path.join(path,'stitching_macro.ijm')
            bash_command = '%s --headless -macro "%s"' % (fiji_path,macro_path)
            
            # Run stitching
            if do_overwrite_stitching or not (os.path.isfile(output_file)):
                print '\nRunning: ' + bash_command
                output = subprocess.Popen(bash_command, shell=True, stdout=subprocess.PIPE).stdout.read()
                skipped = False
            else:
                print '\nSKIPPING - ' + bash_command
                output = ''
                skipped = True
    
            if 'Finished ...' in output:
                print ('\tSuccess')
            elif skipped:
                print ('Keeping previously stitched image')
            else:
                print(output + '\n' + '*** FAILURE ***')
                raise Exception('Stitching failed - please debug.')
    
def make_hdf5_from_stitched(DATA, n_channels=5):
    #### 2 - MAKE HDF5 files from STITCHED IMAGES ####
# Shouldn't repeat without setting parameters

    reload(seq)
    
    # Reorder dimensions to be C,Z,Y,X
    dimension_order = [1,0,2,3]
    
    for pos in DATA['positions']:
        
        # Find path to stitched Image
        stitched_image_path = os.path.join(DATA[pos]['path'],DATA[pos]['stitched_image_fname'])
        assert os.path.isfile(stitched_image_path),'Could not find stitched image at path\n\t%s' % (stitched_image_path)
    
        if True:
            # Load stitched image
            print('\tSaving data into <%s>' % DATA[pos]['data_fname'])
            data = seq.import_micromanager_tiff(stitched_image_path,dimension_order=dimension_order,n_channels=DATA['n_channels'])
            
            # Save into hdf5 file.
            seq.save_tile_data(DATA[pos]['path'],DATA[pos]['data_fname'],data,do_overwrite=do_overwrite_hdf5_file,verbose=False)        
    
        else:
            print('\t Not resaving.')
        print('\t... done')
    
        #break # Break after first sequence position
    
    announce('\nData saved')
    
    
def take_max_proj(DATA):
    

    for pos in DATA['positions']:
        
        image = seq.load_hdf5_data( DATA[pos]['path'] , DATA[pos]['data_fname'] , 'image' , verbose=False )
        image_max_proj = np.squeeze(np.max(image,axis=1))
        seq.save_tile_data(DATA[pos]['path'],DATA[pos]['data_fname'],{'image_max_proj':image_max_proj},do_overwrite=True,verbose=False)
        
    announce('Max projections saved')
        
    
def segment(DATA, mode='cells'):
    #run initial segmentation on data:
    #
    #for each channel and 
    for pp in range(len(DATA['positions'])):
        img = seq.load_hdf5_data( DATA[pp]['path'] , DATA[pp]['data_fname'] , 'image' , verbose=False )
        
    return 0
        
def register_rigid_on_GFP(DATA):
    #after taking max projection
    #register all successive frames based on GFP
    
    import rigid_registration_03 as rigidreg
    
    assert DATA['n_channels']==5, 'Tried to do registration on GFP, this only works if we have 5 channels'
    
    for k, pos in enumerate(DATA['positions']):
        
        img = seq.load_hdf5_data( DATA[pos]['path'] , DATA[pos]['data_fname'] , 'image_max_proj' , verbose=False )

        if(k==0):
            #take the first image as the reference
            reference_img = img
            
            img_to_save = reference_img
            print('\n\t Registering with reference : <%s> ' % DATA[pos]['data_fname'])
        else:
            deformed_img = img
            
            transform = rigidreg.rigid_registration(reference_img[0,:,:], deformed_img[0,:,:])
            print('\n\t Registered image <%s>' % DATA[pos]['data_fname'])
            print('\n\t Angle = %d'%transform['angle'])
            print('\n\t Shift = (%d, %d)'%tuple(transform['shift']))
            for c in range(deformed_img.shape[0]):
                deformed_img[c,:,:] = util.rotate_and_shift(deformed_img[c,:,:],angle=transform['angle'], shift=transform['shift'])
            
            img_to_save = deformed_img
            
        seq.save_tile_data(DATA[pos]['path'],DATA[pos]['data_fname'],{'image_reg':img_to_save},do_overwrite=True,verbose=False)
            
         
    
    
def register_RANSAC_01(DATA):
    
    #### Registration via ransac
    reload(seq)
    import ICP
    reload(ICP)
    
        
    offset = np.zeros(len(DATA['positions']))
    for pp in range(len(DATA['positions'])-1):
        
        pos1 = DATA['positions'][pp+1]
        pos2 = DATA['positions'][pp+2]
        print pp
        # Fixed Image
        if pp == 0:
            fixed_image = seq.load_hdf5_data( DATA[pos1]['path'] , DATA[pos1]['data_fname'] , 'image' , verbose=False )
        else:
            fixed_image = moving_image_registered # From last round.
    
        # Moving image
        moving_image = seq.load_hdf5_data( DATA[pos2]['path'] , DATA[pos2]['data_fname'] , 'image' , verbose=False )
        #moving_image = np.copy(fixed_image)
        
        # Remove_backgorund
        remove_background_params = {'pipeline':['gaussian_hp','gaussian_lp','rectify','subtract_min'],
                                      'hp_sigmas':[3,5,5],'lp_sigmas':[1,1,1]}
        fixed_image = seq.remove_background(fixed_image,params=remove_background_params,verbose=True)
        moving_image = seq.remove_background(moving_image,params=remove_background_params,verbose=True)
    
        # Sum across channels
        fixed_image = np.sum(fixed_image,axis=0)
        moving_image = np.sum(moving_image,axis=0)
        
        # Downsample?
        fixed_image = fixed_image[:,::4,::4]
        moving_image = moving_image[:,::4,::4]
        
        # INTENTIONALLY DEFLECT
        d = None
        if d is not None:
            moving_image = moving_image[:,:-d//2,:-d//2]
            fixed_image = fixed_image[:,d//2:,d//2:]
        
        print fixed_image.shape
        print moving_image.shape
       
        # Find Peaks?
        print('Finding peaks')
        num_peaks = 150
        min_distance = 5
        peaks_fixed = peak_local_max(fixed_image,min_distance=min_distance,num_peaks=num_peaks,exclude_border=0)
        peaks_moving = peak_local_max(moving_image,min_distance=min_distance,num_peaks=num_peaks,exclude_border=0)
    
        # Run Ransac ICP
        print('Running ICP')
        T,d,i = icp.icp(peaks_moving,peaks_fixed,ransac_percentile=50,max_iterations=1000)
        peaks_moved = np.ones((peaks_moving.shape[0],4))
        peaks_moved[:,:3] = peaks_moving
        peaks_moved = np.dot(T,peaks_moved.T).T
    
        for row in T:
            for col in row:
                print('% .3f\t'% (col)),
            print('')
    
        fig = plt.figure(figsize=(10,5),dpi=600)
        plt.subplot(3,1,1)
        plt.imshow(np.sqrt(np.sum(fixed_image,axis=0)))
        plt.plot(peaks_fixed[:,2],peaks_fixed[:,1],'wo',markersize=2,markeredgewidth=0.1,markeredgecolor='w', markerfacecolor='None')
    
        plt.subplot(3,1,2)
        plt.imshow(np.sqrt(np.sum(moving_image,axis=0)))
        plt.plot(peaks_moving[:,2],peaks_moving[:,1],'wo',markersize=3, markeredgewidth=0.1,markeredgecolor='y', markerfacecolor='None')
    
        plt.subplot(3,1,3)
        plt.imshow(np.sqrt(np.sum(fixed_image,axis=0)))
        plt.plot(peaks_fixed[:,2],peaks_fixed[:,1],'wo',markersize=2, markeredgewidth=0.1,markeredgecolor='w', markerfacecolor='None')
        plt.plot(peaks_moved[:,2],peaks_moved[:,1],'wo',markersize=3, markeredgewidth=0.1,markeredgecolor='r', markerfacecolor='None')
    
    
    
        break
    

    
if __name__ == "__main__":
    
    DATA = parse_data_tree(data_path)
    run_stitching(DATA)
    make_hdf5_from_stitched(DATA)
    take_max_proj(DATA)
    register_rigid_on_GFP(DATA)



    