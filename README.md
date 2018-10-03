# automated_sequencing
### pipeline for analyzing in-situ sequencing data

This project is intended to provide a full pipeline for analyzing in situ sequencing data, including different options depending on the imaging resolution (cellular or subcellular).

Authors: C. J. Stoneking, A. Vaughan

The main steps of the pipeline are:

1. stitch image tiles (via ImageJ)
2. take z projections
3. perform rigid registration
4. background subtraction
5. segmentation
6. base-call


We are planning to incorporate options for non-rigid registration via elastix. 

installation instructions for the version with elastix:

1. download the files to a convenient working directory e.g. Users/yourname/automated_sequencing
2. download elastix binaries from http://elastix.isi.uu.nl/download_links.php. Put the elastix folder in the working directory, i.e. you should have something like Users/yourname/automated_sequencing/elastix_macosx64_v4/bin/elastix
3. open elastix_wrapper_script.sh in a text editor and change the variable my_working_path to your working directory. Also change my_elastix_folder if you ended up using a different version of elastix
4. run the python script as root (sudo). There are different options, I use the spyder IDE, using sudo spyder to start it and then running python scripts from within it works well


