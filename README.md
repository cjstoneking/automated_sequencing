# automated_sequencing
pipeline for analyzing in-situ seq data

This was built on Mac OS X. It will hopefully work on linux with minor modifications. It won't work on Windows in its current state.

installation instructions:

1. download the files to a convenient working directory e.g. Users/yourname/automated_sequencing
2. download elastix binaries from http://elastix.isi.uu.nl/download_links.php. Put the elastix folder in the working directory, i.e. you should have something like Users/yourname/automated_sequencing/elastix_macosx64_v4/bin/elastix
3. open elastix_wrapper_script.sh in a text editor and change the variable my_working_path to your working directory. Also change my_elastix_folder if you ended up using a different version of elastix
4. run the python script as root (sudo). There are different options, I use the spyder IDE, using sudo spyder to start it and then running python scripts from within it works well


