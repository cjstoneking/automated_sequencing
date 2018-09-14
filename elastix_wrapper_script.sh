#!/bin/bash
#wrapper script for elastix
#use this because making system-dependent modifications here is cleaner than making them in the python code

my_working_path="/Users/cstoneki/Documents/analysis/AutomatedSequencing"
my_elastix_folder="elastix_macosx64_v4"
#change as needed

export PATH=$my_working_path/$my_elastix_folder/bin:$PATH
export DYLD_LIBRARY_PATH=$my_working_path/$my_elastix_folder/lib:$DYLD_LIBRARY_PATH
elastix -m $my_working_path/elastix_temp_files/input/im1.mhd -f $my_working_path/elastix_temp_files/input/im2.mhd -out $my_working_path/elastix_temp_files/output -p $my_working_path/elastix_temp_files/input/params.txt

