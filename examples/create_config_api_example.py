'''
    The following is an example of using the AMPL api interface to create a yaml file
    
    An example of the folder structure to where the yaml file and the data should be placed can be located within 
    the readme located at the following path: 
        README.md -> Anaconda (Conda) setup -> example-directory-structure    
'''

from ampl import *
from ampl.util import Util
import os


# change the directory 
filename = 'ampl_config.yml'

# Open the root AMPL project in your ide for running this code
# An example of the folder structure can be located within the readme 
#   README.md -> Anaconda (Conda) setup -> example-directory-structure

# use the dataset1_run_dir from the root folder of AMPL
work_dir = './all_run_dir/dataset1_run_dir/'

# this will reset the running directory to dataset1_run_dir
os.chdir(work_dir)

# Create default config file
Util.create_default_config_file(filename)

