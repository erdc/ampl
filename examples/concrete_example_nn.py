'''
    The following is an example of using the AMPL api interface to create a dense neural network using the concrete
    data. The user will need to have created, modified, and placed a yaml file in the correct location for this to work. 
    An example of the folder structure to where the yaml file and the data should be placed can be located within 
    the readme located at the following path: 
        README.md -> Anaconda (Conda) setup -> example-directory-structure
    
    Note: If the user needs/wants to use the api to create a yaml file, run 
        examples/create_config_api_example.py
'''

from ampl import *
from ampl.util import Util
import os

# Change the directory 
filename = 'ampl_config.yml'

# If using an ide, open the AMPL root folder. See the 
# comments at the top of this file for an example directory structure
work_dir = './all_run_dir/concrete_run_dir/'

# This will reset the running directory to concrete_run_dir
os.chdir(work_dir)

# Print the files located at the relative path. This will help with finding the yml 
# file if a user receives file not found while trying to work with relative paths.
#Util.relativePathHelper(work_dir)

config = Configuration(filename)

pipeline = config.create_pipeline_nn()
# run all tests for neural network
pipeline.run_all()

# To test individual steps when running the neural network method
# pipeline.optuna.run()
# pipeline.build.run()
# pipeline.eval.run()
# pipeline.ensemble.run()

