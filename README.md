# An Automated Machine Learning Pipeline (AMPL)

* Why are we looking at NNs and ML models
* A set of steps are defined for the process of training a NN and ML models
* Name the steps and mention briefly what those steps are
* Describe how the results are displayed with the percentage of points under a 20,10,5,2.5,0.5
* With the steps defined, improving the workflow to allow for both non-technical and technical users.
* After improvements were made, the next goal was to allow for automating the process of training a network.

Machine Learning techniques have risen in popularity as ML has shown to be useful in providing an expert level response to predicting values, recognizing patterns, and identifying objects in images. While working through applying ML to ["SYNTHETIC CFD ESTIMATION FOR BLACKHAWK AIRFOIL DRAG COEFFICIENT"](https://doi.org/10.2514/6.2024-1230), and ["ESTIMATING KINETIC ENERGY REDUCTION FOR TERMINAL BALLISTICS"](https://link.springer.com/article/10.1007/s00521-023-09382-3), it was noted that the steps for applying ML was similar enough to where a single workflow could be designed to handle both of these problems along with previously unexplored problem spaces. The steps that were taken for both the rotorcraft and ballistics problem were Feature Importance, Hyperparameter optimization which is searching for superior model parameters, training the best models returned from hyperparameter optimization, and evaluating the performance of the best models. This documentation will describe the details of each of the previously mentioned steps. Details will also be provided on how to utilize each of the steps individually or as an automated workflow. In this documentation we will describe an automated machine learning pipeline and present results of applying the pipeline to sample problems.

The workflow steps used to design the models used for predicting values for the Ballistics and Rotorcraft work were the same steps. Since the Ballistics work and the rotorcraft work use very different data, but the steps in the workflow were the same, a general workflow that could design ML models for many different problems was desired. Having a general method would reduce effort in the beginning stages of working on a new problem or dataset and allow for exploration of methods and techniques to create better models. The general method would also remove the need to implement each step from ground up and would improve the timeline from receiving the data to having a reasonably performing model. The method for the general workflow is called "An Automated Machine Learning Pipeline" and the method will fit the following criteria:

* Easy to get started for novice machine learning users
* Have sufficient options for expert users
* The tools used in each step must be able to be modifiable or replaceable to allow improvements to the pipeline
* Able to run each step in the workflow as a solo component
* Eliminate the need to manually design ML model to allow more time to be focused on adding new or enhancing existing Steps in the automated machine learning pipeline

## Steps to run AMPL

1. [Anaconda (Conda) setup](#anaconda-conda-setup)
2. [Install AMPL](#install-ampl)
   * [GitLab setup](./README_git.md?ref_type=heads#gitlab-setup)
      * [Option 1: Get SSH key](./README_git.md?ref_type=heads#option-1-get-ssh-key-recommended)
      * [Option 2: Get HTTPS Access Token](./README_git.md?ref_type=heads#option-2-get-https-access-token-needed-to-clone-the-repo-as-well-as-pullpush)
   * [Clone AMPL repository](#clone-ampl-repository)
   * [Install AMPL in your conda environment](#install-ampl)
      * [Initial Steps for Running AMPL](#initial-steps-for-running-ampl)
3. [AMPL setup and Configuration file](#ampl-setup-and-configuration-file)
    * [Suggested directory structure for organizing AMPL](#suggested-directory-structure-for-organizing-AMPL)
4. [Two Methods for Interfering with AMPL](#Two-Methods-for-Interfering-with-AMPL---API,-CLI)
    * [AMPL - API](#ampl-api)
    * [AMPL - CLI](#ampl-cli)

## Anaconda (Conda) setup

Please download and install Anaconda python if it has not previously been installed. Installation instructions can be found within [README_anaconda](./README_anaconda.md?ref_type=heads#install-anaconda) if needed by the user.

### Anaconda Setup

It is important to maintain an up-to-date version of Anaconda. Even if a user already has Anaconda, please follow the steps for Updating Conda and Anaconda. Not having an updated version of onda is the most commonly experienced error by new users.

1. Update Conda and Anaconda (Recommended for all users)

```shell
# Update the conda package manager to the latest version in your base environment
conda update -n base conda -y
# Use conda to update Anaconda to the latest version in your base environment
conda update -n base anaconda -y
conda activate    
# [You should then see (base).]
```

2. Create a new conda environment named `ampl` and install the libraries and modules required for AMPL.

Note: If the user experiences an error during these installs, the most common source for that error is the need to update conda and anaconda as seen in the previous step.

```shell
conda create -n ampl python=3.11 pandas numpy yaml scikit-learn jupyter recommonmark scikit-learn-intelex plotly::plotly anaconda::sphinx -y

conda install -n ampl -c conda-forge optuna matplotlib imbalanced-learn sphinx-gallery cloud_sptheme -y

conda activate ampl

python -m pip install --upgrade pip myst-parser joblib

pip install shap pillow requests xgboost jinja2 more_itertools optuna-integration tensorflow
```

## Setting up gitlab

The user will need to setup their SSH key to git in order to pull the source code for AMPL. This step can be skipped by users who have already setup their git SSH key. This process for setting up the git SSH key will be the same whether a user is running locally or on HPC as a user will need to store an SSH key for each machine they use to access git. Instructions for setting up git are found here within the [README_git] (./README_git.md?ref_type=heads#GitLab-setup)

### Creating AMPL Code and Run/working Directory - for both API and CLI users

The directory structure for working with AMPL is the same regardless of using the API or CLI method to interface with AMPL. This recommended directory structure will help with organization and determining where to put the AMPL repository, as well as provide a convenient way to organize the directory structure so that ML models and information (plots, statistics, etc..) are easily accessed by the user. A user can use any directory structure they would like, but will have to customize their yaml input file to account for any differences between their chosen directory structure and the recommended directory structure.

#### Recommended directory structure

The user may use multiple methods for creating a directory structure, however, the example below assumes the user is working from commandline. For the example working with AMPL, we will use an open-source concrete dataset. When a user is working with their own data they should rename any references to 'concrete' with their own dataset name.

1. Create a parent folder for AMPL Code

```shell
# ex: $ mkdir <ampl_dir> 
mkdir AMPL
```

2. Move into the <ampl_dir>

```shell
# ex: cd <ampl_dir> 
cd AMPL 
```

3. Create AMPL repository code directory:

```shell
# whatever you would like to name your designated AMPL repository code directory 
# ex: mkdir <ampl_code_dir>
mkdir code
```

4. Create directory for AMPL runs:

```shell
# whatever you would like to name your directory to hold all of the AMPL runs.
# ex: mkdir <all_run_dir> 
mkdir all_run_dir
```

5. Move into the <all_run_dir>

```shell
# ex: cd <all_run_dir> 
cd all_run_dir
```

6. Create directory for a specific run of AMPL:

```shell
# whatever you would like to name your designated AMPL working directory based on the data being used.
# here we create concrete_run_dir for holding information for the concrete data example. Please name
# concrete to something different based on the data you are working with.
# ex: mkdir <dataset#_run_dir>
mkdir concrete_run_dir
```

Note: Repeat step six to create the structure to maintain the data for different data sets and runs. Example directory structure is provided below.

#### Example Directory Structure

The following is an example directory structure of what AMPL will look like when the code is downloaded and a user is working with multiple datasets. This example shows where the ampl code is located, and provides two examples of where the run data is stored. The first example the user has a csv file and uses one yaml file for specifying how AMPL will run. The second example the user has a sqlite data file and two different yaml files for specifying how AMPL with run when working with the same dataset. AMPL will accept both csv and sqlite data, however, the user will need to modify the yaml file based on which type of data is being used. Examples for working with csv or sqlite are provided below.

Note: When working with AMPL, the normal use case is to create a Folder named 'AMPL' to store everything. This folder is the root folder and should be the folder a user opens when working with AMPL whether through the [API](#ampl-api) or through the [command-line option](ampl-cli).

```shell
├── <ampl_dir>
│   ├── <ampl_code_dir>
│   │   ├── <ampl>
│   │   │   ├──<docs>
│   │   │   ├──<examples>
│   │   │   ├──<src>
│   │   │   ├──<tests> 
│   ├── <all_run_dir>
│   │   ├── <dataset1_run_dir>
│   │   │   ├── <data_dir>
│   │   │   │   ├── <data1.csv>
│   │   │   ├── <config_file.yml>
│   │   ├── <dataset2_run_dir>
│   │   │   ├── <data_dir>
│   │   │   │   ├── <data2.sqlite>
│   │   │   ├── <config_file1.yml>
│   │   │   ├── <config_file2.yml>
```

### Clone AMPL repository

1. Navigate to your AMPL repository code directory <ampl_code_dir> created in the previous step [Recommended directory structure](#recommended-directory-structure):

```shell
# ex: cd <ampl_dir>/<ampl_code_dir>
cd code

# If following along with the examples the path would be
# cd ../code
  ├── ampl_dir
->│   ├── code
  │   ├── all_run_dir

```

4. Go to the AMPL Git repo page in your browser, select Code and then there will be two options to `Clone`. Copy the URL based on whether you are using the SSH Key or HTTPS option (SSH is recommended). Use the following command to clone the repo into your AMPL working directory:

```shell
# example using ssh:
# ex: git clone <copied URL>

git clone git@github.com:erdc/ampl.git

```

Note: If you don't have access or are getting a permission error from Git, please refer to [Option 1: Get SSH key](./README_git.md?ref_type=heads#option-1-get-ssh-key-recommended)

#### Install AMPL in your conda environment

 Note for Windows users: Please open Anaconda power shell as an administrator for the commands to work.

1. Activate the conda environment you created previously while installing Anaconda.

```shell
# ex: conda activate <env_name>
conda activate ampl 
```

2. Install the needed packages.

```shell

# Navigate to the inside the code folder. 
  ├── AMPL
->│   ├── code

# ex: pip install -e <ampl_code_dir>
pip install -e ampl
```

If the install was successful then you will see soemthing similiar to:

```shell
   Successfully built ampl
   Installing collected packages: ampl
   Successfully installed ampl-0.0.3
```

Test AMPL in your env by running the following:

```shell
# 1: Make sure you are in the right conda environment, i.e. the ampl environment

# ex: conda activate <env_name>
conda activate ampl

# 2: Run some tests that are part of the ampl code by first navigating to the tests directory
# ex: cd <ampl_code_dir>/tests
cd ampl/tests
 
# 3: Test the pipeline connections
python -m unittest test_pipeline_nn

```

The command above that tests the pipeline connection does a small test to confirm if the connections are set up properly.  It starts with using Optuna to find the best trial and then runs 10 epochs to train that best trial.  After it is done running, it displays the results of the test in a table that includes the layer type, output shape, and number of parameters. Below the table are more details about the parameters that are used in the test run.

### AMPL setup and Configuration file

#### Initial Steps for Running AMPL

AMPL utilizes configuration file(s) to setup a study/run.

When AMPL is executed, a .yaml file must be passed in as an argument for running in either CLI or API mode. The .yaml file contains a path to the location of the input data, which is used for training, validation, testing, as well as the path to the results directory. The location of these directories are relative to the aforementioned path provided in the .yaml in addition to many other configuration settings.

[Default Config File](./src/ampl/default_config.yml)

#### Suggested directory structure for organizing AMPL

You will want to create a directory for your specific AMPL run to keep your runs organized, especially if you will be running AMPL on multiple datasets. This is where all your project related folders should be created and where your ampl_config.yml should reside. We recommend creating a separate folder to store data within it and refer to it in the config file.

1. Navigate to your AMPL running directory.  The run directory is typically located in the home directory <ampl_run_dir>, previously refered to as AMPL/all_run_dir in examples:

```shell
# ex: cd <ampl_dir>/<ampl_run_dir>
# from the AMPL directory
cd all_run_dir

# If following along from the example the path will be
# cd ../../../all_run_dir

# The folder structure will look like this
  ├── ampl_dir
  │   ├── code
->│   ├── all_run_dir
  
```

2. Create a directory for your run and navigate to this directory.  Use a name that identifies the dataset you will be using. Since the example that we will be using is based on a concrete dataset, we will name the directory appropriately:

```shell
# ex: mkdir <dataset_run_dir>
mkdir concrete_run_dir

# The folder structure will look like this
  ├── ampl_dir
  │   ├── code
  │   ├── all_run_dir
->│   │   ├── concrete_run_dir

```

3. Move into the <concrete_run_dir> directory:

```shell
# ex: cd <dataset_run_dir> 
cd concrete_run_dir

# The current folder structure
  ├── ampl_dir
  │   ├── code
  │   ├── all_run_dir
->│   │   ├── concrete_run_dir

```

4. Create a directory to hold your data:

```shell
# ex: mdir <data_dir>
mkdir concrete_data

# The current folder structure
  ├── AMPL
  │   ├── code
  │   ├── all_run_dir
  │   │   ├── concrete_run_dir
->│   │   │   ├── concrete_data

```

5. Copy your dataset to the <data_dir> directory. Please have the data in a SQLite or a CSV file type.  Most datasets can easily be converted into CSV format, but be sure that the index column is not included.

As an example data set moving forward, we will be utilizing a public concrete data set. Copy the "concrete.csv" file from "AMPL\code\ampl\tests\data\concrete.csv"  to the concrete_data folder created in the previous step. The below folder structure shows where the concrete.csv data file should go.

```shell
    # the current folder structure  
    ├── AMPL
    │   ├── code
    │   ├── all_run_dir
    │   │   ├── concrete_run_dir
    │   │   │   ├── concrete_data
  ->│   │   │   │   ├── concrete.csv
```

A direct link can be found to the example concrete dataset at the following link:
  <https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength>

  Yeh,I-Cheng. (2007). Concrete Compressive Strength. UCI Machine Learning Repository. <https://doi.org/10.24432/C5PK67>.

6. Create a default configuration file:

  Run the following code from within the <concrete_run_dir> directory in the terminal. The following command is using the CLI method of interfacing with ampl.

  ```shell

  python -m ampl ampl_config.yml -c

  # the current folder structure  
    ├── AMPL
    │   ├── code
    │   ├── all_run_dir
    │   │   ├── concrete_run_dir
    │   │   │   ├── concrete_data
    │   │   │   │   ├── concrete.csv
  ->│   │   │   ├── ampl_config.yml
  ```

7. Edit the `ampl_config.yml` file by filling in all the required fields.

The following block of code provides a description of the variables that require user modification in the ampl_config.yml. A table is provided later in this step that shows the edits to the yaml file for the example using the concrete dataset. When using a different dataset, please refer back to the table in this section as a quick reference for the variables you will need to modify when applying ampl to a different dataset.

```shell
# A name that describes what the study is about and needs to be unique.
study_name: 'your_study_name'  

# The column that you are trying to predict
target_variable: 'your_target_variable'

# Name of the data     
dataset_name: 'your_dataset_name'

# The path to the SQLITE data file, set to NULL if using a CSV file. 
data_file: null 

# If the data file is SQLite, the data_table_name needs to be the name of the table you wish to use for the study from the SQLite data file.  This is null if using CSV data.
data_table_name: null 

# The path to the CSV file, set to null for SQLite.
csv_file: 'path_to_CSV'

# The path where it will save the normalized version of the CSV file.
# It is not mandatory to normalize the data, but it is recommended. 
# If using a sqlite input file this line should be set to null
csv_normalized_file : 'path_to_CSV' + '_normalized'

# If you have any columns with categorical data, this enumerates them to numerical data.
# Set to null if there is no categorical data
cols_to_enum:  null 

# Include all of the possible features (columns) to use.  When listing out columns, make sure to not add the column from the target_variable(s).
feature_list: 'col_1'
              'col_2'
              'col_3'
              'col_4'
              'col_5'
              'col_6'
    
# The number of columns used as features in the dataset, not including the target variable
number_of_features: 6 
```

The following table shows all of the variables in the `ampl_config.yml` file that require modification, their default values, and an example modification to work with the concrete data set example.

| Variable           | Default Variables         | Concrete Variables                                                   |
| :----------------- | :------------------------ | :----- |
| study_name         | 'your_study_name'         | 'compress_strength' |
| target_variable    | 'your_target_variable'    | 'Concrete compressive strength' |
| dataset_name       | 'your_dataset_name'       | 'strength_estimate' |
| data_file          | null                      | null |
| data_table_name    | null                      | null |
| csv_file           | 'path_to_CSV'             | 'concrete_data/concrete.csv' |
| cols_to_enum       | null                      | null |
| feature_list       | null                      | - 'Cement' - 'Blast Furnace Slag' - 'Fly Ash' - 'Water' - 'Superplasticizer' - 'Coarse Aggregate' - 'Fine Aggregate' - 'Age' |
| number_of_features | 6                         | 8 |

Most of the fields in the table above are a copy and paste into the ampl_config.yml file. However, the feature list requires a little editing. The following is how the feature_list using the concrete example should look in the yaml file

```shell
# The following is how the feature_list should look in the yml file
feature_importance:
  feature_list:  # REQUIRED USER MODIFICATION # List of columns to include in study/run, don't include any target columns
    - 'Cement' 
    - 'Blast Furnace Slag' 
    - 'Fly Ash'
    - 'Water'
    - 'Superplasticizer'
    - 'Coarse Aggregate' 
    - 'Fine Aggregate'
    - 'Age'
```

This is what the directory structure looks like after using the examples provided in each of the previous steps in this guide and after running the concrete example code:

```shell
├── AMPL
│   ├── code
│   ├── all_run_dir
│   │   ├── concrete_run_dir
│   │   │   ├── concrete_data
│   │   │   │   ├── concrete.csv
│   │   │   ├── ampl_config.yml
│   │   │   ├── results_compress_strength
│   │   │   │   ├── plots
│   │   │   │   ├── saved_models
```

### Two Methods for Interfering with AMPL - API, CLI

AMPL provides two different ways to interface with it. One is through the AMPL API (Application Programming Interface) and the other is via AMPL CLI (Command Line Interface). The API will be helpful to users wishing to use scripts to run AMPL, and the CLI will be helpful for those wishing to run AMPL without writing scripts. An example of using the CLI would be running on HPC. Both the AMPL API and the AMPL CLI will be described in detail in their respective sections below.

#### AMPL API

The AMPL API will provide the user with the means to interface with AMPL through scripts. This type of interface is useful for creating scripts to run AMPL as part of a workflow. In the next section we will show an example of using a script that uses the AMPL API.

##### Getting started API

The following script will use AMPL to create a fully dense neural network using the previously created [folder structure](#Suggested-directory-structure-for-organizing-AMPL), and example downloaded concrete.csv dataset. This script is available in the examples folder under [examples/concrete_example.py]("./examples/concrete_example.py")

```python
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
pipeline.run_all()

```

The following is an example using python to create a yaml file. You will still need to edit the yaml file with the information for the run. The lines within the yaml file that require modification are listed in an example within the [Suggested directory structure for organizing AMPL](Suggested-directory-structure-for-organizing-AMPL) section. This script is available in the examples folder under [examples/create_config_api_example.py]("./examples/create_config_api_example.py")

```python
from ampl import *
from ampl.util import Util
import os
# Project should be run 

# change the directory 
filename = 'ampl_config.yml'

# Open the root AMPL project in your ide for running this code
# An example of the folder structure can be located within the readme 
# README.md -> AMPL Interface -> Creating AMPL Code and Run/working Directory - for both API and CLI users -> Example Directory Structure

# use the concrete_run_dir from the root folder of AMPL
work_dir = './all_run_dir/concrete_run_dir/'

# this will reset the running directory to concrete_run_dir
os.chdir(work_dir)

# Create default config file
Util.create_default_config_file(filename)

```

The next example shown here is for both Neural networks and Decision Trees. The following code is an example of what to do if your dataset contains columns that are used to create a target column. In the below example we use the starting_velcoity cubed minus the ending_velocity cubed to populate the values within the target column for every row within the dataframe. The commented lines

```python
import ampl

# Ignore this step if the dataset contains the target column already.
# Here the 'starting_Velocity(m/s)' and 'ending_Velocity(m/s)' are two columns that are already part of the dataset and are used
# to create the target column 
target_func = lambda df_: df_['starting_Velocity(m/s)'].astype('float') ** 3 - df_['ending_Velocity(m/s)'].astype('float') ** 3
                            
config = ampl.Configuration('data/pipeline_config.yml', target_col_function=target_func)

# Creating Neural Network AMPL
pipeline_nn = config.create_pipeline_nn()

pipeline_nn.optuna.run()
# Getting Best Trial and Top Trials from Optuna Study
best_trial_df, top_trials_df = pipeline_nn.optuna.load_trials_df() 
print(best_trial_df)
print(top_trials_df)
pipeline_nn.build.run()
pipeline_nn.eval.run()

# only a few methods for ensembling are implented at this time. 
#pipeline_nn.ensemble.run()

 
# Creating Decision Tree AMPL - If the user would like to try decision trees instead of neural networks, uncomment the lines below
# pipeline_dt = config.create_pipeline_dt()

# pipeline_dt.optuna.run()
# pipeline_dt.build.run()
# pipeline_dt.eval.run()
# pipeline_dt.ensemble.run()

```

#### AMPL CLI

The AMPL CLI will provide an easy to use means for calling AMPL functionality through the command line. The CLI is helpful when the user doesn't want to write scripts to call AMPL, but wants to call the functions of AMPL. An example application of using the CLI would be to run AMPL on HPC.

##### Getting started CLI

Once AMPL is installed in your conda env, use a terminal to run AMPL.

AMPL CLI help - To display the AMPL CLI help use the following command

```shell
python -m ampl -h
```

It will display similar text as shown below, this text may differ as the application matures

```text
usage: FAIT AMPL package [-h] [-d] [-c] [-dt] [-o] [-b] [-ev] [-en]
                         config_file

Process AMPL Pipeline Configuration YAML file.

positional arguments:
  config_file           Path to YAML Configuration file

optional arguments:
  -h, --help            show this help message and exit
  -d, --debug           Print debug info
  -c, --create_config   Creates a new default YAML Configuration file
  -dt, --decision_tree  Use Decision Tree instead of Neural Networks
  -o, --optuna          Run Optuna Model step
  -b, --build           Run Build Model step
  -ev, --evaluate       Run Evaluate Model step
  -en, --ensemble       Run Ensemble Model step
```

##### Run AMPL using CLI

From the terminal run the following commands. The examples assume that steps were taken to create the following directory structure:

├── AMPL
│   ├── code
│   ├── all_run_dir
│   │   ├── concrete_run_dir
│   │   │   ├── concrete_data
│   │   │   │   ├── concrete.csv
│   │   │   ├── ampl_config.yml

```shell
conda activate ampl
cd concrete_run_dir #cd <dataset1_run_dir>
python -m ampl ampl_config.yml 
```

However, if you would like to run a single step within the pipeline, you can use the following commands instead. For example, to just run the build and evaluate steps, you would use the following AMPL-CLI command with the 'build' and 'evaluate' command line parameters. Additional command line parameters are located in the [Getting started CLI](#Getting-started-CLI) section:

```shell
cd concrete_run_dir #cd <dataset1_run_dir>
python -m ampl ampl_config.yml -b -ev
```
